"""
时间积分器抽象基类
"""

from abc import ABC, abstractmethod
import time
import os
import numpy as np


class TimeIntegratorBase(ABC):
    """
    时间积分器抽象基类
    """
    
    def __init__(self, config, mesh, source, receivers, 
                 visualizer, viz_strategy=None,
                 physics_model=None, compute_backend=None, 
                 material=None, assembler=None):
        """
        初始化时间积分器
        
        参数:
            config: dict, 配置参数字典
            mesh: Mesh, 网格对象
            source: Source, 震源对象
            receivers: Receivers, 接收器对象
            visualizer: Visualizer, 可视化器对象
            assembler: AssemblerBase, 组装器对象
            viz_strategy: VisualizationStrategy, 可视化策略（可选）
            physics_model: PhysicsModelBase, 物理模型（必需）
            compute_backend: ComputeBackendBase, 计算后端（必需）
            material: Material, 材料对象（可选）

        """
        self.p = config
        self.mesh = mesh
        self.source = source
        self.receivers = receivers
        self.visualizer = visualizer
        self.viz_strategy = viz_strategy
        
        # 架构参数
        self.physics_model = physics_model
        self.compute_backend = compute_backend
        
        # 其他参数
        self.material = material
        self.assembler = assembler
        
        # 计算震源节点索引
        from ..utils import find_nearest_node_indices
        self.src_node_indices = find_nearest_node_indices(
            self.source.locations, self.mesh.global_nodes
        )
        
        print(f"\nSource Node Indices: {self.src_node_indices}")
        print(f"Number of sources: {len(self.src_node_indices)}")
        for i, idx in enumerate(self.src_node_indices):
            print(f"  Source {i+1}: Node {idx} at position {self.mesh.global_nodes[idx]}")
    
    def run(self):
        """
        模板方法：执行完整的时间步进循环
        """
        # 步骤0: 检查快速模式
        fast_mode = self.p.get('fast_mode', False)
        
        # 快速模式下自动禁用实时可视化
        if fast_mode:
            if self.p.get('enable_realtime_viz', False):
                print("\nWARNING: Real-time visualization is automatically disabled in fast mode")
                self.p['enable_realtime_viz'] = False
        
        # 步骤1: 初始化波场和GPU内核
        print("\n" + "="*70)
        print("INITIALIZING TIME INTEGRATOR")
        print("="*70)
        self.initialize_fields()
        
        # 步骤2: 根据模式分发执行
        if fast_mode:
            # 快速模式：调用子类的快速执行方法
            if hasattr(self, 'run_fast_mode'):
                self.run_fast_mode()
            else:
                raise NotImplementedError(
                    f"{self.__class__.__name__} does not support fast mode. "
                    f"Please set fast_mode=false in config or implement run_fast_mode()."
                )
        else:
            # 逐步模式：使用原有的时间循环
            self._run_step_by_step_mode()
    
    def _run_step_by_step_mode(self):
        """
        逐步模式：原有的时间步进循环, 支持实时可视化
        """
        # 读取配置参数
        dt = self.p['dt']
        nt = self.p['nt']
        
        # 可视化配置
        enable_realtime_viz = self.p.get('enable_realtime_viz', False)
        viz_interval = self.p.get('viz_interval', 500)
        viz_field_type = self.p.get('viz_field_type', 1)
        
        # 进度输出配置
        progress_print_interval = self.p.get('progress_print_interval', 100)
        
        # 快照输出配置
        enable_snapshot = self.p.get('enable_snapshot', False)
        snapshot_interval = self.p.get('snapshot_interval', 100)
        snapshot_format = self.p.get('snapshot_format', 'vtu')
        snapshot_fields = self.p.get('snapshot_field', [1])
        async_snapshot = self.p.get('async_snapshot', True)
        
        # 步骤2: 初始化可视化
        if enable_realtime_viz and self.viz_strategy:
            print("\nInitializing visualization...")
            self.viz_strategy.initialize()
            if self.viz_strategy.is_running():
                print("  Real-time visualization enabled")
            else:
                print("  Real-time visualization initialization failed, disabled")
                enable_realtime_viz = False
        
        # 步骤3: 初始化快照输出
        if enable_snapshot:
            self._initialize_snapshot_output(
                snapshot_format, snapshot_fields, 
                snapshot_interval, async_snapshot
            )
        
        # 步骤4: 时间循环
        print("\n" + "="*70)
        print("STARTING TIME LOOP (STEP-BY-STEP MODE)")
        print("="*70)
        print(f"  Total time steps: {nt}")
        print(f"  Time step size: {dt}s")
        print(f"  Total duration: {nt * dt:.2f}s")
        print(f"  Progress print interval: {progress_print_interval} steps")
        if enable_realtime_viz:
            print(f"  Visualization: Enabled (interval: {viz_interval} steps)")
        else:
            print(f"  Visualization: Disabled")
        print("")
        
        self.start_time = time.time()  # 存储为实例变量，供_visualize_wavefield使用
        start_time = self.start_time
        geo_bounds = [self.p['xmin'], self.p['xmax'], self.p['zmin'], self.p['zmax']]
        
        for it in range(nt):
            current_time = it * dt
            
            # 实时可视化
            visualized = False
            if enable_realtime_viz and it > 0 and it % viz_interval == 0:
                if self.viz_strategy and self.viz_strategy.is_running():
                    self._visualize_wavefield(it, current_time, viz_field_type)
                    visualized = True
                else:
                    # GUI已关闭，禁用后续可视化
                    enable_realtime_viz = False
            
            # 进度输出
            if not visualized and it > 0 and it % progress_print_interval == 0:
                elapsed = time.time() - start_time
                self._print_progress(it, nt, current_time, elapsed)
            
            # 执行一步物理计算
            self.step_physics(it)
            
            # 添加震源
            self.add_sources(it)
            
            # 计算速度和加速度
            self.compute_derivatives(it)
            
            # 记录接收器数据
            self.record_receivers(it)
            
            # 输出波场快照
            if enable_snapshot and it % snapshot_interval == 0 and it >= self.first_snapshot_step:
                self._save_snapshot(it, current_time)
        
        # 步骤5: 完成计算
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n" + "="*70)
        print("TIME LOOP COMPLETED")
        print("="*70)
        print(f"  Total duration: {total_time:.2f}s")
        print(f"  Average time per step: {total_time/nt*1000:.2f}ms")
        print("")
        
        # 步骤6: 清理可视化
        if self.viz_strategy:
            self.viz_strategy.finalize()
        
        # 步骤7: 传输结果到CPU
        print("Finalizing results...")
        self.finalize_results()
        
        # 步骤8: 保存地震图
        if self.p.get('save_seismograms', True):
            self._save_seismograms()
        
        # 步骤9: 等待异步I/O完成
        if enable_snapshot:
            self._finalize_snapshot_output()
    
    # ==================== 抽象方法 ====================
    
    @abstractmethod
    def initialize_fields(self):
        """
        初始化波场变量和计算内核
        """
        pass
    
    @abstractmethod
    def update_displacement_terms(self):
        """
        更新位移的分解项
        """
        pass
    
    @abstractmethod
    def update_auxiliary_fields(self):
        """
        更新辅助场（p_x, p_z等）
        """
        pass
    
    @abstractmethod
    def update_history_and_combine(self):
        """
        更新历史项并合成总位移
        """
        pass
    
    @abstractmethod
    def add_sources(self, it):
        """
        添加震源贡献
        
        参数:
            it: int, 当前时间步
        """
        pass
    
    @abstractmethod
    def compute_derivatives(self, it):
        """
        计算速度和加速度场
        
        参数:
            it: int, 当前时间步
        """
        pass
    
    @abstractmethod
    def record_receivers(self, it):
        """
        记录接收器数据（位移、速度、加速度）
        
        参数:
            it: int, 当前时间步
        """
        pass
    
    @abstractmethod
    def get_wavefields_for_viz(self, field_type):
        """
        获取用于可视化的波场数据
        
        参数:
            field_type: int, 波场类型 (1=位移, 2=速度, 3=加速度)
        
        返回:
            波场数据（Taichi field或NumPy数组）
        """
        pass
    
    @abstractmethod
    def finalize_results(self):
        """
        完成计算, 传输结果到CPU端接收器
        """
        pass
    
    # ==================== 默认实现 ====================
    
    def step_physics(self, it):
        """
        执行一步物理计算
        """
        self.update_displacement_terms()
        self.update_auxiliary_fields()
        self.update_history_and_combine()
    
    # ==================== 辅助方法 ====================
    
    def _print_progress(self, it, nt, current_time, elapsed):
        """打印进度信息"""
        print(f" Time Step: {it}/{nt}, Time: {current_time:.3f}s, "
              f"Elapsed: {elapsed:.2f}s")
    
    def _visualize_wavefield(self, it, current_time, field_type):
        """实时可视化波场"""
        if not self.viz_strategy or not self.viz_strategy.is_running():
            return
        
        # 获取波场数据
        wavefield = self.get_wavefields_for_viz(field_type)
        
        # 渲染
        max_amp = self.viz_strategy.render(wavefield, current_time, it, field_type)
        
        # 打印完整的进度信息
        if max_amp is not None:
            elapsed = time.time() - self.start_time
            print(f" Time Step: {it}/{self.p['nt']}, Time: {current_time:.3f}s, "
                  f"Elapsed: {elapsed:.2f}s, Max Amp: {max_amp:.4e}")
        
        return max_amp
    
    def _initialize_snapshot_output(self, snapshot_format, snapshot_fields, 
                                    snapshot_interval, async_snapshot):
        """初始化快照输出"""
        from ..io import DataIO
        
        if async_snapshot:
            DataIO.init_async_io(max_workers=2)
        
        self.snapshot_dir = os.path.join(self.visualizer.output_dir, 'snapshots')
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_files = []
        self.snapshot_times = []
        
        # 确保数据可用性, 计算第一个快照的时间步
        max_field_requirement = max(
            [2 if f == 2 else 3 if f == 3 else 0 for f in snapshot_fields] + [0]
        )
        self.first_snapshot_step = max(snapshot_interval, max_field_requirement)
        
        print(f"\n  Snapshot output: Enabled (async={'ON' if async_snapshot else 'OFF'})")
        print(f"  Snapshot interval: every {snapshot_interval} steps")
        print(f"  Snapshot format: {snapshot_format}")
        print(f"  Snapshot fields: {snapshot_fields} (1=disp, 2=vel, 3=acc)")
        print(f"  First snapshot at step: {self.first_snapshot_step}")
    
    def _save_snapshot(self, it, current_time):
        """保存波场快照"""
        from ..io import DataIO
        
        snapshot_fields = self.p.get('snapshot_field', [1])
        snapshot_format = self.p.get('snapshot_format', 'vtu')
        async_snapshot = self.p.get('async_snapshot', True)
        
        # 根据snapshot_fields获取波场数据
        disp_data = None
        vel_data = None
        acc_data = None
        
        if 1 in snapshot_fields:
            disp_field = self.get_wavefields_for_viz(1)
            disp_data = disp_field.to_numpy() if hasattr(disp_field, 'to_numpy') else disp_field
        
        if 2 in snapshot_fields:
            vel_field = self.get_wavefields_for_viz(2)
            vel_data = vel_field.to_numpy() if hasattr(vel_field, 'to_numpy') else vel_field
        
        if 3 in snapshot_fields:
            acc_field = self.get_wavefields_for_viz(3)
            acc_data = acc_field.to_numpy() if hasattr(acc_field, 'to_numpy') else acc_field
        
        # 写入快照文件
        snapshot_base = os.path.join(self.snapshot_dir, f'snapshot_{it:05d}')
        
        vtu_file = DataIO.write_snapshot_data(
            file_path=snapshot_base,
            mesh=self.mesh,
            displacement=disp_data,
            velocity=vel_data,
            acceleration=acc_data,
            time=current_time,
            timestep=it,
            snapshot_fields=snapshot_fields,
            file_format=snapshot_format,
            async_write=async_snapshot
        )
        
        if vtu_file:
            self.snapshot_files.append(vtu_file)
            self.snapshot_times.append(current_time)
    
    def _finalize_snapshot_output(self):
        """完成快照输出，生成PVD集合文件"""
        from ..io import DataIO
        
        async_snapshot = self.p.get('async_snapshot', True)
        
        if async_snapshot:
            # print("\nWaiting for async I/O to complete...")
            DataIO.wait_async_io()
        
        if len(self.snapshot_files) > 0:
            pvd_file = os.path.join(self.snapshot_dir, 'snapshots.pvd')
            DataIO.write_pvd_collection(pvd_file, self.snapshot_files, self.snapshot_times)
            print(f"  PVD collection file created: {pvd_file}")
    
    def _save_seismograms(self):
        """保存地震图文件"""
        from ..io import DataIO
        
        print("\nSaving seismograms...")
        
        # 绘制地震图
        self.visualizer.plot_seismograms(self.receivers, self.p)
        
        # 导出CSV格式
        DataIO.export_all_seismograms(
            self.receivers,
            self.visualizer.output_dir,
            self.p['dt'],
            self.p['nt']
        )
        
        print("  Seismograms saved successfully")
