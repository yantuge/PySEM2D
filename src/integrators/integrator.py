"""
时间积分器
"""

from .base import TimeIntegratorBase
import numpy as np


class Integrator(TimeIntegratorBase):
    """
    时间积分器
    """
    
    def __init__(self, config, mesh, source, receivers, 
                 visualizer, physics_model, compute_backend, assembler,
                 viz_strategy=None):
        """
        初始化积分器
        
        参数:
            config: dict, 配置参数
            mesh: Mesh, 网格对象
            source: Source, 震源对象
            receivers: Receivers, 接收器对象
            visualizer: Visualizer, 可视化器
            physics_model: PhysicsModelBase, 物理模型
            compute_backend: ComputeBackendBase, 计算后端
            assembler: Assembler, 矩阵组装器
            viz_strategy: VisualizationStrategy, 可视化策略（可选）
        """
        # 调用父类构造函数
        super().__init__(
            config=config,
            mesh=mesh,
            source=source,
            receivers=receivers,
            visualizer=visualizer,
            viz_strategy=viz_strategy,
            physics_model=physics_model,
            compute_backend=compute_backend,
            assembler=assembler  # 仍需要assembler获取矩阵
        )
        
        # 检查物理模型是否适配
        self._validate_physics_model()
        
        # 震源节点索引由父类 BaseIntegrator.__init__ 计算和打印
        # 此处复用 self.src_node_indices
    
    def _validate_physics_model(self):
        """
        验证物理模型是否适用于PML积分
        """
        # 检查波场维度
        field_dim = self.physics_model.get_field_dimension()
        if field_dim != 2:
            raise ValueError(
                f"Integrator expects 2D field dimension, got {field_dim}"
            )
        
        # 检查辅助场（PML需要）
        aux_fields = self.physics_model.get_auxiliary_field_names()
        if len(aux_fields) == 0:
            print("  Warning: No auxiliary fields defined. PML may not work correctly.")
        
        print(f"  Physics model validated: {self.physics_model.get_model_name()}")
        print(f"  Field dimension: {field_dim}")
        print(f"  Auxiliary fields: {aux_fields}")
    
    def initialize_fields(self):
        """
        初始化波场变量和计算后端
        """
        num_nodes = self.mesh.num_nodes
        num_receivers = len(self.receivers.indices)
        nt = self.p['nt']
        dt = self.p['dt']
        
        # 检查是否启用快速模式
        fast_mode = self.p.get('fast_mode', False)
        
        print(f"  Number of nodes: {num_nodes}")
        print(f"  Number of receivers: {num_receivers}")
        print(f"  Time steps: {nt}")
        print(f"  Fast mode: {'Enabled' if fast_mode else 'Disabled'}")
        
        # 1. 初始化计算后端
        print(f"\n  Initializing compute backend: {self.compute_backend.get_backend_name()}...")
        
        # 快速模式需要额外参数
        if fast_mode:
            num_sources = len(self.src_node_indices)
            
            # 计算快照数量
            enable_snapshot = self.p.get('enable_snapshot', False)
            snapshot_interval = self.p.get('snapshot_interval', 1000)
            if enable_snapshot and snapshot_interval > 0:
                num_snapshots = (nt // snapshot_interval) + 1
            else:
                num_snapshots = 0
            
            print(f"  Fast mode parameters:")
            print(f"    - Number of sources: {num_sources}")
            print(f"    - Number of snapshots: {num_snapshots}")
            
            # 初始化后端（快速模式）
            self.compute_backend.initialize(num_nodes, num_receivers, nt, 
                                           fast_mode=True, 
                                           num_sources=num_sources,
                                           num_snapshots=num_snapshots)
        else:
            # 初始化后端（逐步模式）
            self.compute_backend.initialize(num_nodes, num_receivers, nt)
        
        # 2. 预计算时间积分常数
        print("  Computing time integration constants...")
        a0 = 1 / dt**2
        a1 = 1 / (2 * dt)
        a2 = 2 * a0
        
        # 保存常数
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        
        # 3. 从assembler通过physics_model获取质量和阻尼矩阵
        if not hasattr(self.assembler, 'matrices'):
            raise RuntimeError(
                "Assembler does not have 'matrices' attribute. "
                "Please ensure SEMSolver.assemble_matrices() has been called"
            )
        
        # 使用physics_model的接口提取质量和阻尼矩阵
        print("  Extracting mass and damping matrices from physics model...")
        mass_damping = self.physics_model.get_mass_damping_matrices(self.assembler.matrices)
        
        M = mass_damping['M']
        Cx = mass_damping['Cx']
        Cz = mass_damping['Cz']
        Cxx = mass_damping['Cxx']
        Cxz = mass_damping['Cxz']
        Czz = mass_damping['Czz']
        
        if self.p.get('use_pml_or_not') and hasattr(self.compute_backend, 'kernels'):
            x, z = self.mesh.global_nodes.T
            fixed = np.zeros(M.size, dtype=bool)
            for coord, damping, bound in ((x, Cx, 'xmin'), (x, Cx, 'xmax'),
                                          (z, Cz, 'zmin'), (z, Cz, 'zmax')):
                fixed |= (damping > 0) & np.isclose(coord, self.p[bound], rtol=0, atol=1e-8)
            self.compute_backend.kernels.fixed_outer.from_numpy(fixed.astype(np.int32))

        # 4. 计算逆矩阵
        print("  Computing inverse matrices...")
        A1 = a0 * M + a1 * 2 * Cx
        A2 = a0 * M + a1 * (Cx + Cz)
        A3 = a0 * M + a1 * 2 * Cz
        AL = 2 * a1 * M
        
        A1_inv = 1.0 / A1
        A2_inv = 1.0 / A2
        A3_inv = 1.0 / A3
        AL_inv = 1.0 / AL
        
        # 5. 加载矩阵到计算后端
        # 使用TaichiBackend的特定方法
        print("  Loading matrices to compute backend...")
        
        # 对于Taichi后端，使用load_diagonal_matrices和load_sparse_matrices
        if hasattr(self.compute_backend, 'load_diagonal_matrices'):
            self.compute_backend.load_diagonal_matrices(
                M, Cx, Cz, Cxx, Cxz, Czz,
                A1_inv, A2_inv, A3_inv, AL_inv
            )
        else:
            raise NotImplementedError(
                f"Backend {self.compute_backend.get_backend_name()} does not support load_diagonal_matrices"
            )
        
        # 使用physics_model的接口提取刚度矩阵
        print("  Extracting stiffness matrices from physics model...")
        stiffness = self.physics_model.get_stiffness_matrices(self.assembler.matrices)
        
        if hasattr(self.compute_backend, 'load_sparse_matrices'):
            self.compute_backend.load_sparse_matrices(
                stiffness['Kx1'], stiffness['Kx2'], stiffness['Kx3'],
                stiffness['Kx4'], stiffness['Kx5'],
                stiffness['Kz1'], stiffness['Kz2'], stiffness['Kz3'],
                stiffness['Kz4'], stiffness['Kz5']
            )
        else:
            raise NotImplementedError(
                f"Backend {self.compute_backend.get_backend_name()} does not support load_sparse_matrices"
            )
        
        # 6. 加载接收器索引
        if hasattr(self.compute_backend, 'load_receivers'):
            print("  Loading receiver indices...")
            self.compute_backend.load_receivers(self.receivers.indices)
        
        # 7. 预计算时间积分常数
        if hasattr(self.compute_backend, 'kernels'):
            print("  Precomputing time integration coefficients...")
            self.compute_backend.kernels.precompute_coefficients(self.a0, self.a1, self.a2)
        
        # 8. 初始化矩震源
        if self.source.source_type == 'moment_tensor':
            from ..gll import GLL
            gll = GLL(self.p['P'])
            self.source.initialize_moment_tensor_sources(
                self.mesh, self.assembler, gll
            )
            # 加载矩震源数据到后端
            if hasattr(self.compute_backend, 'load_moment_tensor_sources'):
                self.compute_backend.load_moment_tensor_sources(
                    self.source.source_arrays,
                    self.source.source_elements,
                    self.mesh.connectivity
                )
                
        print(f"  Initialization complete")
    
    def update_displacement_terms(self):
        """
        更新位移的三个分解项
        """
        backend = self.compute_backend
        
        if hasattr(backend, 'kernels'):
            # Taichi后端
            gpu = backend.kernels
            
            # 一次性缓存位移分量，避免多次提取
            gpu.cache_displacement_components()
            
            # 获取稀疏矩阵引用
            Kx1 = gpu.sparse_matrices['Kx1']
            Kx2 = gpu.sparse_matrices['Kx2']
            Kx3 = gpu.sparse_matrices['Kx3']
            Kz1 = gpu.sparse_matrices['Kz1']
            Kz2 = gpu.sparse_matrices['Kz2']
            Kz3 = gpu.sparse_matrices['Kz3']
            
            # 将所有矩阵乘法结果存储到专用buffer，然后调用融合kernel
            # 这样避免了buffer覆盖问题，同时减少了kernel启动次数

            # # X方向的矩阵乘法
            # gpu.sparse_matvec(
            #     Kx1.data, Kx1.indices, Kx1.indptr,
            #     gpu.displ_x, gpu.temp_Kx1, Kx1.nrows
            # )
            # gpu.sparse_matvec(
            #     Kx2.data, Kx2.indices, Kx2.indptr,
            #     gpu.displ_z, gpu.temp_Kx2, Kx2.nrows
            # )
            # gpu.sparse_matvec(
            #     Kx3.data, Kx3.indices, Kx3.indptr,
            #     gpu.displ_x, gpu.temp_Kx3, Kx3.nrows
            # )
            # 
            # # Z方向的矩阵乘法（复用temp_result和temp_vec）
            # gpu.sparse_matvec(
            #     Kz1.data, Kz1.indices, Kz1.indptr,
            #     gpu.displ_z, gpu.temp_result, Kz1.nrows
            # )
            # gpu.sparse_matvec(
            #     Kz2.data, Kz2.indices, Kz2.indptr,
            #     gpu.displ_x, gpu.temp_vec, Kz2.nrows
            # )
            # gpu.sparse_matvec(
            #     Kz3.data, Kz3.indices, Kz3.indptr,
            #     gpu.displ_z, gpu.temp_Kz3, Kz3.nrows
            # )
            
            # 10次sparse_matvec调用变为2次融合调用
            gpu.sparse_matvec_6x_fused(
                Kx1.data, Kx1.indices, Kx1.indptr,
                Kx2.data, Kx2.indices, Kx2.indptr,
                Kx3.data, Kx3.indices, Kx3.indptr,
                Kz1.data, Kz1.indices, Kz1.indptr,
                Kz2.data, Kz2.indices, Kz2.indptr,
                Kz3.data, Kz3.indices, Kz3.indptr,
                gpu.displ_x, gpu.displ_z,
                gpu.temp_Kx1, gpu.temp_Kx2, gpu.temp_Kx3,
                gpu.temp_result, gpu.temp_vec, gpu.temp_Kz3,
                Kx1.nrows
            )
            
            # 一次性更新X方向的3个term
            gpu.update_displ_terms_x_fused(
                gpu.temp_Kx1, gpu.temp_Kx2, gpu.temp_Kx3
            )
            
            # 一次性更新Z方向的3个term
            gpu.update_displ_terms_z_fused(
                gpu.temp_result, gpu.temp_vec, gpu.temp_Kz3
            )
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support PML displacement update"
            )
    
    def update_auxiliary_fields(self):
        """
        更新PML辅助场
        """
        backend = self.compute_backend
        
        if hasattr(backend, 'kernels'):
            # Taichi后端
            gpu = backend.kernels
            
            Kx4 = gpu.sparse_matrices['Kx4']
            Kx5 = gpu.sparse_matrices['Kx5']
            Kz4 = gpu.sparse_matrices['Kz4']
            Kz5 = gpu.sparse_matrices['Kz5']
            
            # # 先完成所有4个矩阵乘法，然后调用融合kernel
            # gpu.sparse_matvec(
            #     Kx4.data, Kx4.indices, Kx4.indptr,
            #     gpu.displ_x, gpu.temp_Kx1, Kx4.nrows
            # )
            # gpu.sparse_matvec(
            #     Kx5.data, Kx5.indices, Kx5.indptr,
            #     gpu.displ_x, gpu.temp_Kx2, Kx5.nrows
            # )
            # gpu.sparse_matvec(
            #     Kz4.data, Kz4.indices, Kz4.indptr,
            #     gpu.displ_z, gpu.temp_result, Kz4.nrows
            # )
            # gpu.sparse_matvec(
            #     Kz5.data, Kz5.indices, Kz5.indptr,
            #     gpu.displ_z, gpu.temp_vec, Kz5.nrows
            # )
            
            gpu.sparse_matvec_4x_fused(
                Kx4.data, Kx4.indices, Kx4.indptr,
                Kx5.data, Kx5.indices, Kx5.indptr,
                Kz4.data, Kz4.indices, Kz4.indptr,
                Kz5.data, Kz5.indices, Kz5.indptr,
                gpu.displ_x, gpu.displ_z,
                gpu.temp_Kx1, gpu.temp_Kx2,
                gpu.temp_result, gpu.temp_vec,
                Kx4.nrows
            )
            
            #一次性更新所有4个p场分量
            gpu.update_p_fields_fused(
                gpu.temp_Kx1, gpu.temp_Kx2,
                gpu.temp_result, gpu.temp_vec
            )
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support auxiliary field update"
            )
    
    def update_history_and_combine(self):
        """
        更新历史项并合成总位移
        """
        backend = self.compute_backend
        
        if hasattr(backend, 'kernels'):
            backend.kernels.update_history_and_combine()
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support history update"
            )
    
    def add_sources(self, it):
        """
        添加震源贡献（力源）
        
        参数:
            it: int, 当前时间步
        """
        backend = self.compute_backend
        dt = self.p['dt']
        
        if self.source.source_type == 'force':
            # 力源：单点施加
            for src_idx_num, src_node_idx in enumerate(self.src_node_indices):
                angle_rad = self.source.angles[src_idx_num]
                sin_angle = np.sin(angle_rad)
                cos_angle = np.cos(angle_rad)
                
                stf_val = self.source.get_stf_at_time(src_idx_num, it)
                
                if hasattr(backend, 'kernels'):
                    backend.kernels.add_source(src_node_idx, stf_val, sin_angle, cos_angle, dt)
                else:
                    raise NotImplementedError(
                        f"Backend {backend.get_backend_name()} does not support add_source"
                    )
        
        elif self.source.source_type == 'moment_tensor':
            # 矩震源：单元级别施加
            for src_idx_num in range(self.source.num_sources):
                stf_val = self.source.get_stf_at_time(src_idx_num, it)
                
                if stf_val == 0.0:
                    continue
                
                # 获取震源数组和所在单元
                source_array = self.source.get_source_array(src_idx_num)
                ispec = self.source.get_source_element(src_idx_num)
                
                if hasattr(backend, 'kernels'):
                    backend.kernels.add_moment_tensor_source(
                        ispec, source_array, stf_val, dt
                    )
                else:
                    raise NotImplementedError(
                        f"Backend {backend.get_backend_name()} does not support add_moment_tensor_source"
                    )
    
    def compute_derivatives(self, it):
        """
        计算速度和加速度场
        
        参数:
            it: int, 当前时间步
        """
        dt = self.p['dt']
        backend = self.compute_backend
        
        if hasattr(backend, 'kernels'):
            # 计算速度（it >= 1）
            if it >= 1:
                backend.kernels.compute_velocity(dt)
            
            # 计算加速度（it >= 2）
            if it >= 2:
                backend.kernels.compute_acceleration(dt)
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support derivative computation"
            )
    
    def record_receivers(self, it):
        """
        记录接收器数据
        
        参数:
            it: int, 当前时间步
        """
        backend = self.compute_backend
        
        if hasattr(backend, 'kernels'):
            # 记录位移
            backend.kernels.record_seismograms_disp(it)
            
            # 记录速度（it >= 1）
            if it >= 1:
                backend.kernels.record_seismograms_vel(it)
            
            # 记录加速度（it >= 2）
            if it >= 2:
                backend.kernels.record_seismograms_acc(it)
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support seismogram recording"
            )
    
    def get_wavefields_for_viz(self, field_type):
        """
        获取用于可视化的波场数据
        
        参数:
            field_type: int, 波场类型
                1 = 位移
                2 = 速度
                3 = 加速度
        
        返回:
            波场数据（Taichi field或NumPy数组）
        """
        backend = self.compute_backend
        
        if hasattr(backend, 'kernels'):
            if field_type == 2:  # velocity
                return backend.kernels.velocity
            elif field_type == 3:  # acceleration
                return backend.kernels.acceleration
            else:  # displacement (default)
                return backend.kernels.displ_now
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support wavefield visualization"
            )
    
    def finalize_results(self):
        """
        完成计算，将结果传输到CPU端接收器
        """
        print("  Transferring seismogram data from backend to receivers...")
        
        backend = self.compute_backend
        
        if hasattr(backend, 'kernels'):
            # 传输地震图数据
            self.receivers.displacement = backend.kernels.get_seismograms_displacement()
            self.receivers.velocity = backend.kernels.get_seismograms_velocity()
            self.receivers.acceleration = backend.kernels.get_seismograms_acceleration()
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support seismogram retrieval"
            )
        
        print("  Data transfer complete")
    
    def run_fast_mode(self):
        """
        快速计算模式：将整个时间循环在GPU中执行
        """
        dt = self.p['dt']
        nt = self.p['nt']
        
        # 快照配置
        enable_snapshot = self.p.get('enable_snapshot', False)
        snapshot_interval = self.p.get('snapshot_interval', 1000)
        snapshot_format = self.p.get('snapshot_format', 'vtu')
        snapshot_fields = self.p.get('snapshot_field', [1, 2, 3])
        
        print("\n" + "="*70)
        print("FAST MODE EXECUTION")
        print("="*70)
        print(f"  Total time steps: {nt}")
        print(f"  Time step size: {dt}s")
        print(f"  Total duration: {nt * dt:.2f}s")
        print(f"  Real-time visualization: Disabled (fast mode)")
        if enable_snapshot:
            print(f"  Snapshot output: Enabled (interval: {snapshot_interval} steps)")
        else:
            print(f"  Snapshot output: Disabled")
        print("")
        
        # 判断是CPU还是GPU后端, 对应后面打印信息
        backend = self.compute_backend.get_backend_name().lower()
        if 'cpu' in backend:
            backen_name = 'CPU'
        else:
            backen_name = 'GPU'
            
        # 步骤1: 预加载震源数据到GPU
        print(f"Preloading source data to {backen_name}...")
        moment_source_step = None
        if self.source.source_type == 'moment_tensor':
            moment_source_step = self.add_sources
        else:
            self._preload_sources_to_gpu()
        
        # 步骤2: 在GPU中执行完整时间循环
        print(f"\nExecuting full time loop on {backen_name}...")
        print("  (This may take a while, no progress output in fast mode)")
        
        import time
        start_time = time.time()
        
        backend = self.compute_backend
        if hasattr(backend, 'kernels'):
            backend.kernels.run_full_time_loop_batch(
                nt, dt, snapshot_interval, enable_snapshot, moment_source_step
            )
        else:
            raise NotImplementedError(
                f"Backend {backend.get_backend_name()} does not support fast mode"
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n" + "="*70)
        print("FAST EXECUTION COMPLETED")
        print("="*70)
        print(f"  Total {backen_name} time: {total_time:.2f}s")
        print(f"  Average time per step: {total_time/nt*1000:.2f}ms")
        print("")
        
        # 步骤3: 传输结果到CPU
        if backen_name == 'GPU':
            print("Transferring results from GPU to CPU...")
        else:
            print("Transferring results from CPU...")
            
        self.finalize_results()
        
        # 步骤4: 批量写入保存快照
        if enable_snapshot:
            print("\nWriting snapshots to disk...")
            self._write_batch_snapshots(snapshot_format, snapshot_fields)
        
        # 步骤5: 保存地震图
        if self.p.get('save_seismograms', True):
            self._save_seismograms()
        
        print("\nFast mode execution finished successfully")
    
    def _preload_sources_to_gpu(self):
        """
        预加载所有震源数据到GPU（快速模式）
        """
        backend = self.compute_backend
        
        if not hasattr(backend, 'kernels') or not backend.kernels.fast_mode:
            raise RuntimeError("_preload_sources_to_gpu can only be called in fast mode")
        
        # 准备震源参数
        num_sources = len(self.src_node_indices)
        src_node_indices = np.array(self.src_node_indices, dtype=np.int32)
        
        # 计算角度的sin和cos
        sin_angles = np.zeros(num_sources, dtype=np.float64)
        cos_angles = np.zeros(num_sources, dtype=np.float64)
        time_delays = np.zeros(num_sources, dtype=np.int32)
        
        for i in range(num_sources):
            angle_rad = self.source.angles[i]
            sin_angles[i] = np.sin(angle_rad)
            cos_angles[i] = np.cos(angle_rad)
            time_delays[i] = self.source.time_delays[i]
        
        # 加载震源数据到GPU
        backend.kernels.load_source_data(
            self.source.stf_list,
            src_node_indices,
            sin_angles,
            cos_angles,
            time_delays
        )
        
        print(f"  Loaded {num_sources} sources to GPU")
        for i in range(num_sources):
            stf_len = len(self.source.stf_list[i])
            print(f"    Source {i+1}: Node {src_node_indices[i]}, "
                  f"Angle {self.source.angles[i]*180/np.pi:.1f}°, "
                  f"STF length {stf_len}")
    
    def _write_batch_snapshots(self, snapshot_format, snapshot_fields):
        """
        批量写入快照文件（快速模式）
        """
        from ..io import DataIO
        import os
        
        backend = self.compute_backend
        num_snapshots = backend.kernels.num_snapshots
        
        if num_snapshots == 0:
            print("  No snapshots to write")
            return
        
        # 创建快照目录
        snapshot_dir = os.path.join(self.visualizer.output_dir, 'snapshots')
        os.makedirs(snapshot_dir, exist_ok=True)
        
        snapshot_files = []
        snapshot_times = []
        dt = self.p['dt']
        
        print(f"  Writing {num_snapshots} snapshots...")
        
        # 遍历所有快照
        for snap_idx in range(num_snapshots):
            # 从GPU获取快照数据
            snapshot_data = backend.kernels.get_snapshot_data(snap_idx)
            timestep = snapshot_data['timestep']
            current_time = timestep * dt
            
            # 准备波场数据
            disp_data = snapshot_data['displacement'] if 1 in snapshot_fields else None
            vel_data = snapshot_data['velocity'] if 2 in snapshot_fields else None
            acc_data = snapshot_data['acceleration'] if 3 in snapshot_fields else None
            
            # 写入文件
            snapshot_base = os.path.join(snapshot_dir, f'snapshot_{timestep:05d}')
            
            vtu_file = DataIO.write_snapshot_data(
                file_path=snapshot_base,
                mesh=self.mesh,
                displacement=disp_data,
                velocity=vel_data,
                acceleration=acc_data,
                time=current_time,
                timestep=timestep,
                snapshot_fields=snapshot_fields,
                file_format=snapshot_format,
                async_write=False  # 批量模式下使用同步写入
            )
            
            if vtu_file:
                snapshot_files.append(vtu_file)
                snapshot_times.append(current_time)
            
            if (snap_idx + 1) % 10 == 0 or snap_idx == num_snapshots - 1:
                print(f"    Progress: {snap_idx + 1}/{num_snapshots} snapshots written")
        
        # 生成PVD集合文件
        if len(snapshot_files) > 0:
            pvd_file = os.path.join(snapshot_dir, 'snapshots.pvd')
            DataIO.write_pvd_collection(pvd_file, snapshot_files, snapshot_times)
            print(f"  PVD collection file created: {pvd_file}")
        
        print(f"  All snapshots written successfully")
