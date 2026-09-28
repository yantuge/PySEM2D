import numpy as np
import time
import os

try:
    from .utils import find_nearest_node_indices
except ImportError:
    from utils import find_nearest_node_indices


class SEMSolver:
    """
    核心求解器
    """
    def __init__(self, config, mesh, material, bc, source, receivers, assembler, visualizer):
        self.p = config
        self.mesh = mesh
        self.material = material
        self.bc = bc
        self.source = source
        self.receivers = receivers
        self.assembler = assembler
        self.visualizer = visualizer
        self.integrator = None  # 时间积分器策略

    def assemble_matrices(self, use_gpu=False):
        """
        委托 SEMAssembler 组装全局矩阵
        
        参数:
            use_gpu: bool, 刚度矩阵组装是否使用GPU加速
        """
        print("Starting to assemble the global matrix...")
        
        # 定位所有震源节点, 震源信息会在 integrator 初始化时打印显示
        self.src_node_indices = find_nearest_node_indices(self.source.locations, self.mesh.global_nodes)
        
        # 矩阵存储在 assembler.matrices 字典中
        self.assembler.assemble_matrices(use_gpu=use_gpu)
        
        print("Global matrix assembly complete.")

    def _create_physics_model(self):
        """
        根据配置创建物理模型
        
        返回:
            PhysicsModelBase: 物理模型实例
        """
        # 从 config['physics_type'] 读取
        physics_type = self.p.get('physics_type', 'elastic_isotropic').lower()
        
        print(f"\n  Creating physics model: {physics_type}")
        
        if physics_type == 'elastic_isotropic':
            from .physics.elastic import IsotropicElastic
            physics_model = IsotropicElastic(self.material, self.bc)
            print(f"    [OK] Physics model: {physics_model.get_model_name()}")
            return physics_model
        elif physics_type == 'elastic_anisotropic':
            from .physics.elastic import AnisotropicElastic
            physics_model = AnisotropicElastic(self.material, self.bc)
            print(f"    [OK] Physics model: {physics_model.get_model_name()}")
            return physics_model
        else:
            raise ValueError(
                f"Unknown physics_type: '{physics_type}'. "
                f"Supported types: 'elastic_isotropic', 'elastic_anisotropic'"
            )
    
    def _create_compute_backend(self):
        """
        根据配置创建计算后端
        
        返回:
            ComputeBackendBase: 计算后端实例
        """
        # 当前只支持 Taichi 后端
        print(f"  Creating compute backend: Taichi")
        
        from .backend import TaichiBackend
        
        # 准备完整的配置
        backend_config = {
            'nt': self.p['nt'],
            'dt': self.p['dt'],
        }
        
        # 获取 Taichi 特定配置
        taichi_config = self.p.get('taichi_config', {})
        if not taichi_config:
            # 尝试从 solver 配置中获取
            taichi_config = self.p.get('solver', {}).get('taichi', {})
        
        # 合并配置
        backend_config.update(taichi_config)
        
        backend = TaichiBackend(backend_config)
        print(f"    [OK] Compute backend: {backend.get_backend_name()}")
        return backend
    
    def run(self):
        """
        根据配置创建并运行时间积分器
        """
        solver_type = self.p.get('solver_type', 'cpu').lower()
        
        print("\n" + "="*70)
        print(f"CREATING TIME INTEGRATOR")
        print("="*70)
        print(f"  Solver type: {solver_type.upper()}")
        
        # 如果启用实时可视化, 创建可视化策略
        viz_strategy = None
        if self.p.get('enable_realtime_viz', False):
            try:
                from .visualization import TaichiGUIVisualizer
                # 只需要指定高度，宽度会根据模型宽高比自动计算
                viz_strategy = TaichiGUIVisualizer(
                    self.mesh, self.p,
                    height=self.p.get('viz_height', 800),
                    colormap=self.p.get('viz_colormap', 'jet')
                )
                print("  Visualization strategy: Taichi GUI")
            except Exception as e:
                print(f"  WARNING: Failed to create Taichi GUI visualizer: {e}")
                print("  Real-time visualization will be disabled")
        
        # 1. 创建物理模型
        physics_model = self._create_physics_model()
        
        # 2. 创建计算后端
        compute_backend = self._create_compute_backend()
        
        # 3. 创建积分器
        from .integrators import Integrator
        print(f"  Creating integrator: Integrator")
        
        self.integrator = Integrator(
            config=self.p,
            mesh=self.mesh,
            source=self.source,
            receivers=self.receivers,
            visualizer=self.visualizer,
            physics_model=physics_model,
            compute_backend=compute_backend,
            assembler=self.assembler,
            viz_strategy=viz_strategy
        )
        print(f"    [OK] Integrator created successfully")
        
        # 委托运行
        print("="*70)
        self.integrator.run()
        
        print("\n" + "="*70)
        print("SOLVER COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")