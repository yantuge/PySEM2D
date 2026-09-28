"""
Taichi计算后端实现
"""

import numpy as np
from typing import Any, Optional
import sys
import os

# 导入核心抽象
try:
    from core import ComputeBackendBase
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core import ComputeBackendBase

# 导入Taichi内核
try:
    import taichi as ti
    from .taichi_kernels import SEMGPUKernels
    TAICHI_AVAILABLE = True
except ImportError as e:
    TAICHI_AVAILABLE = False
    print(f"Warning: Taichi not available: {e}")
    SEMGPUKernels = None


class TaichiBackend(ComputeBackendBase):
    """
    基于Taichi框架的计算后端
    """
    
    def __init__(self, config: dict):
        """
        初始化Taichi计算后端
        
        参数:
            config: dict, 配置参数字典
                必需: 'nt' (时间步数), 'dt' (时间步长)
                可选: 'backend_type' ('cuda', 'cpu', 'vulkan'等)
        """
        if not TAICHI_AVAILABLE:
            raise RuntimeError(
                "Taichi is not available. Please install: pip install taichi"
            )
        
        super().__init__(config)
        
        # Taichi内核实例, 延迟初始化
        self.kernels = None
        
        # 稀疏矩阵存储, 用于后续访问
        self.sparse_matrices = {}
        
        # 接收器索引
        self.receiver_indices = None
        
        print(f"TaichiBackend initialized (backend: {ti.cfg.arch})")
    
    def initialize(self, num_nodes: int, num_receivers: int, nt: int = None,
                   fast_mode: bool = False, num_sources: int = 0, num_snapshots: int = 0):
        """
        初始化Taichi内核和数据结构
        
        参数:
            num_nodes: int, 全局节点数
            num_receivers: int, 接收器数量
            nt: int, 时间步数（可选，如果不提供则使用self.nt）
            fast_mode: bool, 是否启用快速计算模式
            num_sources: int, 震源数量（快速模式需要）
            num_snapshots: int, 快照数量（快速模式需要）
        """
        if nt is None:
            nt = self.nt
        
        print(f"Initializing Taichi kernels: {num_nodes} nodes, {num_receivers} receivers")
        if fast_mode:
            print(f"  Fast mode enabled: {num_sources} sources, {num_snapshots} snapshots")
        
        self.kernels = SEMGPUKernels(num_nodes, num_receivers, nt, 
                                     fast_mode=fast_mode,
                                     num_sources=num_sources,
                                     num_snapshots=num_snapshots)
        print("Taichi kernels initialized successfully")
    
    # ==================== 内存管理实现 ====================
    
    def allocate_scalar_field(self, num_nodes: int, initial_value: float = 0.0) -> Any:
        """
        分配Taichi field
        
        参数:
            num_nodes: int, 节点数量
            initial_value: float, 初始值
        
        返回:
            ti.field, Taichi标量场

        """
        # 对于Taichi后端，标量场已经在kernels中预分配
        # 这里返回一个占位符或引用, 实际使用中，调用会直接使用kernels中的field
        raise NotImplementedError(
            "TaichiBackend uses pre-allocated fields in SEMGPUKernels. "
            "Access fields directly via backend.kernels.<field_name>"
        )
    
    def allocate_vector_field(self, num_nodes: int, dimension: int, 
                             initial_value: float = 0.0) -> Any:
        """
        分配矢量场（Taichi Vector field）
        
        参数:
            num_nodes: int, 节点数量
            dimension: int, 矢量维度
            initial_value: float, 初始值
        
        返回:
            ti.Vector.field, Taichi矢量场
        """
        # Taichi后端的波场已经预分配
        raise NotImplementedError(
            "TaichiBackend uses pre-allocated fields in SEMGPUKernels. "
            "Access fields directly via backend.kernels.<field_name>"
        )
    
    def allocate_seismogram_storage(self, num_receivers: int, dimension: int) -> Any:
        """
        分配地震图存储 (Taichi field)
        
        参数:
            num_receivers: int, 接收器数量
            dimension: int, 波场维度
        
        返回:
            ti.field, 地震图存储 (num_receivers, dimension, nt)
        """
        # 地震图已经在kernels中分配
        raise NotImplementedError(
            "TaichiBackend uses pre-allocated seismogram fields in SEMGPUKernels. "
            "Access via backend.kernels.seismogram_*"
        )
    
    # ==================== 数值运算实现 ====================
    
    def sparse_matvec(self, matrix_data: Any, matrix_indices: Any, 
                     matrix_indptr: Any, input_vector: Any, 
                     output_vector: Any, nrows: int) -> None:
        """
        稀疏矩阵-向量乘法 (CSR格式)
        
        参数:
            matrix_data: Taichi field, 非零元素值
            matrix_indices: Taichi field, 行索引
            matrix_indptr: Taichi field, 行指针（CSR格式）
            input_vector: Taichi field, 输入向量
            output_vector: Taichi field, 输出向量
            nrows: int, 矩阵行数（CSR格式）
        """
        self.kernels.sparse_matvec(
            matrix_data, matrix_indices, matrix_indptr,
            input_vector, output_vector, nrows
        )
    
    def vector_add(self, vec_a: Any, vec_b: Any, result: Any, 
                   scale_a: float = 1.0, scale_b: float = 1.0) -> None:
        """
        向量加法: result = scale_a * vec_a + scale_b * vec_b
        
        参数:
            vec_a: Taichi field, 向量A
            vec_b: Taichi field, 向量B
            result: Taichi field, 结果
            scale_a: float, A的缩放系数
            scale_b: float, B的缩放系数

        """
        # SEMGPUKernels中应该有相应的kernel
        # 这里假设有一个vector_add kernel
        raise NotImplementedError(
            "vector_add kernel needs to be added to SEMGPUKernels"
        )
    
    def scalar_multiply(self, vector: Any, scalar: float, result: Any) -> None:
        """
        标量-向量乘法: result = scalar * vector
        
        参数:
            vector: Taichi field, 输入向量
            scalar: float, 标量
            result: Taichi field, 结果
        """
        raise NotImplementedError(
            "scalar_multiply kernel needs to be added to SEMGPUKernels"
        )
    
    def copy_field(self, source: Any, destination: Any) -> None:
        """
        复制波场
        
        参数:
            source: Taichi field, 源
            destination: Taichi field, 目标
        """
        # Taichi支持直接赋值: destination.copy_from(source)
        destination.copy_from(source)
    
    # ==================== 数据访问实现 ====================
    
    def get_field_value(self, field: Any, node_index: int) -> np.ndarray:
        """
        获取指定节点的波场值
        
        参数:
            field: Taichi field, 波场
            node_index: int, 节点索引
        
        返回:
            np.ndarray, 该节点的值
        """
        # Taichi field可以通过索引直接访问
        value = field[node_index]
        
        # 如果是Vector field，转换为numpy数组
        if hasattr(value, 'to_numpy'):
            return value.to_numpy()
        else:
            return np.array(value)
    
    def set_field_value(self, field: Any, node_index: int, value: np.ndarray) -> None:
        """
        设置指定节点的波场值
        
        参数:
            field: Taichi field, 波场
            node_index: int, 节点索引
            value: np.ndarray, 要设置的值
        """
        field[node_index] = value
    
    def to_numpy(self, field: Any) -> np.ndarray:
        """
        将Taichi field转换为NumPy数组
        
        参数:
            field: Taichi field, 波场
        
        返回:
            np.ndarray, NumPy数组表示
        """
        if hasattr(field, 'to_numpy'):
            return field.to_numpy()
        else:
            raise TypeError(f"Field type {type(field)} does not support to_numpy()")
    
    # ==================== 地震图记录实现 ====================
    
    def record_seismogram(self, field: Any, receiver_indices: np.ndarray, 
                         seismogram_storage: Any, timestep: int) -> None:
        """
        记录当前时刻的地震图数据
        
        参数:
            field: Taichi Vector field, 当前波场
            receiver_indices: np.ndarray, 接收器节点索引
            seismogram_storage: Taichi field, 地震图存储
            timestep: int, 当前时间步

        """
        # SEMGPUKernels应该有记录地震图的kernel
        # 这里调用kernels的方法
        if hasattr(self.kernels, 'record_seismograms_disp'):
            self.kernels.record_seismograms_disp(timestep)
        else:
            raise NotImplementedError(
                "record_seismograms kernel not found in SEMGPUKernels"
            )
    
    # ==================== 高层接口 ====================
    
    def load_diagonal_matrices(self, M, Cx, Cz, Cxx, Cxz, Czz,
                               A1_inv, A2_inv, A3_inv, AL_inv):
        """
        加载对角矩阵到Taichi fields
        
        参数:
            M, Cx, Cz, Cxx, Cxz, Czz: np.ndarray, 质量和阻尼矩阵
            A1_inv, A2_inv, A3_inv, AL_inv: np.ndarray, 预计算的逆矩阵
        """
        self.kernels.load_diagonal_matrices(
            M, Cx, Cz, Cxx, Cxz, Czz,
            A1_inv, A2_inv, A3_inv, AL_inv
        )
    
    def load_sparse_matrices(self, Kx1, Kx2, Kx3, Kx4, Kx5,
                            Kz1, Kz2, Kz3, Kz4, Kz5):
        """
        加载稀疏刚度矩阵到Taichi
        
        参数:
            Kx1-Kx5, Kz1-Kz5: scipy.sparse.csc_matrix, 刚度矩阵
        """
        self.kernels.load_sparse_matrices(
            Kx1, Kx2, Kx3, Kx4, Kx5,
            Kz1, Kz2, Kz3, Kz4, Kz5
        )
        
        # 保存引用供后续使用
        self.sparse_matrices = {
            'Kx1': self.kernels.sparse_matrices['Kx1'],
            'Kx2': self.kernels.sparse_matrices['Kx2'],
            'Kx3': self.kernels.sparse_matrices['Kx3'],
            'Kx4': self.kernels.sparse_matrices['Kx4'],
            'Kx5': self.kernels.sparse_matrices['Kx5'],
            'Kz1': self.kernels.sparse_matrices['Kz1'],
            'Kz2': self.kernels.sparse_matrices['Kz2'],
            'Kz3': self.kernels.sparse_matrices['Kz3'],
            'Kz4': self.kernels.sparse_matrices['Kz4'],
            'Kz5': self.kernels.sparse_matrices['Kz5'],
        }
    
    def load_receivers(self, receiver_indices: np.ndarray):
        """
        加载接收器索引
        
        参数:
            receiver_indices: np.ndarray, 接收器节点索引
        """
        self.kernels.load_receivers(receiver_indices)
        self.receiver_indices = receiver_indices
    
    def load_moment_tensor_sources(self, source_arrays, source_elements, connectivity):
        """
        加载矩震源数据到Taichi
        
        参数:
            source_arrays: numpy.ndarray, (num_sources, 2, NGLL, NGLL) 的震源分布数组
            source_elements: numpy.ndarray, (num_sources,) 的震源所在单元索引
            connectivity: numpy.ndarray, (num_elements, NGLL, NGLL) 的单元连接关系
        """
        self.kernels.load_moment_tensor_sources(source_arrays, source_elements, connectivity)
        
    # ==================== 性能和状态查询 ====================
    
    def get_backend_name(self) -> str:
        """
        返回后端名称
        
        返回:
            str, 例如 'Taichi (CUDA)', 'Taichi (CPU)'
        """
        arch_name = str(ti.cfg.arch).replace('Arch.', '')
        return f"Taichi ({arch_name})"
    
    def get_device_info(self) -> dict:
        """
        返回计算设备信息
        
        返回:
            dict, 设备信息
        """
        info = {
            'backend': str(ti.cfg.arch).replace('Arch.', ''),
            'type': 'GPU' if 'cuda' in str(ti.cfg.arch).lower() or 
                            'vulkan' in str(ti.cfg.arch).lower() or
                            'metal' in str(ti.cfg.arch).lower()
                    else 'CPU',
        }
        
        # 尝试获取更多信息
        try:
            if hasattr(ti, 'core'):
                info['device_name'] = str(ti.cfg.arch)
        except:
            pass
        
        return info
    
    def synchronize(self) -> None:
        """
        同步所有Taichi计算, 等待GPU完成
        """
        ti.sync()
    
    # ==================== 辅助方法 ====================
    
    def __repr__(self):
        """字符串表示"""
        if self.kernels:
            return (f"TaichiBackend(arch={ti.cfg.arch}, "
                   f"nodes={self.kernels.num_nodes}, "
                   f"receivers={self.kernels.num_receivers})")
        else:
            return f"TaichiBackend(arch={ti.cfg.arch}, uninitialized)"
