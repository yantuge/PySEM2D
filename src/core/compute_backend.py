"""
计算后端抽象基类

"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import numpy as np


class ComputeBackendBase(ABC):
    """
    计算后端抽象基类
    """
    
    def __init__(self, config: dict):
        """
        初始化计算后端
        
        参数:
            config: dict, 配置参数字典
                必需键: 'nt' (时间步数), 'dt' (时间步长)
        """
        self.config = config
        self.nt = config['nt']
        self.dt = config['dt']
    
    # ==================== 内存管理====================
    
    @abstractmethod
    def allocate_scalar_field(self, num_nodes: int, initial_value: float = 0.0) -> Any:
        """
        分配标量场内存
        
        参数:
            num_nodes: int, 节点数量
            initial_value: float, 初始值（默认0）
        
        返回:
            标量场对象
        """
        pass
    
    @abstractmethod
    def allocate_vector_field(self, num_nodes: int, dimension: int, 
                             initial_value: float = 0.0) -> Any:
        """
        分配矢量场内存
        
        参数:
            num_nodes: int, 节点数量
            dimension: int, 矢量维度（2D弹性波为2，3D为3）
            initial_value: float, 初始值（默认0）
        
        返回:
            矢量场对象
        """
        pass
    
    @abstractmethod
    def allocate_seismogram_storage(self, num_receivers: int, dimension: int) -> Any:
        """
        分配地震图存储空间
        
        参数:
            num_receivers: int, 接收器数量
            dimension: int, 波场维度
        
        返回:
            地震图存储对象，形状 (num_receivers, dimension, nt)
        """
        pass
    
    # ==================== 数值运算====================
    
    @abstractmethod
    def sparse_matvec(self, matrix_data: Any, matrix_indices: Any, 
                     matrix_indptr: Any, input_vector: Any, 
                     output_vector: Any, ncols: int) -> None:
        """
        稀疏矩阵-向量乘法: output = matrix @ input
        
        参数:
            matrix_data: 稀疏矩阵的非零元素值
            matrix_indices: CSC格式的列索引
            matrix_indptr: CSC格式的行指针
            input_vector: 输入向量
            output_vector: 输出向量
            ncols: 矩阵列数
        """
        pass
    
    @abstractmethod
    def vector_add(self, vec_a: Any, vec_b: Any, result: Any, 
                   scale_a: float = 1.0, scale_b: float = 1.0) -> None:
        """
        向量加法: result = scale_a * vec_a + scale_b * vec_b
        
        参数:
            vec_a: 向量A
            vec_b: 向量B
            result: 结果向量
            scale_a: A的缩放系数（默认1.0）
            scale_b: B的缩放系数（默认1.0）
        """
        pass
    
    @abstractmethod
    def scalar_multiply(self, vector: Any, scalar: float, result: Any) -> None:
        """
        标量-向量乘法: result = scalar * vector
        
        参数:
            vector: 输入向量
            scalar: 标量
            result: 结果向量
        """
        pass
    
    @abstractmethod
    def copy_field(self, source: Any, destination: Any) -> None:
        """
        复制波场
        
        参数:
            source: 源波场
            destination: 目标波场
        """
        pass
    
    # ==================== 数据访问====================
    
    @abstractmethod
    def get_field_value(self, field: Any, node_index: int) -> np.ndarray:
        """
        获取指定节点的波场值
        
        参数:
            field: 波场对象
            node_index: int, 节点索引
        
        返回:
            np.ndarray, 该节点的值
        """
        pass
    
    @abstractmethod
    def set_field_value(self, field: Any, node_index: int, value: np.ndarray) -> None:
        """
        设置指定节点的波场值
        
        参数:
            field: 波场对象
            node_index: int, 节点索引
            value: np.ndarray, 要设置的值
        """
        pass
    
    @abstractmethod
    def to_numpy(self, field: Any) -> np.ndarray:
        """
        将波场转换为NumPy数组, 用于I/O和可视化
        
        参数:
            field: 波场对象
        
        返回:
            np.ndarray, NumPy数组表示
        """
        pass
    
    # ==================== 地震图记录====================
    
    @abstractmethod
    def record_seismogram(self, field: Any, receiver_indices: np.ndarray, 
                         seismogram_storage: Any, timestep: int) -> None:
        """
        记录当前时刻的地震图数据
        
        参数:
            field: 当前波场
            receiver_indices: np.ndarray, 接收器节点索引数组
            seismogram_storage: 地震图存储对象
            timestep: int, 当前时间步
        """
        pass
    
    # ==================== 性能和状态查询====================
    
    def get_backend_name(self) -> str:
        """
        返回后端名称
        
        返回:
            str, 例如 'Taichi (CUDA)', 'NumPy (CPU)'
        """
        return self.__class__.__name__
    
    def get_device_info(self) -> dict:
        """
        返回计算设备信息
        
        返回:
            dict, 包含设备相关信息，例如:
                {
                    'type': 'GPU',
                    'name': 'NVIDIA GeForce RTX 5090',
                    'memory': '32GB'
                }
        """
        return {'type': 'Unknown', 'name': 'Unknown', 'memory': 'Unknown'}
    
    def synchronize(self) -> None:
        """
        同步计算, 等待所有异步操作完成

        """
        pass
    
    # ==================== 辅助方法 ====================
    
    def __repr__(self):
        """字符串表示"""
        return f"{self.__class__.__name__}(backend='{self.get_backend_name()}')"
