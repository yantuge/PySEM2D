"""
组装器抽象基类

定义谱元法矩阵组装的标准接口
"""

from abc import ABC, abstractmethod
import numpy as np


class AssemblerBase(ABC):
    """
    矩阵组装器抽象基类
    """
    
    def __init__(self, gll, mesh, physics_model=None, material=None, bc=None):
        """
        初始化组装器
        
        参数:
            gll: GLL积分对象
            mesh: Mesh, 网格对象
            physics_model: PhysicsModelBase, 物理模型
            material: Material, 材料对象
            bc: BoundaryConditions, 边界条件对象

        """
        self.gll = gll
        self.mesh = mesh
        self.physics_model = physics_model
        self.material = material
        self.bc = bc
        
        # 存储组装后的矩阵
        self.matrices = {}
    
    @abstractmethod
    def assemble_matrices(self):
        """
        组装所有需要的全局矩阵
        
        返回:
            dict: 矩阵字典，键为矩阵名称（如'M', 'Kx1'等），值为矩阵数据
        """
        pass
    
    @abstractmethod
    def get_matrix(self, matrix_name):
        """
        获取指定名称的矩阵
        
        参数:
            matrix_name: str, 矩阵名称（如'M', 'Kx1'等）
        
        返回:
            矩阵数据（NumPy数组或SciPy稀疏矩阵）
        """
        pass
    
    @abstractmethod
    def has_matrix(self, matrix_name):
        """
        检查是否有指定名称的矩阵
        
        参数:
            matrix_name: str, 矩阵名称
        
        返回:
            bool: True如果矩阵已组装，否则False
        """
        pass
    
    def get_all_matrices(self):
        """
        获取所有已组装的矩阵
        
        返回:
            dict: 矩阵字典
        """
        return self.matrices.copy()
    
    def get_matrix_names(self):
        """
        获取所有已组装矩阵的名称列表
        
        返回:
            list: 矩阵名称列表
        """
        return list(self.matrices.keys())
    
    def clear_matrices(self):
        """
        清除所有已组装的矩阵, 释放内存
        """
        self.matrices.clear()
