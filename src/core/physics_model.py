"""
物理模型抽象基类
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Tuple, Any
import numpy as np


class WaveFieldType(Enum):
    """波场类型枚举"""
    DISPLACEMENT = 1  # 位移场
    VELOCITY = 2      # 速度场
    ACCELERATION = 3  # 加速度场
    PRESSURE = 4      # 压力场
    STRESS = 5        # 应力场


class PhysicsModelBase(ABC):
    """
    物理模型抽象基类
    """
    
    def __init__(self, material, boundary_conditions):
        """
        初始化物理模型
        
        参数:
            material: Material对象，提供材料属性
            boundary_conditions: BoundaryConditions对象，提供边界条件
        """
        self.material = material
        self.bc = boundary_conditions
    
    # ==================== 基本属性 ====================
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        返回物理模型名称
        
        返回:
            str, 例如 'Isotropic Elastic', 'Acoustic', 'Viscoelastic'
        """
        pass
    
    @abstractmethod
    def get_field_dimension(self) -> int:
        """
        返回波场的维度
        
        返回:
            int, 波场分量数
                - 弹性波2D: 2 (ux, uz)
                - 声波: 1 (pressure)
                - 弹性波3D: 3 (ux, uy, uz)
        """
        pass
    
    @abstractmethod
    def get_primary_field_type(self) -> WaveFieldType:
        """
        返回主要波场类型
        """
        pass
    
    # ==================== 矩阵需求（必须实现） ====================
    
    @abstractmethod
    def get_required_matrix_types(self) -> List[str]:
        """
        返回该物理模型需要的全局矩阵类型列表
        
        返回:
            List[str], 矩阵名称列表
            例如: ['M', 'Cx', 'Cz', 'Kx1', 'Kx2', ..., 'Kz5']
        """
        pass
    
    @abstractmethod
    def get_auxiliary_field_names(self) -> List[str]:
        """
        返回辅助场变量名称列表
        
        返回:
            List[str], 例如 ['p_x', 'p_z'] (PML辅助场)
                            ['memory_var_1', 'memory_var_2'] (粘弹性记忆变量)
        """
        pass
    
    # ==================== 物理参数 ====================
    
    @abstractmethod
    def get_material_properties(self) -> Dict[str, np.ndarray]:
        """
        返回物理模型需要的材料属性
        
        返回:
            Dict[str, np.ndarray], 键为属性名，值为节点值数组
            例如: {
                'rho': np.ndarray,  # 密度
                'vp': np.ndarray,   # P波速度
                'vs': np.ndarray,   # S波速度
                'c11': np.ndarray,  # 弹性常数
                ...
            }
        """
        pass
    
    @abstractmethod
    def get_boundary_properties(self) -> Dict[str, np.ndarray]:
        """
        返回边界条件相关的属性
        
        返回:
            Dict[str, np.ndarray], 键为属性名，值为节点值数组
            例如: {
                'pml_dx': np.ndarray,   # PML x方向阻尼
                'pml_dxx': np.ndarray,  # PML x方向阻尼平方
            }
        """
        pass
    
    # ==================== 矩阵提取接口 ====================
    
    @abstractmethod
    def get_mass_damping_matrices(self, matrices_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        从全局矩阵字典中提取质量和阻尼矩阵, 时间积分器调用此方法获取隐式时间积分所需的矩阵
        
        参数:
            matrices_dict: Dict[str, Any], 包含所有已组装矩阵的字典
        
        返回:
            Dict[str, Any], 包含质量和阻尼矩阵的字典
            如: {
                'M': matrices_dict['M'],
                'Cx': matrices_dict['Cx'],
                'Cz': matrices_dict['Cz'],
                ...
            }

        """
        pass
    
    @abstractmethod
    def get_stiffness_matrices(self, matrices_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        从全局矩阵字典中提取刚度矩阵, 时间积分器调用此方法获取力项计算所需的刚度矩阵
        
        参数:
            matrices_dict: Dict[str, Any], 包含所有已组装矩阵的字典
        
        返回:
            Dict[str, Any], 包含刚度矩阵的字典
            例如: {
                'Kx1': matrices_dict['Kx1'],
                'Kx2': matrices_dict['Kx2'],
                ...
                'Kz5': matrices_dict['Kz5']
            }
        """
        pass
    
    # ==================== 物理方程 ====================
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        验证物理模型配置是否合法
        
        返回:
            Tuple[bool, List[str]]

        """
        errors = []
        
        # 基本验证：材料对象是否存在
        if self.material is None:
            errors.append("Material object is not provided")
        
        # 基本验证：边界条件对象是否存在
        if self.bc is None:
            errors.append("Boundary conditions object is not provided")
        
        return (len(errors) == 0, errors)
    
    def get_stability_criterion(self, dt: float, mesh) -> Tuple[bool, float]:
        """
        计算CFL稳定性条件
        
        参数:
            dt: float, 时间步长
            mesh: Mesh对象
        
        返回:
            Tuple[bool, float]

        """
        return (True, float('inf'))
    
    # ==================== 辅助方法 ====================
    
    def __repr__(self):
        """字符串表示"""
        return f"{self.__class__.__name__}(model='{self.get_model_name()}')"
