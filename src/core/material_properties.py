"""
材料属性抽象基类

"""

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class MaterialPropertiesBase(ABC):
    """
    材料属性抽象基类
    """
    
    def __init__(self, mesh):
        """
        初始化材料属性
        
        参数:
            mesh: Mesh对象，提供网格信息
        """
        self.mesh = mesh
    
    # ==================== 基本属性 ====================
    
    @abstractmethod
    def get_material_type(self) -> str:
        """
        返回材料类型名称
        
        返回:
            str, 例如 'Isotropic Elastic', 'Anisotropic Elastic', 'Viscoelastic'
        """
        pass
    
    @abstractmethod
    def get_property_names(self) -> List[str]:
        """
        返回材料包含的所有属性名称列表
        
        返回:
            List[str], 例如 ['rho', 'vp', 'vs', 'c11', 'c13', 'c33', 'c44']
        """
        pass
    
    @abstractmethod
    def get_property(self, name: str) -> np.ndarray:
        """
        获取指定名称的材料属性
        
        参数:
            name: str, 属性名称（如 'rho', 'vp' 等）
        
        返回:
            np.ndarray, 该属性在所有节点的值 (num_nodes,)

        """
        pass
    
    # ==================== 材料设置====================
    
    @abstractmethod
    def set_homogeneous(self, **kwargs) -> None:
        """
        设置均匀材料属性
        
        参数:
            **kwargs: 材料参数，例如 rho=2700, vp=5000, vs=2887
        
        示例:
            material.set_homogeneous(rho=2700, vp=5000, vs=2887)
        """
        pass
    
    # ====================验证方法====================
    
    def validate_properties(self) -> tuple:
        """
        验证材料属性的物理合理性
        
        返回:
            Tuple[bool, List[str]]

        """
        return (True, [])
    
    def get_wave_speeds(self) -> Dict[str, np.ndarray]:
        """
        计算并返回波速信息
        
        返回:
            Dict[str, np.ndarray], 键为波速类型，值为节点值数组
            例如: {'vp': np.ndarray, 'vs': np.ndarray}

        """
        return {}
    
    # ==================== 辅助方法 ====================
    
    def __repr__(self):
        """字符串表示"""
        return f"{self.__class__.__name__}(type='{self.get_material_type()}')"
