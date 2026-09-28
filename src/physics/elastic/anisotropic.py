"""
横向各向同性/VTI弹性波物理模型
"""

import numpy as np
from typing import Dict, List, Tuple
import sys
import os

# 添加src目录到路径以支持导入
if __name__ != '__main__':
    try:
        from core import PhysicsModelBase, WaveFieldType
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from core import PhysicsModelBase, WaveFieldType
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from core import PhysicsModelBase, WaveFieldType


class AnisotropicElastic(PhysicsModelBase):
    """
    VTI弹性波物理模型
    """
    
    def __init__(self, material, boundary_conditions):
        """
        初始化横向各向同性弹性波物理模型
        
        参数:
            material: Material对象
            boundary_conditions: BoundaryConditions对象
        """
        super().__init__(material, boundary_conditions)
        
        # 验证材料参数
        self._validate_material_parameters()
    
    def _validate_material_parameters(self):
        """
        验证材料参数是否满足各向异性弹性波的要求
        """
        # 检查必需的弹性常数
        required_props = ['rho', 'c11', 'c13', 'c33', 'c44']
        for prop in required_props:
            if not hasattr(self.material, prop) or getattr(self.material, prop) is None:
                raise ValueError(
                    f"Material is missing required property '{prop}' for anisotropic elastic model"
                )
        
        # 检查物理约束: 所有刚度常数和密度必须为正
        if np.any(self.material.rho <= 0):
            raise ValueError("Density (rho) must be positive")
        
        if np.any(self.material.c11 <= 0):
            raise ValueError("Elastic constant c11 must be positive")
        
        if np.any(self.material.c33 <= 0):
            raise ValueError("Elastic constant c33 must be positive")
        
        if np.any(self.material.c44 <= 0):
            raise ValueError("Elastic constant c44 (shear modulus) must be positive")
        
        # 检查弹性张量的正定性条件
        # 对于VTI介质: c11 > 0, c33 > 0, c44 > 0, c11*c33 > c13^2
        determinant_condition = self.material.c11 * self.material.c33 - self.material.c13**2
        if np.any(determinant_condition <= 0):
            raise ValueError(
                "Elastic tensor must be positive definite: c11*c33 > c13^2"
            )
        
        # 检查等效波速
        if hasattr(self.material, 'vp') and self.material.vp is not None:
            if np.any(self.material.vp <= 0):
                raise ValueError("Equivalent P-wave velocity must be positive")
        
        if hasattr(self.material, 'vs') and self.material.vs is not None:
            if np.any(self.material.vs <= 0):
                raise ValueError("Equivalent S-wave velocity must be positive")
    
    # ==================== 基本属性实现 ====================
    
    def get_model_name(self) -> str:
        """返回物理模型名称"""
        return "Anisotropic Elastic (VTI)"
    
    def get_field_dimension(self) -> int:
        """
        返回波场维度
        
        返回:
            2 - 2D弹性波有两个位移分量 (ux, uz)
        """
        return 2
    
    def get_primary_field_type(self) -> WaveFieldType:
        """
        返回主要波场类型
        
        返回:
            WaveFieldType.DISPLACEMENT - 弹性波求解位移场
        """
        return WaveFieldType.DISPLACEMENT
    
    # ==================== 矩阵需求实现 ====================
    
    def get_required_matrix_types(self) -> List[str]:
        """
        返回各向异性弹性波PML所需的全局矩阵类型
        
        返回:
            List[str], 包含16个矩阵:
                6个质量/阻尼矩阵: M, Cx, Cz, Cxx, Cxz, Czz
                10个刚度矩阵: Kx1-Kx5, Kz1-Kz5
        """
        return [
            # 质量和阻尼矩阵 (对角矩阵)
            'M',      # 质量矩阵
            'Cx',     # x方向阻尼
            'Cz',     # z方向阻尼
            'Cxx',    # x方向阻尼平方
            'Cxz',    # xz交叉阻尼
            'Czz',    # z方向阻尼平方
            
            # x方向刚度矩阵 (稀疏矩阵)
            # 各向异性材料中c11 ≠ c33，所以Kx1、Kz3等有不同的值
            'Kx1',    # c11 * K1
            'Kx2',    # c13 * K3 + c44 * K4
            'Kx3',    # c44 * K2 
            'Kx4',    # c11 * pml_dxx * K5 (PML阻尼项1)
            'Kx5',    # c44 * pml_dzz * K6 (PML阻尼项2)
            
            # z方向刚度矩阵 (稀疏矩阵)
            'Kz1',    # c44 * K1
            'Kz2',    # c44 * K3 + c13 * K4
            'Kz3',    # c33 * K2
            'Kz4',    # c44 * pml_dxx * K5
            'Kz5',    # c33 * pml_dzz * K6
        ]
    
    def get_auxiliary_field_names(self) -> List[str]:
        """
        返回PML辅助场变量名称
        
        返回:
            List[str], PML辅助场名称
        """
        return ['p_x_0', 'p_x_1', 'p_z_0', 'p_z_1']
    
    # ==================== 物理参数实现 ====================
    
    def get_material_properties(self) -> Dict[str, np.ndarray]:
        """
        返回各向异性弹性波所需的材料属性
        
        返回:
            Dict[str, np.ndarray], 材料属性字典
        """
        result = {
            'rho': self.material.rho,
            'c11': self.material.c11,
            'c13': self.material.c13,
            'c33': self.material.c33,
            'c44': self.material.c44,
        }
        
        # 添加等效波速
        if hasattr(self.material, 'vp') and self.material.vp is not None:
            result['vp'] = self.material.vp
        if hasattr(self.material, 'vs') and self.material.vs is not None:
            result['vs'] = self.material.vs
        if hasattr(self.material, 'mu') and self.material.mu is not None:
            result['mu'] = self.material.mu
        if hasattr(self.material, 'lambda_') and self.material.lambda_ is not None:
            result['lambda'] = self.material.lambda_
        
        return result
    
    def get_boundary_properties(self) -> Dict[str, np.ndarray]:
        """
        返回PML边界条件的阻尼系数
        
        返回:
            Dict[str, np.ndarray], PML阻尼系数
        """
        return {
            'pml_dx': self.bc.pml_dx,
            'pml_dz': self.bc.pml_dz,
            'pml_dxx': self.bc.pml_dxx,
            'pml_dzz': self.bc.pml_dzz,
            'pml_dxz': self.bc.pml_dx * self.bc.pml_dz,
        }
    
    # ==================== 稳定性条件 ====================
    def get_stability_criterion(self, dt: float, mesh) -> Tuple[bool, float]:
        """
        计算CFL稳定性条件
        
        参数:
            dt: float, 时间步长
            mesh: Mesh对象
        
        返回:
            Tuple[bool, float]
        """
        # 估算最小网格尺寸
        coords = mesh.global_nodes
        dx_approx = (coords[:, 0].max() - coords[:, 0].min()) / np.sqrt(mesh.num_nodes)
        dz_approx = (coords[:, 1].max() - coords[:, 1].min()) / np.sqrt(mesh.num_nodes)
        h_min = min(dx_approx, dz_approx)
        
        # 计算各方向的最大波速
        # 水平P波: sqrt(c11/rho), 垂直P波: sqrt(c33/rho)
        vp_horizontal_max = np.sqrt(np.max(self.material.c11 / self.material.rho))
        vp_vertical_max = np.sqrt(np.max(self.material.c33 / self.material.rho))
        v_max = max(vp_horizontal_max, vp_vertical_max)
        
        # CFL条件
        dt_max = h_min / (v_max * np.sqrt(2.0)) * 0.5
        
        is_stable = dt <= dt_max
        
        return (is_stable, dt_max)
    
    # ==================== 验证方法 ====================
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        验证物理模型配置
        
        返回:
            Tuple[bool, List[str]]
        """
        is_valid, errors = super().validate_configuration()
        
        # 验证材料参数
        try:
            self._validate_material_parameters()
        except ValueError as e:
            errors.append(str(e))
            is_valid = False
        
        # 验证边界条件
        required_bc_props = ['pml_dx', 'pml_dz', 'pml_dxx', 'pml_dzz']
        for prop in required_bc_props:
            if not hasattr(self.bc, prop):
                errors.append(f"Boundary conditions missing required property '{prop}' for PML")
                is_valid = False
        
        return (is_valid, errors)
    
    # ==================== 辅助方法 ====================
    
    def get_wave_types(self) -> List[str]:
        """
        返回该物理模型支持的波类型
        
        返回:
            List[str], 各向异性介质中的波类型
        """
        return ['qP-wave', 'qS-wave']  # quasi-P和quasi-S波
    
    def compute_wavelengths(self, frequency: float) -> Dict[str, np.ndarray]:
        """
        计算给定频率下的波长
        
        参数:
            frequency: float, 频率
        
        返回:
            Dict[str, np.ndarray], 波长字典
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        
        # 计算各方向波速
        vp_horizontal = np.sqrt(self.material.c11 / self.material.rho)
        vp_vertical = np.sqrt(self.material.c33 / self.material.rho)
        vs = np.sqrt(self.material.c44 / self.material.rho)
        
        return {
            'lambda_p_horizontal': vp_horizontal / frequency,
            'lambda_p_vertical': vp_vertical / frequency,
            'lambda_s': vs / frequency,
        }
    
    def get_anisotropy_parameters(self) -> Dict[str, np.ndarray]:
        """
        计算Thomsen各向异性参数
        
        返回:
            Dict[str, np.ndarray], Thomsen参数:
                - epsilon: P波各向异性参数
                - delta: 近垂直传播的各向异性参数
                - gamma: S波各向异性参数
        """
        c11 = self.material.c11
        c13 = self.material.c13
        c33 = self.material.c33
        c44 = self.material.c44
        
        epsilon = (c11 - c33) / (2.0 * c33)
        
        # 简化的delta计算
        numerator = (c13 + c44)**2 - (c33 - c44)**2
        denominator = 2.0 * c33 * (c33 - c44)
        # 避免除零
        delta = np.where(
            np.abs(denominator) > 1e-10,
            numerator / denominator,
            0.0
        )
        
        return {
            'epsilon': epsilon,
            'delta': delta,
            'gamma': np.zeros_like(epsilon),  # VTI假设
        }
    
    # ==================== 矩阵提取接口实现 ====================
    
    def get_mass_damping_matrices(self, matrices_dict: Dict[str, any]) -> Dict[str, any]:
        """
        从全局矩阵字典中提取质量和阻尼矩阵
        
        参数:
            matrices_dict: Dict[str, any], 包含所有已组装矩阵的字典
        
        返回:
            Dict[str, any], 质量和阻尼矩阵
        """
        required_matrices = ['M', 'Cx', 'Cz', 'Cxx', 'Cxz', 'Czz']
        
        missing = [name for name in required_matrices if name not in matrices_dict]
        if missing:
            raise KeyError(
                f"Missing required mass/damping matrices: {missing}. "
                f"Available matrices: {list(matrices_dict.keys())}"
            )
        
        return {
            'M': matrices_dict['M'],
            'Cx': matrices_dict['Cx'],
            'Cz': matrices_dict['Cz'],
            'Cxx': matrices_dict['Cxx'],
            'Cxz': matrices_dict['Cxz'],
            'Czz': matrices_dict['Czz'],
        }
    
    def get_stiffness_matrices(self, matrices_dict: Dict[str, any]) -> Dict[str, any]:
        """
        从全局矩阵字典中提取刚度矩阵
        
        参数:
            matrices_dict: Dict[str, any], 包含所有已组装矩阵的字典
        
        返回:
            Dict[str, any], 刚度矩阵
        """
        required_matrices = [
            'Kx1', 'Kx2', 'Kx3', 'Kx4', 'Kx5',
            'Kz1', 'Kz2', 'Kz3', 'Kz4', 'Kz5'
        ]
        
        missing = [name for name in required_matrices if name not in matrices_dict]
        if missing:
            raise KeyError(
                f"Missing required stiffness matrices: {missing}. "
                f"Available matrices: {list(matrices_dict.keys())}"
            )
        
        return {
            'Kx1': matrices_dict['Kx1'],
            'Kx2': matrices_dict['Kx2'],
            'Kx3': matrices_dict['Kx3'],
            'Kx4': matrices_dict['Kx4'],
            'Kx5': matrices_dict['Kx5'],
            'Kz1': matrices_dict['Kz1'],
            'Kz2': matrices_dict['Kz2'],
            'Kz3': matrices_dict['Kz3'],
            'Kz4': matrices_dict['Kz4'],
            'Kz5': matrices_dict['Kz5'],
        }
    
    def __repr__(self):
        """字符串表示"""
        c11_range = (np.min(self.material.c11), np.max(self.material.c11))
        c33_range = (np.min(self.material.c33), np.max(self.material.c33))
        return (f"AnisotropicElastic(VTI, c11={c11_range[0]:.2e}-{c11_range[1]:.2e} Pa, "
                f"c33={c33_range[0]:.2e}-{c33_range[1]:.2e} Pa)")
