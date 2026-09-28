"""
各向同性弹性波物理模型
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
        # 如果相对导入失败，尝试从父目录导入
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from core import PhysicsModelBase, WaveFieldType
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from core import PhysicsModelBase, WaveFieldType


class IsotropicElastic(PhysicsModelBase):
    """
    各向同性弹性波物理模型
    """
    
    def __init__(self, material, boundary_conditions):
        """
        初始化各向同性弹性波物理模型
        
        参数:
            material: Material对象
            boundary_conditions: BoundaryConditions对象
        """
        super().__init__(material, boundary_conditions)
        
        # 验证材料参数
        self._validate_material_parameters()
    
    def _validate_material_parameters(self):
        """
        验证材料参数是否满足各向同性弹性波的要求
        """
        required_props = ['rho', 'vp', 'vs', 'c11', 'c13', 'c33', 'c44']
        for prop in required_props:
            if not hasattr(self.material, prop) or getattr(self.material, prop) is None:
                raise ValueError(
                    f"Material is missing required property '{prop}' for isotropic elastic model"
                )
        
        # 检查物理约束: vp > vs > 0
        if np.any(self.material.vp <= self.material.vs):
            raise ValueError("P-wave velocity (vp) must be greater than S-wave velocity (vs)")
        
        if np.any(self.material.vs <= 0):
            raise ValueError("S-wave velocity (vs) must be positive")
        
        if np.any(self.material.rho <= 0):
            raise ValueError("Density (rho) must be positive")
    
    # ==================== 基本属性实现 ====================
    
    def get_model_name(self) -> str:
        """返回物理模型名称"""
        return "Isotropic Elastic"
    
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
        返回各向同性弹性波PML所需的全局矩阵类型
        
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
            'Kx1',    # c11 * K1
            'Kx2',    # c13 * K3 + c44 * K4
            'Kx3',    # c44 * K2
            'Kx4',    # c11 * pml_dxx * K5
            'Kx5',    # c44 * pml_dzz * K6
            
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
        返回各向同性弹性波所需的材料属性
        
        返回:
            Dict[str, np.ndarray], 材料属性字典
        """
        return {
            'rho': self.material.rho,
            'vp': self.material.vp,
            'vs': self.material.vs,
            'mu': self.material.mu,
            'lambda': self.material.lambda_,
            'c11': self.material.c11,
            'c13': self.material.c13,
            'c33': self.material.c33,
            'c44': self.material.c44,
        }
    
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
            'pml_dxz': self.bc.pml_dx * self.bc.pml_dz,  # 交叉项
        }
    
    # ==================== 稳定性条件 ====================
    
    def get_stability_criterion(self, dt: float, mesh) -> Tuple[bool, float]:
        """
        计算CFL稳定性条件
        
        参数:
            dt: float, 时间步长 (s)
            mesh: Mesh对象
        
        返回:
            Tuple[bool, float], (是否稳定, 最大允许时间步)
        
        CFL条件:
            dt < h_min / (vp_max * sqrt(ndim))
            
        其中:
            - h_min: 最小网格尺寸
            - vp_max: 最大P波速度
            - ndim: 空间维度 (2D = 2)
        """
        # 估算最小网格尺寸
        # 真实的h_min应该从mesh的单元大小中计算
        coords = mesh.global_nodes
        dx_approx = (coords[:, 0].max() - coords[:, 0].min()) / np.sqrt(mesh.num_nodes)
        dz_approx = (coords[:, 1].max() - coords[:, 1].min()) / np.sqrt(mesh.num_nodes)
        h_min = min(dx_approx, dz_approx)
        
        # 最大P波速度
        vp_max = np.max(self.material.vp)
        
        # CFL条件
        dt_max = h_min / (vp_max * np.sqrt(2.0)) * 0.5  # 乘以0.5作为安全系数
        
        is_stable = dt <= dt_max
        
        return (is_stable, dt_max)
    
    # ==================== 验证方法 ====================
    
    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        验证物理模型配置
        
        返回:
            Tuple[bool, List[str]]
        """
        # 调用父类基本验证
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
            List[str], ['P-wave', 'S-wave']
        """
        return ['P-wave', 'S-wave']
    
    def compute_wavelengths(self, frequency: float) -> Dict[str, np.ndarray]:
        """
        计算给定频率下的波长
        
        参数:
            frequency: float, 频率 (Hz)
        
        返回:
            Dict[str, np.ndarray], 波长字典:
                - 'lambda_p': P波波长 (m)
                - 'lambda_s': S波波长 (m)
        """
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        
        lambda_p = self.material.vp / frequency
        lambda_s = self.material.vs / frequency
        
        return {
            'lambda_p': lambda_p,
            'lambda_s': lambda_s,
        }
    
    # ==================== 矩阵提取接口实现 ====================
    
    def get_mass_damping_matrices(self, matrices_dict: Dict[str, any]) -> Dict[str, any]:
        """
        从全局矩阵字典中提取质量和阻尼矩阵
        参数:
            matrices_dict: Dict[str, any], 包含所有已组装矩阵的字典
        
        返回:
            Dict[str, any], 包含以下矩阵:
                - M: 质量矩阵 (对角)
                - Cx: x方向阻尼矩阵 (对角)
                - Cz: z方向阻尼矩阵 (对角)
                - Cxx: x方向阻尼平方矩阵 (对角)
                - Cxz: xz交叉阻尼矩阵 (对角)
                - Czz: z方向阻尼平方矩阵 (对角)
        """
        required_matrices = ['M', 'Cx', 'Cz', 'Cxx', 'Cxz', 'Czz']
        
        # 验证所有需要的矩阵都存在
        missing = [name for name in required_matrices if name not in matrices_dict]
        if missing:
            raise KeyError(
                f"Missing required mass/damping matrices: {missing}. "
                f"Available matrices: {list(matrices_dict.keys())}"
            )
        
        # 提取并返回
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
            Dict[str, any], 包含以下矩阵:
                - Kx1-Kx5: x方向刚度矩阵 (稀疏)
                - Kz1-Kz5: z方向刚度矩阵 (稀疏)
        """
        required_matrices = [
            'Kx1', 'Kx2', 'Kx3', 'Kx4', 'Kx5',
            'Kz1', 'Kz2', 'Kz3', 'Kz4', 'Kz5'
        ]
        
        # 验证所有需要的矩阵都存在
        missing = [name for name in required_matrices if name not in matrices_dict]
        if missing:
            raise KeyError(
                f"Missing required stiffness matrices: {missing}. "
                f"Available matrices: {list(matrices_dict.keys())}"
            )
        
        # 提取并返回
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
        vp_range = (np.min(self.material.vp), np.max(self.material.vp))
        vs_range = (np.min(self.material.vs), np.max(self.material.vs))
        return (f"IsotropicElastic(vp={vp_range[0]:.0f}-{vp_range[1]:.0f} m/s, "
                f"vs={vs_range[0]:.0f}-{vs_range[1]:.0f} m/s)")
