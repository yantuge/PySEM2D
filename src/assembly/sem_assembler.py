"""
谱元法组装器实现
该类实现了谱元法的全局矩阵组装
"""

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, diags
from .base import AssemblerBase
from src.taichi_config import TaichiConfig

# 尝试导入Taichi以支持GPU加速
try:
    import taichi as ti
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class SEMAssembler(AssemblerBase):
    """
    谱元法矩阵组装器
    
    该类负责组装谱元法所需的全局矩阵：
    质量矩阵 (M)
    阻尼矩阵 (Cx, Cz, Cxx, Cxz, Czz)
    刚度矩阵 (Kx1-Kx5, Kz1-Kz5)
    """
    
    def __init__(self, gll, mesh, physics_model=None, material=None, bc=None):
        """
        初始化谱元法组装器
        
        参数:
            gll: GLL积分对象
            mesh: Mesh, 网格对象
            physics_model: PhysicsModelBase, 物理模型
            material: Material, 材料对象
            bc: BoundaryConditions, 边界条件对象
        """
        super().__init__(gll, mesh, physics_model, material, bc)
        
        # 计算并缓存雅可比相关量和全局节点坐标
        print("  Computing Jacobian and coordinate derivatives...")
        (global_nodes, self.jacobian, 
         self.xix, self.xiz, self.gammax, self.gammaz) = self._compute_jacobian_and_derivatives()
        
        # 重要：更新Mesh对象的global_nodes，确保整个系统使用统一的GLL坐标
        # 这会覆盖Gmsh读入的等距节点坐标（如果有），使用正确的GLL节点坐标
        self.mesh.global_nodes = global_nodes
        print(f"  Jacobian computation complete")
        print(f"  Updated mesh.global_nodes with GLL coordinates")
    
    def _compute_jacobian_and_derivatives(self):
        """
        计算每个单元在每个 GLL 点上的雅可比矩阵及其导数，同时计算全局节点坐标
        
        返回:
            global_nodes: 全局节点坐标 [num_nodes, 2]
            jacobian, xix, xiz, gammax, gammaz: 雅可比及其导数
        """
        NGLL = self.gll.order + 1
        num_elements = self.mesh.num_elements
        num_nodes = self.mesh.num_nodes
        ref_coords = self.gll.points
        ngnod = 4  # 4节点双线性单元

        # 初始化输出数组
        jacobian = np.zeros((num_elements, NGLL, NGLL))
        xix = np.zeros_like(jacobian)
        xiz = np.zeros_like(jacobian)
        gammax = np.zeros_like(jacobian)
        gammaz = np.zeros_like(jacobian)
        
        # 初始化全局节点坐标数组
        global_nodes = np.zeros((num_nodes, 2))

        for e in range(num_elements):
            control_node_indices = self.mesh.elements[e, :]
            el_control_nodes = self.mesh.control_nodes[control_node_indices, :]
            
            for j in range(NGLL):  # eta方向
                for i in range(NGLL):  # xi方向
                    xi = ref_coords[i]
                    eta = ref_coords[j]
                    
                    # 4节点双线性形函数
                    sp = xi + 1.0
                    sm = xi - 1.0
                    tp = eta + 1.0
                    tm = eta - 1.0
                    
                    N = 0.25 * np.array([sm * tm, -sp * tm, sp * tp, -sm * tp])
                    dN_dxi = 0.25 * np.array([tm, -tm, tp, -tp])
                    dN_deta = 0.25 * np.array([sm, -sp, sp, -sm])
                    
                    # 计算全局坐标
                    x = np.dot(N, el_control_nodes[:, 0])
                    z = np.dot(N, el_control_nodes[:, 1])
                    
                    # 雅可比矩阵分量
                    xxi = np.dot(dN_dxi, el_control_nodes[:, 0])
                    xgamma = np.dot(dN_deta, el_control_nodes[:, 0])
                    zxi = np.dot(dN_dxi, el_control_nodes[:, 1])
                    zgamma = np.dot(dN_deta, el_control_nodes[:, 1])

                    # 雅可比行列式
                    J = xxi * zgamma - xgamma * zxi
                    
                    if J <= 0:
                        raise ValueError(f"Element {e} has non-positive Jacobian: J = {J}")
                    
                    # 全局节点索引
                    iglob = self.mesh.connectivity[e, j, i]
                    global_nodes[iglob, :] = [x, z]
                    
                    # 雅可比及其导数
                    jacobian[e, i, j] = J
                    xix[e, i, j] = zgamma / J
                    xiz[e, i, j] = -xgamma / J
                    gammax[e, i, j] = -zxi / J
                    gammaz[e, i, j] = xxi / J
        
        return global_nodes, jacobian, xix, xiz, gammax, gammaz
    
    def assemble_matrices(self, use_gpu=False):
        """
        组装所有需要的全局矩阵
        
        参数:
            use_gpu: bool, 刚度矩阵组装是否使用GPU加速
        
        返回:
            dict: 矩阵字典
        """
        print("\n  Assembling global matrices...")
        
        # 查询physics_model需要的矩阵类型
        if self.physics_model is not None:
            required_matrices = self.physics_model.get_required_matrix_types()
            print(f"  Physics model '{self.physics_model.get_model_name()}' requires {len(required_matrices)} matrices")
            print(f"  Required matrices: {required_matrices}")
        else:
            # 如果没有physics_model，使用默认列表
            required_matrices = [
                'M', 'Cx', 'Cz', 'Cxx', 'Cxz', 'Czz',
                'Kx1', 'Kx2', 'Kx3', 'Kx4', 'Kx5',
                'Kz1', 'Kz2', 'Kz3', 'Kz4', 'Kz5'
            ]
            print(f"  Warning: No physics model provided, assembling default 16 matrices")
        
        # 确定需要组装的矩阵类型
        need_mass_damping = any(name in required_matrices for name in ['M', 'Cx', 'Cz', 'Cxx', 'Cxz', 'Czz'])
        need_stiffness = any(name.startswith('K') for name in required_matrices)
        
        # 1. 组装质量和阻尼矩阵,对角矩阵
        if need_mass_damping:
            print("  Assembling mass and damping matrices...")
            M, Cx, Cz, Cxx, Cxz, Czz = self._assemble_mass_and_damping()
            
            # 只存储需要的矩阵
            if 'M' in required_matrices:
                self.matrices['M'] = M
            if 'Cx' in required_matrices:
                self.matrices['Cx'] = Cx
            if 'Cz' in required_matrices:
                self.matrices['Cz'] = Cz
            if 'Cxx' in required_matrices:
                self.matrices['Cxx'] = Cxx
            if 'Cxz' in required_matrices:
                self.matrices['Cxz'] = Cxz
            if 'Czz' in required_matrices:
                self.matrices['Czz'] = Czz
        
        # 2. 组装刚度矩阵,稀疏矩阵
        if need_stiffness:
            print("  Assembling stiffness matrices...")
            (Kx1, Kx2, Kx3, Kx4, Kx5,
             Kz1, Kz2, Kz3, Kz4, Kz5) = self._assemble_stiffness(use_gpu=use_gpu)
            
            # 只存储需要的矩阵
            if 'Kx1' in required_matrices:
                self.matrices['Kx1'] = Kx1
            if 'Kx2' in required_matrices:
                self.matrices['Kx2'] = Kx2
            if 'Kx3' in required_matrices:
                self.matrices['Kx3'] = Kx3
            if 'Kx4' in required_matrices:
                self.matrices['Kx4'] = Kx4
            if 'Kx5' in required_matrices:
                self.matrices['Kx5'] = Kx5
            if 'Kz1' in required_matrices:
                self.matrices['Kz1'] = Kz1
            if 'Kz2' in required_matrices:
                self.matrices['Kz2'] = Kz2
            if 'Kz3' in required_matrices:
                self.matrices['Kz3'] = Kz3
            if 'Kz4' in required_matrices:
                self.matrices['Kz4'] = Kz4
            if 'Kz5' in required_matrices:
                self.matrices['Kz5'] = Kz5
        
        print(f"  Matrix assembly complete. Total matrices: {len(self.matrices)}")
        print(f"  Assembled matrices: {list(self.matrices.keys())}")
        
        return self.matrices
    
    def _assemble_mass_and_damping(self):
        """
        组装对角化的质量矩阵和阻尼矩阵
        
        返回:
            M, Cx, Cz, Cxx, Cxz, Czz: 对角矩阵
        """
        num_nodes = self.mesh.num_nodes
        NGLL = self.gll.order + 1
        
        M = np.zeros(num_nodes)
        w = self.gll.weights

        for e in range(self.mesh.num_elements):
            conn = self.mesh.connectivity[e, :, :]
            for j in range(NGLL):
                for i in range(NGLL):
                    iglob = conn[j, i]
                    M[iglob] += w[i] * w[j] * self.jacobian[e, i, j]
        
        # 获取材料密度
        if self.physics_model is not None:
            mat_props = self.physics_model.get_material_properties()
            rho = mat_props.get('rho', 1.0)
        elif self.material is not None:
            rho = self.material.rho
        else:
            raise ValueError("No material information provided (need physics_model or material)")
        
        M *= rho
        
        # 获取PML阻尼系数
        if self.physics_model is not None:
            bc_props = self.physics_model.get_boundary_properties()
            pml_dx = bc_props.get('pml_dx', np.zeros(num_nodes))
            pml_dz = bc_props.get('pml_dz', np.zeros(num_nodes))
        elif self.bc is not None:
            pml_dx = self.bc.pml_dx
            pml_dz = self.bc.pml_dz
        else:
            # 没有边界条件，使用零阻尼
            pml_dx = np.zeros(num_nodes)
            pml_dz = np.zeros(num_nodes)
        
        Cx = M * pml_dx
        Cz = M * pml_dz
        Cxx = Cx * pml_dx
        Cxz = Cx * pml_dz
        Czz = Cz * pml_dz

        return M, Cx, Cz, Cxx, Cxz, Czz
    
    def _assemble_stiffness(self, use_gpu=False):
        """
        组装刚度矩阵
        
        参数:
            use_gpu: bool, 是否使用GPU加速
        
        返回:
            (Kx1, Kx2, Kx3, Kx4, Kx5, Kz1, Kz2, Kz3, Kz4, Kz5): 稀疏矩阵（CSC格式）
        """
        if use_gpu and GPU_AVAILABLE:
            print("    Using GPU acceleration for stiffness assembly")
            return self._assemble_stiffness_gpu()
        else:
            if use_gpu and not GPU_AVAILABLE:
                print("    Warning: GPU requested but Taichi not available, falling back to CPU")
            return self._assemble_stiffness_cpu()
    
    def _assemble_stiffness_cpu(self):
        """
        CPU版本的刚度矩阵组装
        
        返回:
            (Kx1, Kx2, Kx3, Kx4, Kx5, Kz1, Kz2, Kz3, Kz4, Kz5): 稀疏矩阵（CSC格式）
        """
        num_nodes = self.mesh.num_nodes
        num_elements = self.mesh.num_elements
        NGLL = self.gll.order + 1
        
        w = self.gll.weights
        hprime = self.gll.derivative_matrix
        delta = np.eye(NGLL)

        # COO格式构建
        rows, cols = [], []
        k1_vals, k2_vals, k3_vals, k4_vals, k5_vals, k6_vals = [], [], [], [], [], []

        print(f"    Total elements: {num_elements}")
        progress_step = max(1, num_elements // 10)
        last_progress = -1
        
        for e in range(num_elements):
            # 进度显示
            current_progress = int((e + 1) * 100 / num_elements)
            if (e + 1) % progress_step == 0 or e == num_elements - 1:
                if current_progress != last_progress or e == num_elements - 1:
                    print(f"    Progress: {current_progress}% ({e+1}/{num_elements})")
                    last_progress = current_progress
            
            conn = self.mesh.connectivity[e, :, :]
            conn_flat = conn.T.flatten(order='F')
            
            # 单元刚度矩阵: 6个基础矩阵
            Ke1 = np.zeros((NGLL*NGLL, NGLL*NGLL))
            Ke2 = np.zeros_like(Ke1)
            Ke3 = np.zeros_like(Ke1)
            Ke4 = np.zeros_like(Ke1)
            Ke5 = np.zeros_like(Ke1)
            Ke6 = np.zeros_like(Ke1)

            for l in range(NGLL):  # eta积分点
                for k in range(NGLL):  # xi积分点
                    wkl = w[k] * w[l]
                    jac = self.jacobian[e, k, l]
                    
                    xix_kl = self.xix[e, k, l]
                    xiz_kl = self.xiz[e, k, l]
                    gammax_kl = self.gammax[e, k, l]
                    gammaz_kl = self.gammaz[e, k, l]
                    
                    # 计算基函数导数
                    dphi_dx = np.zeros((NGLL, NGLL))
                    dphi_dz = np.zeros((NGLL, NGLL))
                    phi = np.zeros((NGLL, NGLL))
                    
                    for i in range(NGLL):
                        for j in range(NGLL):
                            dphi_dx[i, j] = hprime[i, k] * delta[j, l] * xix_kl + \
                                           delta[i, k] * hprime[j, l] * gammax_kl
                            dphi_dz[i, j] = hprime[i, k] * delta[j, l] * xiz_kl + \
                                           delta[i, k] * hprime[j, l] * gammaz_kl
                            phi[i, j] = delta[i, k] * delta[j, l]
                    
                    dphi_dx_flat = dphi_dx.flatten(order='F')
                    dphi_dz_flat = dphi_dz.flatten(order='F')
                    phi_flat = phi.flatten(order='F')
                    
                    # 组装6个基础单元刚度矩阵
                    factor = wkl * jac
                    
                    Ke1 += factor * np.outer(dphi_dx_flat, dphi_dx_flat)
                    Ke2 += factor * np.outer(dphi_dz_flat, dphi_dz_flat)
                    Ke3 += factor * np.outer(dphi_dx_flat, dphi_dz_flat)
                    Ke4 += factor * np.outer(dphi_dz_flat, dphi_dx_flat)
                    Ke5 += factor * np.outer(phi_flat, dphi_dx_flat)
                    Ke6 += factor * np.outer(phi_flat, dphi_dz_flat)
            
            # 添加到全局COO列表
            for i in range(NGLL*NGLL):
                for j in range(NGLL*NGLL):
                    rows.append(conn_flat[i])
                    cols.append(conn_flat[j])
                    k1_vals.append(Ke1[i, j])
                    k2_vals.append(Ke2[i, j])
                    k3_vals.append(Ke3[i, j])
                    k4_vals.append(Ke4[i, j])
                    k5_vals.append(Ke5[i, j])
                    k6_vals.append(Ke6[i, j])
        
        # 转换为CSC格式
        print("    Converting to CSC format...")
        K1 = coo_matrix((k1_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K2 = coo_matrix((k2_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K3 = coo_matrix((k3_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K4 = coo_matrix((k4_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K5 = coo_matrix((k5_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K6 = coo_matrix((k6_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        
        # 获取材料弹性系数
        if self.physics_model is not None:
            mat_props = self.physics_model.get_material_properties()
            c11 = mat_props.get('c11', 1.0)
            c13 = mat_props.get('c13', 0.0)
            c33 = mat_props.get('c33', 1.0)
            c44 = mat_props.get('c44', 1.0)
        elif self.material is not None:
            c11 = self.material.c11
            c13 = self.material.c13
            c33 = self.material.c33
            c44 = self.material.c44
        else:
            raise ValueError("No material information provided")
        
        # 获取PML阻尼导数
        if self.physics_model is not None:
            bc_props = self.physics_model.get_boundary_properties()
            pml_dxx = bc_props.get('pml_dxx', np.zeros(num_nodes))
            pml_dzz = bc_props.get('pml_dzz', np.zeros(num_nodes))
        elif self.bc is not None:
            pml_dxx = self.bc.pml_dxx
            pml_dzz = self.bc.pml_dzz
        else:
            pml_dxx = np.zeros(num_nodes)
            pml_dzz = np.zeros(num_nodes)
        
        # 用材料属性对角矩阵左乘组合最终的刚度矩阵
        print("    Combining stiffness matrices with material properties...")
        C11 = diags(c11)
        C13 = diags(c13)
        C33 = diags(c33)
        C44 = diags(c44)

        Kx1 = C11 @ K1
        Kx2 = C13 @ K3 + C44 @ K4
        Kx3 = C44 @ K2
        Kx4 = diags(c11 * pml_dxx) @ K5
        Kx5 = diags(c44 * pml_dzz) @ K6

        Kz1 = C44 @ K1
        Kz2 = C44 @ K3 + C13 @ K4
        Kz3 = C33 @ K2
        Kz4 = diags(c44 * pml_dxx) @ K5
        Kz5 = diags(c33 * pml_dzz) @ K6
        
        print("    Stiffness matrices assembled successfully")
        
        return Kx1, Kx2, Kx3, Kx4, Kx5, Kz1, Kz2, Kz3, Kz4, Kz5
    
    def get_matrix(self, matrix_name):
        """
        获取指定名称的矩阵
        
        参数:
            matrix_name: str, 矩阵名称
        
        返回:
            矩阵数据
        """
        if matrix_name not in self.matrices:
            raise KeyError(f"Matrix '{matrix_name}' not found. Available: {list(self.matrices.keys())}")
        return self.matrices[matrix_name]
    
    def has_matrix(self, matrix_name):
        """
        检查是否有指定名称的矩阵
        
        参数:
            matrix_name: str, 矩阵名称
        
        返回:
            bool
        """
        return matrix_name in self.matrices
    
    def _assemble_stiffness_gpu(self):
        """
        GPU加速版本的刚度矩阵组装
        使用Taichi并行计算单元刚度矩阵，然后在CPU上组装全局稀疏矩阵
  
        返回:
            (Kx1, Kx2, Kx3, Kx4, Kx5, Kz1, Kz2, Kz3, Kz4, Kz5): 稀疏矩阵（CSC格式）
        """
        if not GPU_AVAILABLE:
            raise RuntimeError("Taichi not available for GPU acceleration")
        
        num_nodes = self.mesh.num_nodes
        num_elements = self.mesh.num_elements
        NGLL = self.gll.order + 1
        
        print(f"    Total elements: {num_elements}")
        print(f"    Initializing Taichi fields...")
        
        # 初始化Taichi字段
        connectivity = ti.field(dtype=ti.i32, shape=(num_elements, NGLL, NGLL))
        jacobian_field = ti.field(dtype=float, shape=(num_elements, NGLL, NGLL))
        xix_field = ti.field(dtype=float, shape=(num_elements, NGLL, NGLL))
        xiz_field = ti.field(dtype=float, shape=(num_elements, NGLL, NGLL))
        gammax_field = ti.field(dtype=float, shape=(num_elements, NGLL, NGLL))
        gammaz_field = ti.field(dtype=float, shape=(num_elements, NGLL, NGLL))
        
        weights = ti.field(dtype=float, shape=NGLL)
        hprime = ti.field(dtype=float, shape=(NGLL, NGLL))
        
        # 输出字段：6个基础刚度矩阵
        k1_field = ti.field(dtype=float, shape=(num_elements, NGLL*NGLL, NGLL*NGLL))
        k2_field = ti.field(dtype=float, shape=(num_elements, NGLL*NGLL, NGLL*NGLL))
        k3_field = ti.field(dtype=float, shape=(num_elements, NGLL*NGLL, NGLL*NGLL))
        k4_field = ti.field(dtype=float, shape=(num_elements, NGLL*NGLL, NGLL*NGLL))
        k5_field = ti.field(dtype=float, shape=(num_elements, NGLL*NGLL, NGLL*NGLL))
        k6_field = ti.field(dtype=float, shape=(num_elements, NGLL*NGLL, NGLL*NGLL))
        
        # 填充数据,确保数据类型匹配以避免警告
        print(f"    Copying data to GPU...")
        connectivity.from_numpy(self.mesh.connectivity.astype(np.int32))
        jacobian_field.from_numpy(self.jacobian.astype(TaichiConfig.get_np_float_type()))
        xix_field.from_numpy(self.xix.astype(TaichiConfig.get_np_float_type()))
        xiz_field.from_numpy(self.xiz.astype(TaichiConfig.get_np_float_type()))
        gammax_field.from_numpy(self.gammax.astype(TaichiConfig.get_np_float_type()))
        gammaz_field.from_numpy(self.gammaz.astype(TaichiConfig.get_np_float_type()))
        weights.from_numpy(self.gll.weights.astype(TaichiConfig.get_np_float_type()))
        hprime.from_numpy(self.gll.derivative_matrix.astype(TaichiConfig.get_np_float_type()))
        
        # 定义Taichi内核
        @ti.kernel
        def assemble_element_stiffness():
            """GPU并行计算所有单元的6个基础刚度矩阵"""
            for e in range(num_elements):
                # 每个单元独立计算
                for l in range(NGLL):  # eta积分点
                    for k in range(NGLL):  # xi积分点
                        wkl = weights[k] * weights[l]
                        jac = jacobian_field[e, k, l]
                        
                        xix_kl = xix_field[e, k, l]
                        xiz_kl = xiz_field[e, k, l]
                        gammax_kl = gammax_field[e, k, l]
                        gammaz_kl = gammaz_field[e, k, l]
                        
                        factor = wkl * jac
                        
                        # 计算基函数导数和形函数值
                        for i in range(NGLL):
                            for j in range(NGLL):
                                # 索引转换
                                idx_ij = i + j * NGLL
                                
                                # 计算dphi/dx, dphi/dz, phi
                                delta_ik = 1.0 if i == k else 0.0
                                delta_jl = 1.0 if j == l else 0.0
                                
                                dphi_dx_ij = hprime[i, k] * delta_jl * xix_kl + \
                                            delta_ik * hprime[j, l] * gammax_kl
                                dphi_dz_ij = hprime[i, k] * delta_jl * xiz_kl + \
                                            delta_ik * hprime[j, l] * gammaz_kl
                                phi_ij = delta_ik * delta_jl
                                
                                for i2 in range(NGLL):
                                    for j2 in range(NGLL):
                                        idx_i2j2 = i2 + j2 * NGLL
                                        
                                        delta_i2k = 1.0 if i2 == k else 0.0
                                        delta_j2l = 1.0 if j2 == l else 0.0
                                        
                                        dphi_dx_i2j2 = hprime[i2, k] * delta_j2l * xix_kl + \
                                                      delta_i2k * hprime[j2, l] * gammax_kl
                                        dphi_dz_i2j2 = hprime[i2, k] * delta_j2l * xiz_kl + \
                                                      delta_i2k * hprime[j2, l] * gammaz_kl
                                        phi_i2j2 = delta_i2k * delta_j2l
                                        
                                        # 组装6个基础刚度矩阵
                                        ti.atomic_add(k1_field[e, idx_ij, idx_i2j2], 
                                                     factor * dphi_dx_ij * dphi_dx_i2j2)
                                        ti.atomic_add(k2_field[e, idx_ij, idx_i2j2], 
                                                     factor * dphi_dz_ij * dphi_dz_i2j2)
                                        ti.atomic_add(k3_field[e, idx_ij, idx_i2j2], 
                                                     factor * dphi_dx_ij * dphi_dz_i2j2)
                                        ti.atomic_add(k4_field[e, idx_ij, idx_i2j2], 
                                                     factor * dphi_dz_ij * dphi_dx_i2j2)
                                        ti.atomic_add(k5_field[e, idx_ij, idx_i2j2], 
                                                     factor * phi_ij * dphi_dx_i2j2)
                                        ti.atomic_add(k6_field[e, idx_ij, idx_i2j2], 
                                                     factor * phi_ij * dphi_dz_i2j2)
        
        # 执行GPU计算
        print(f"    Computing element stiffness on GPU...")
        import time
        start_time = time.time()
        assemble_element_stiffness()
        ti.sync()
        gpu_time = time.time() - start_time
        print(f"    GPU computation completed in {gpu_time:.2f}s")
        
        # 复制回CPU并组装全局稀疏矩阵
        print(f"    Transferring data from GPU to CPU...")
        k1_np = k1_field.to_numpy()
        k2_np = k2_field.to_numpy()
        k3_np = k3_field.to_numpy()
        k4_np = k4_field.to_numpy()
        k5_np = k5_field.to_numpy()
        k6_np = k6_field.to_numpy()
        
        # COO格式组装
        print(f"    Assembling global sparse matrices on CPU...")
        rows, cols = [], []
        k1_vals, k2_vals, k3_vals, k4_vals, k5_vals, k6_vals = [], [], [], [], [], []
        
        progress_step = max(1, num_elements // 10)
        last_progress = -1
        
        for e in range(num_elements):
            # 进度显示
            current_progress = int((e + 1) * 100 / num_elements)
            if (e + 1) % progress_step == 0 or e == num_elements - 1:
                if current_progress != last_progress or e == num_elements - 1:
                    print(f"    Progress: {current_progress}% ({e+1}/{num_elements})")
                    last_progress = current_progress
            
            conn = self.mesh.connectivity[e, :, :]
            conn_flat = conn.T.flatten(order='F')
            
            for i in range(NGLL*NGLL):
                for j in range(NGLL*NGLL):
                    rows.append(conn_flat[i])
                    cols.append(conn_flat[j])
                    k1_vals.append(k1_np[e, i, j])
                    k2_vals.append(k2_np[e, i, j])
                    k3_vals.append(k3_np[e, i, j])
                    k4_vals.append(k4_np[e, i, j])
                    k5_vals.append(k5_np[e, i, j])
                    k6_vals.append(k6_np[e, i, j])
        
        # 转换为CSC格式
        print("    Converting to CSC format...")
        K1 = coo_matrix((k1_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K2 = coo_matrix((k2_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K3 = coo_matrix((k3_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K4 = coo_matrix((k4_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K5 = coo_matrix((k5_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        K6 = coo_matrix((k6_vals, (rows, cols)), shape=(num_nodes, num_nodes)).tocsc()
        
        # 获取材料弹性系数
        if self.physics_model is not None:
            mat_props = self.physics_model.get_material_properties()
            c11 = mat_props.get('c11', 1.0)
            c13 = mat_props.get('c13', 0.0)
            c33 = mat_props.get('c33', 1.0)
            c44 = mat_props.get('c44', 1.0)
        elif self.material is not None:
            c11 = self.material.c11
            c13 = self.material.c13
            c33 = self.material.c33
            c44 = self.material.c44
        else:
            raise ValueError("No material information provided")
        
        # 获取PML阻尼导数
        if self.physics_model is not None:
            bc_props = self.physics_model.get_boundary_properties()
            pml_dxx = bc_props.get('pml_dxx', np.zeros(num_nodes))
            pml_dzz = bc_props.get('pml_dzz', np.zeros(num_nodes))
        elif self.bc is not None:
            pml_dxx = self.bc.pml_dxx
            pml_dzz = self.bc.pml_dzz
        else:
            pml_dxx = np.zeros(num_nodes)
            pml_dzz = np.zeros(num_nodes)
        
        # 用材料属性对角矩阵左乘组合最终的刚度矩阵
        print("    Combining stiffness matrices with material properties...")
        C11 = diags(c11)
        C13 = diags(c13)
        C33 = diags(c33)
        C44 = diags(c44)

        Kx1 = C11 @ K1
        Kx2 = C13 @ K3 + C44 @ K4
        Kx3 = C44 @ K2
        Kx4 = diags(c11 * pml_dxx) @ K5
        Kx5 = diags(c44 * pml_dzz) @ K6

        Kz1 = C44 @ K1
        Kz2 = C44 @ K3 + C13 @ K4
        Kz3 = C33 @ K2
        Kz4 = diags(c44 * pml_dxx) @ K5
        Kz5 = diags(c33 * pml_dzz) @ K6
        
        print("    GPU stiffness matrices assembled successfully")
        
        return Kx1, Kx2, Kx3, Kx4, Kx5, Kz1, Kz2, Kz3, Kz4, Kz5

