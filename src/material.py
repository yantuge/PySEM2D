import numpy as np

class Material:
    """
    定义和管理计算区域的物理介质属性
    """
    def __init__(self, mesh):
        self.mesh = mesh
        self.rho = None
        self.vp = None
        self.vs = None
        self.mu = None
        self.lambda_ = None
        self.c11 = None
        self.c13 = None
        self.c33 = None
        self.c44 = None
        
    def set_homogeneous(self, rho, vp, vs):
        """
        设置均匀介质
        """
        num_nodes = self.mesh.num_nodes
        self.rho = np.full(num_nodes, rho)
        self.vp = np.full(num_nodes, vp)
        self.vs = np.full(num_nodes, vs)
        
        self._update_elastic_params()
        print("The homogeneous media setting is complete.")
        
    def set_layered_model(self, layers, assign_to='element'):
        """
        设置层状模型
        """
        self.mesh.set_layered_model(layers, assign_to=assign_to)
        # 尝试同步 rho/vp/vs
        props = self.mesh.layer_properties
        for key in ('rho', 'vp', 'vs'):
            arr = props.get(key, None)
            if arr is None:
                raise ValueError(f"Layer property '{key}' not found in mesh.layer_properties.")
            setattr(self, key, arr)
        self._update_elastic_params()
        print(f"Layered model assigned to material ({assign_to}-wise).")

    def set_by_physical_groups(self, physical_group_properties):
        """
        根据Gmsh物理组设置材料属性
        
        参数:
            physical_group_properties: dict, 物理组名称到材料属性的映射
                格式: {
                    'Layer1': {'rho': 2700, 'vp': 5000, 'vs': 2887},
                    'Layer2': {'rho': 2500, 'vp': 4000, 'vs': 2309},
                }
        """
        if self.mesh.element_physical_group is None:
            raise ValueError("Mesh does not have physical group information. Load mesh from Gmsh first.")
        
        # 建立物理组名称到ID的映射
        name_to_id = {}
        for phys_id, info in self.mesh.physical_groups.items():
            name_to_id[info['name']] = phys_id
        
        # 初始化节点属性数组
        self.rho = np.zeros(self.mesh.num_nodes)
        self.vp = np.zeros(self.mesh.num_nodes)
        self.vs = np.zeros(self.mesh.num_nodes)
        
        assigned_elements = 0
        
        # 遍历每个单元
        for e in range(self.mesh.num_elements):
            phys_id = self.mesh.element_physical_group[e]
            
            # 查找物理组名称
            phys_name = None
            if phys_id in self.mesh.physical_groups:
                phys_name = self.mesh.physical_groups[phys_id]['name']
            
            # 查找对应的材料属性
            if phys_name and phys_name in physical_group_properties:
                props = physical_group_properties[phys_name]
                
                # 获取该单元的所有GLL节点
                gll_indices = self.mesh.connectivity[e, :, :].flatten()
                
                # 分配属性
                self.rho[gll_indices] = props['rho']
                self.vp[gll_indices] = props['vp']
                self.vs[gll_indices] = props['vs']
                
                assigned_elements += 1
        
        if assigned_elements == 0:
            print("Warning: No elements were assigned material properties!")
        else:
            print(f"Material properties assigned to {assigned_elements}/{self.mesh.num_elements} elements based on physical groups")
        
        # 更新弹性参数
        self._update_elastic_params()

    def _update_elastic_params(self):
        """根据rho, vp, vs计算其他弹性参数"""
        self.mu = self.rho * self.vs**2
        self.lambda_ = self.rho * self.vp**2 - 2.0 * self.mu
        lambdaplus2mu = self.lambda_ + 2.0 * self.mu
        
        self.c11 = lambdaplus2mu
        self.c13 = self.lambda_
        self.c33 = lambdaplus2mu
        self.c44 = self.mu
        
        # 标记为各向同性材料
        self.is_anisotropic = False
    
    def set_anisotropic(self, c11, c13, c33, c44, rho):
        """
        设置各向异性VTI介质的弹性参数
        
        对于VTI介质，弹性张量仅需4个独立参数：c11, c13, c33, c44
        
        参数:
            c11: float or array, 弹性常数 C11 (Pa)
            c13: float or array, 弹性常数 C13 (Pa)
            c33: float or array, 弹性常数 C33 (Pa)
            c44: float or array, 弹性常数 C44 (Pa)
            rho: float or array, 密度 (kg/m^3)
        """
        num_nodes = self.mesh.num_nodes
        
        # 将标量转换为数组
        self.c11 = np.full(num_nodes, c11) if np.isscalar(c11) else np.asarray(c11)
        self.c13 = np.full(num_nodes, c13) if np.isscalar(c13) else np.asarray(c13)
        self.c33 = np.full(num_nodes, c33) if np.isscalar(c33) else np.asarray(c33)
        self.c44 = np.full(num_nodes, c44) if np.isscalar(c44) else np.asarray(c44)
        self.rho = np.full(num_nodes, rho) if np.isscalar(rho) else np.asarray(rho)
        
        # 计算等效波速, 用于稳定性检查和PML设计
        # 对于VTI介质，使用垂直方向波速
        self.vp = np.sqrt(self.c33 / self.rho)
        self.vs = np.sqrt(self.c44 / self.rho)
        
        # 计算等效拉梅参数
        self.mu = self.c44.copy()
        self.lambda_ = self.c13.copy()
        
        # 标记为各向异性材料
        self.is_anisotropic = True
        
        print(f"Anisotropic (VTI) media setting complete:")
        print(f"  c11 range: [{self.c11.min():.2e}, {self.c11.max():.2e}] Pa")
        print(f"  c13 range: [{self.c13.min():.2e}, {self.c13.max():.2e}] Pa")
        print(f"  c33 range: [{self.c33.min():.2e}, {self.c33.max():.2e}] Pa")
        print(f"  c44 range: [{self.c44.min():.2e}, {self.c44.max():.2e}] Pa")
        print(f"  Equivalent vp range: [{self.vp.min():.1f}, {self.vp.max():.1f}] m/s")
        print(f"  Equivalent vs range: [{self.vs.min():.1f}, {self.vs.max():.1f}] m/s")
    
    def set_anisotropic_by_physical_groups(self, physical_group_properties):
        """
        根据Gmsh物理组设置各向异性材料属性
        
        参数:
            physical_group_properties: dict, 物理组名称到材料属性的映射
                格式: {
                    'Layer1': {'c11': 165e9, 'c13': 50e9, 'c33': 62e9, 'c44': 39.6e9, 'rho': 7100},
                    'Layer2': {'c11': 100e9, 'c13': 30e9, 'c33': 80e9, 'c44': 25e9, 'rho': 5000},
                }
        """
        if self.mesh.element_physical_group is None:
            raise ValueError("Mesh does not have physical group information.")
        
        # 建立物理组名称到ID的映射
        name_to_id = {}
        for phys_id, info in self.mesh.physical_groups.items():
            name_to_id[info['name']] = phys_id
        
        # 初始化数组
        num_nodes = self.mesh.num_nodes
        self.c11 = np.zeros(num_nodes)
        self.c13 = np.zeros(num_nodes)
        self.c33 = np.zeros(num_nodes)
        self.c44 = np.zeros(num_nodes)
        self.rho = np.zeros(num_nodes)
        
        assigned_elements = 0
        
        for e in range(self.mesh.num_elements):
            phys_id = self.mesh.element_physical_group[e]
            
            phys_name = None
            if phys_id in self.mesh.physical_groups:
                phys_name = self.mesh.physical_groups[phys_id]['name']
            
            if phys_name and phys_name in physical_group_properties:
                props = physical_group_properties[phys_name]
                gll_indices = self.mesh.connectivity[e, :, :].flatten()
                
                self.c11[gll_indices] = props['c11']
                self.c13[gll_indices] = props['c13']
                self.c33[gll_indices] = props['c33']
                self.c44[gll_indices] = props['c44']
                self.rho[gll_indices] = props['rho']
                
                assigned_elements += 1
        
        if assigned_elements == 0:
            print("Warning: No elements were assigned anisotropic material properties!")
        else:
            print(f"Anisotropic properties assigned to {assigned_elements}/{self.mesh.num_elements} elements")
        
        # 计算等效波速
        self.vp = np.sqrt(self.c33 / self.rho)
        self.vs = np.sqrt(self.c44 / self.rho)
        self.mu = self.c44.copy()
        self.lambda_ = self.c13.copy()
        
        self.is_anisotropic = True
    
    def _calculate_vp_vs(self, E, nu, rho):
        """
        根据弹模 E、泊松比 nu 和密度 rho 计算 Vp 和 Vs
        单位：SI 标准单位
            - E: Pa (或 N/m²)
            - nu: 无量纲
            - rho: kg/m³
            - Vp, Vs: m/s
        公式：
            mu = E / (2 * (1 + nu))
            lambda = E * nu / ((1 + nu) * (1 - 2*nu))
            Vp = sqrt((lambda + 2*mu) / rho)
            Vs = sqrt(mu / rho)
        """
        # 计算拉梅参数
        mu = E / (2.0 * (1.0 + nu))
        lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        # 计算波速
        vp = np.sqrt((lambda_ + 2.0 * mu) / rho)
        vs = np.sqrt(mu / rho)
        
        return vp, vs