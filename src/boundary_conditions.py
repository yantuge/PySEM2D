import numpy as np

class BoundaryConditions:
    """
    设置和管理边界条件
    """
    def __init__(self, params):
        self.params = params
        self.pml_dx = None
        self.pml_dz = None
        self.pml_dxx = None
        self.pml_dzz = None

    def setup_pml(self, assembler, material):
        """
        计算所有节点的PML参数。
        
        参数:
            assembler: Assembler对象，提供mesh对象, 包含GLL节点坐标
            material: Material对象，提供vp等材料参数
        """
        p = self.params
        # 使用Mesh对象中由SEMAssembler计算的GLL全局节点坐标
        coords = assembler.mesh.global_nodes
        num_nodes = len(coords)
        
        self.pml_dx = np.zeros(num_nodes)
        self.pml_dz = np.zeros(num_nodes)
        self.pml_dxx = np.zeros(num_nodes)
        self.pml_dzz = np.zeros(num_nodes)

        if not p['use_pml_or_not']:
            print("PML is not activated.")
            return

        # 获取各边界的PML控制参数
        use_pml_x_left = p.get('use_pml_x_left', True)
        use_pml_x_right = p.get('use_pml_x_right', True)
        use_pml_z_bottom = p.get('use_pml_z_bottom', True)
        # 顶部PML由enable_PML_top控制
        use_pml_z_top = p.get('use_pml_z_top', False)

        Rcoef = 0.0001
        vp_max = np.max(material.vp)

        xmin_pml = p['xmin'] + p['pml_x_thick']
        xmax_pml = p['xmax'] - p['pml_x_thick']
        zmin_pml = p['zmin'] + p['pml_z_thick']
        zmax_pml = p['zmax'] - p['pml_z_thick']

        d0_x = 3.0 * vp_max * np.log10(1.0 / Rcoef) / (2.0 * p['pml_x_thick'])
        d0_z = 3.0 * vp_max * np.log10(1.0 / Rcoef) / (2.0 * p['pml_z_thick'])

        for i in range(num_nodes):
            x, z = coords[i, 0], coords[i, 1]
            
            # X方向 - 左边界PML
            if x <= xmin_pml and use_pml_x_left:
                dist = (xmin_pml - x) / p['pml_x_thick']
                self.pml_dx[i] = d0_x * dist**2
                self.pml_dxx[i] = -d0_x * 2 * dist / p['pml_x_thick']
            # X方向 - 右边界PML
            elif x >= xmax_pml and use_pml_x_right:
                dist = (x - xmax_pml) / p['pml_x_thick']
                self.pml_dx[i] = d0_x * dist**2
                self.pml_dxx[i] = d0_x * 2 * dist / p['pml_x_thick']

            # Z方向 - 底边界PML
            if z <= zmin_pml and use_pml_z_bottom:
                dist = (zmin_pml - z) / p['pml_z_thick']
                self.pml_dz[i] = d0_z * dist**2
                self.pml_dzz[i] = -d0_z * 2 * dist / p['pml_z_thick']
            # Z方向 - 顶边界PML
            elif z >= zmax_pml and use_pml_z_top:
                dist = (z - zmax_pml) / p['pml_z_thick']
                self.pml_dz[i] = d0_z * dist**2
                self.pml_dzz[i] = d0_z * 2 * dist / p['pml_z_thick']

        print("PML setup completed.")