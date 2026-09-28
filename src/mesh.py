import numpy as np
from .gll import GLL
from .parsers import GmshParser

class Mesh:

    """
    管理计算网格, 包括生成、读取和存储网格信息
    """
    def __init__(self, params, gll):
        self.params = params
        self.gll = gll
        self.control_nodes = None      # 控制节点坐标 (knods_coords)
        self.elements = None           # 单元与控制节点的连接关系 (knods)
        self.connectivity = None       # 单元局部节点到全局节点的映射 (ibool)
        self.global_nodes = None       # 所有 GLL 节点的全局坐标
        self.num_nodes = 0
        self.num_elements = 0
        self.num_control_nodes = 0
        self.physical_groups = None    # 物理组信息, 从Gmsh读取时使用
        self.element_physical_group = None  # 每个单元的物理组ID

    def generate_regular_mesh(self):
        """
        生成规则矩形网格，并计算所有 GLL 节点的全局坐标
        """
        p = self.params
        NGLL = self.gll.order + 1
        
        # 1. 生成控制节点和单元
        nx = p['nelem_x'] + 1
        nz = p['nelem_z'] + 1
        self.num_control_nodes = nx * nz
        
        x_coords = np.linspace(p['xmin'], p['xmax'], nx)
        z_coords = np.linspace(p['zmin'], p['zmax'], nz)
        xv, zv = np.meshgrid(x_coords, z_coords)
        self.control_nodes = np.vstack([xv.ravel(), zv.ravel()]).T

        self.elements = np.zeros((p['nelem_x'] * p['nelem_z'], 4), dtype=int)
        for j in range(p['nelem_z']):
            for i in range(p['nelem_x']):
                el_idx = j * p['nelem_x'] + i
                n0 = j * nx + i          # 左下
                n1 = n0 + 1              # 右下
                n2 = (j + 1) * nx + i    # 左上
                n3 = n2 + 1              # 右上
                # 逆时针顺序: 左下(n0), 右下(n1), 右上(n3), 左上(n2)
                self.elements[el_idx, :] = [n0, n1, n3, n2]
        
        # 2. 建立 GLL 全局节点映射
        self.num_elements = p['nelem_x'] * p['nelem_z']
        self.num_nodes = (p['nelem_x'] * (NGLL - 1) + 1) * (p['nelem_z'] * (NGLL - 1) + 1)
        
        self.connectivity = np.zeros((self.num_elements, NGLL, NGLL), dtype=int)
        
        for j in range(p['nelem_z']):
            for i in range(p['nelem_x']):
                el_idx = j * p['nelem_x'] + i
                for lz in range(NGLL):
                    for lx in range(NGLL):
                        glob_j = j * (NGLL - 1) + lz
                        glob_i = i * (NGLL - 1) + lx
                        iglob = glob_j * (p['nelem_x'] * (NGLL - 1) + 1) + glob_i
                        # 注意索引顺序, 用(e,j,i) -> (element, z_idx, x_idx)
                        self.connectivity[el_idx, lz, lx] = iglob
        
        # 3. 计算所有 GLL 节点的全局坐标
        self.global_nodes = np.zeros((self.num_nodes, 2))
        ref_coords = self.gll.points
        
        # 形函数 N(xi, eta) = 0.25 * (1 +/- xi) * (1 +/- eta)
        N_functions = [
            lambda xi, eta: 0.25 * (1 - xi) * (1 - eta), # N1 (左下)
            lambda xi, eta: 0.25 * (1 + xi) * (1 - eta), # N2 (右下)
            lambda xi, eta: 0.25 * (1 + xi) * (1 + eta), # N3 (右上)
            lambda xi, eta: 0.25 * (1 - xi) * (1 + eta)  # N4 (左上)
        ]
        
        for e in range(self.num_elements):
            control_node_indices = self.elements[e, :]
            # 节点顺序已经是左下, 右下, 右上, 左上 (逆时针方向)
            el_control_nodes = self.control_nodes[control_node_indices, :]
            
            for j in range(NGLL): # eta
                for i in range(NGLL): # xi
                    iglob = self.connectivity[e, j, i]
                    xi, eta = ref_coords[i], ref_coords[j]
                    
                    # 使用形函数进行插值
                    coords = np.zeros(2)
                    for k in range(4):
                        coords += N_functions[k](xi, eta) * el_control_nodes[k, :]
                    self.global_nodes[iglob, :] = coords
                    
        print("The structured grid and GLL node coordinates have been generated.")

    def read_from_gmsh(self, filepath):
        """
        从Gmsh文件读取网格
        
        参数:
            filepath: str, Gmsh msh文件路径
        """
        NGLL = self.gll.order + 1
        
        # 使用GmshParser解析文件
        parser = GmshParser(filepath)
        coorg, knods, ibool = parser.parse(NGLLX=NGLL, NGLLZ=NGLL)
        
        # 保存物理组信息
        self.physical_groups = parser.physical_names
        self.element_physical_group = parser.get_element_physical_groups()
        
        # coorg: (2, N) - 所有节点的坐标
        # knods: (4, nelem) - 每个单元的4个角点 (几何控制节点)
        # ibool: (NGLLX, NGLLZ, nelem) - 每个单元的GLL节点全局编号
        
        self.num_control_nodes = coorg.shape[1]
        self.num_elements = knods.shape[1]
        
        # 转换为内部数据结构
        # control_nodes: (N, 2)
        self.control_nodes = coorg.T
        
        # elements: (nelem, 4) - 每个单元的4个角点索引（0-based）
        # knods是1-based，需要转换为0-based
        self.elements = (knods.T - 1).astype(int)
        
        # connectivity: (nelem, NGLL, NGLL) - 0-based索引
        # ibool是1-based，需要转换为0-based
        self.connectivity = (ibool - 1).astype(int)
        self.connectivity = np.transpose(self.connectivity, (2, 0, 1))  # (nelem, NGLLZ, NGLLX)
        
        # 计算全局节点数（ibool中的最大值+1）
        self.num_nodes = int(ibool.max())
        
        # 初始化全局节点坐标数组
        self.global_nodes = np.zeros((self.num_nodes, 2))
        
        # 临时使用Gmsh提供的坐标填充全局节点坐标
        # 注意, 这些坐标是Gmsh的等距节点，不是真正的GLL节点!!
        # SEMAssembler会通过等参变换重新计算正确的GLL节点坐标并覆盖此数组
        # 在SEMAssembler初始化之前，这些坐标可用于可视化网格拓扑
        for i in range(self.num_nodes):
            # coorg是(2, N)格式，索引是1-based到0-based
            self.global_nodes[i, :] = coorg[:, i]
        
        print(f"Gmsh mesh loaded: {self.num_elements} elements, {self.num_nodes} GLL nodes")
        print(f"  Note: Node coordinates are from Gmsh (equidistant). SEMAssembler will recompute GLL coordinates.")
        if self.physical_groups:
            print(f"Physical groups found: {list(self.physical_groups.values())}")
        
    def set_layered_model(self, layers, assign_to='element'):
        """
        为网格设置层状模型
        """
        # 校验基础数据是否存在
        if self.control_nodes is None or self.elements is None:
            raise ValueError("Control nodes and elements not generated. Call generate_regular_mesh() first.")

        if assign_to not in ('element', 'node'):
            raise ValueError("assign_to must be 'element' or 'node'.")

        # 简单校验 layers 格式
        if not isinstance(layers, (list, tuple)) or len(layers) == 0:
            raise ValueError("layers must be a non-empty list of layer dicts.")

        # 规范层定义并保存
        processed_layers = []
        for idx, L in enumerate(layers):
            if not isinstance(L, dict):
                raise ValueError(f"Each layer must be a dict, got {type(L)} at index {idx}")
            if 'z_top' not in L or 'z_bottom' not in L or 'properties' not in L:
                raise ValueError("Each layer dict must contain 'z_top', 'z_bottom' and 'properties' keys.")
            processed_layers.append({
                'z_top': float(L['z_top']),
                'z_bottom': float(L['z_bottom']),
                'properties': dict(L['properties'])
            })

        # 保存原始层信息
        self.layers = processed_layers

        # 准备目标索引容器
        if assign_to == 'element':
            mat_idx = np.zeros(self.num_elements, dtype=int)

            # 以控制节点的平均高度作为单元高度判定
            for e in range(self.num_elements):
                cn_idx = self.elements[e, :]
                zc = float(np.mean(self.control_nodes[cn_idx, 1]))
                assigned = False
                for li, layer in enumerate(self.layers):
                    ztop = layer['z_top']
                    zbot = layer['z_bottom']
                    # 支持任意 z 顺序（top 起始可能比 bottom 大或小）
                    zmin, zmax = min(ztop, zbot), max(ztop, zbot)
                    if zc >= zmin and zc <= zmax:
                        mat_idx[e] = li
                        assigned = True
                        break
                if not assigned:
                    # 若未落入任一层，设为 -1 表示未分配
                    mat_idx[e] = -1

            self.element_material = mat_idx
            target_len = self.num_elements
        else:
            # 按全局 GLL 节点的 z 坐标分配
            if self.global_nodes is None:
                raise ValueError("Global GLL nodes not generated. Call generate_regular_mesh() first.")
            mat_idx = np.zeros(self.num_nodes, dtype=int)
            for n in range(self.num_nodes):
                zn = float(self.global_nodes[n, 1])
                assigned = False
                for li, layer in enumerate(self.layers):
                    ztop = layer['z_top']
                    zbot = layer['z_bottom']
                    zmin, zmax = min(ztop, zbot), max(ztop, zbot)
                    if zn >= zmin and zn <= zmax:
                        mat_idx[n] = li
                        assigned = True
                        break
                if not assigned:
                    mat_idx[n] = -1

            self.node_material = mat_idx
            target_len = self.num_nodes

        # 收集所有属性键并为每个属性生成数组
        prop_keys = set()
        for layer in self.layers:
            prop_keys.update(layer['properties'].keys())

        layer_props = {}
        for key in prop_keys:
            arr = np.full(target_len, np.nan, dtype=float)
            if assign_to == 'element':
                for e in range(self.num_elements):
                    li = self.element_material[e]
                    if li >= 0:
                        val = self.layers[li]['properties'].get(key, np.nan)
                        arr[e] = float(val) if val is not None else np.nan
            else:
                for n in range(self.num_nodes):
                    li = self.node_material[n]
                    if li >= 0:
                        val = self.layers[li]['properties'].get(key, np.nan)
                        arr[n] = float(val) if val is not None else np.nan
            layer_props[key] = arr

        # 保存属性映射
        self.layer_properties = layer_props

        print(f"Layered model assigned to {assign_to}s. Number of layers: {len(self.layers)}")