import numpy as np
import re


class GmshParser:
    """
    解析Gmsh 2.2格式的网格文件
    """

    def __init__(self, filepath):
        """
        初始化Gmsh解析器

        参数:
            filepath: str, msh文件的路径
        """
        self.filepath = filepath
        self.nodes = None           # 节点坐标数组 (N, 3)
        self.elements = None        # 原始单元数据
        self.index_n_modes = None  # N阶四边形单元的节点局部编号顺序
        self.elem_type_id = None    # N阶四边形单元的Gmsh类型ID
        self.physical_names = {}    # 物理组名称映射 {id: name}
        self.coorg = None          # 节点坐标 (2, N)
        self.knods = None          # 几何控制节点连接 (4, nelem)
        self.ibool = None          # GLL节点全局编号 (NGLLX, NGLLZ, nelem)

    def parse(self, NGLLX=5, NGLLZ=5):
        """
        解析Gmsh网格文件
        
        参数:
            NGLLX: int, x方向GLL点数, 默认5
            NGLLZ: int, z方向GLL点数, 默认5
            
        返回:
            coorg: 节点坐标 (2, N)
            knods: 几何控制节点 (4, nelem)
            ibool: GLL节点编号 (NGLLX, NGLLZ, nelem)
        """
        #if NGLLX != 5 or NGLLZ != 5:
        #    raise ValueError('Only 5 points element is supported, while other cases are under development')

        with open(self.filepath, 'r') as fp:
            content = fp.read()

        # 解析物理组名称
        self._parse_physical_names(content)

        # 解析节点
        self._parse_nodes(content)

        # 解析单元
        self._get_index_n_modes(NGLLX)
        self._parse_elements(content, NGLLX, NGLLZ)

        return self.coorg, self.knods, self.ibool

    def _parse_physical_names(self, content):
        """
        解析物理组名称段
        """
        match = re.search(r'\$PhysicalNames\s+(.*?)\$EndPhysicalNames', content, re.DOTALL)
        if match:
            lines = match.group(1).strip().split('\n')
            num_names = int(lines[0])
            for i in range(1, num_names + 1):
                parts = lines[i].split()
                if len(parts) >= 3:
                    dimension = int(parts[0])
                    phys_id = int(parts[1])
                    name = parts[2].strip('"')
                    self.physical_names[phys_id] = {'name': name, 'dimension': dimension}

    def _parse_nodes(self, content):
        """
        解析节点坐标段
        """
        # 提取$Nodes...$EndNodes段
        match = re.search(r'\$Nodes\s+(.*?)\$EndNodes', content, re.DOTALL)
        if not match:
            raise ValueError('Cannot find $Nodes section in mesh file')

        lines = match.group(1).strip().split('\n')
        node_num = int(lines[0])

        # 初始化节点数组
        nodes = np.zeros((node_num, 3))

        for i in range(1, node_num + 1):
            parts = lines[i].split()
            node_id = int(parts[0])
            if node_id != i:
                raise ValueError(f'Wrong node number! Expected {i}, got {node_id}')

            nodes[i - 1, 0] = float(parts[1])  # x
            nodes[i - 1, 1] = float(parts[2])  # y
            nodes[i - 1, 2] = float(parts[3])  # z

        self.nodes = nodes
        # 转置为(2, N)格式，只取x和y坐标
        self.coorg = nodes[:, :2].T

    def _parse_elements(self, content, NGLLX, NGLLZ):
        """
        解析单元段，提取N阶四边形单元，并重新编号节点
        """
        # 提取$Elements...$EndElements段
        match = re.search(r'\$Elements\s+(.*?)\$EndElements', content, re.DOTALL)
        if not match:
            raise ValueError('Cannot find $Elements section in mesh file')

        lines = match.group(1).strip().split('\n')
        total_elements = int(lines[0])

        # 存储原始单元数据用于后续处理
        all_elements = []
        
        # 单元类型ID列表
        # elem_type_list = [3, 10, 36, 37, 38, 47, 48, 49]  # 包括常见的单元类型
        # 对应 [4-node quad, 9-node quad, 16-node quad, 25-node quad, 36-node quad, 49-node quad, 64-node quad, 81-node quad]

        # 收集所有单元，识别N阶四边形单元(elem_type_list)
        quadN_elements = []
        
        for i in range(1, total_elements + 1):
            parts = lines[i].split()
            elem_id = int(parts[0])
            elem_type = int(parts[1])
            num_tags = int(parts[2])

            # 提取标签, 通常包含物理组ID和几何实体ID
            tags = [int(parts[3 + j]) for j in range(num_tags)]

            # 提取节点编号
            node_start_idx = 3 + num_tags
            nodes = [int(parts[j]) for j in range(node_start_idx, len(parts))]

            elem_data = {
                'id': elem_id,
                'type': elem_type,
                'tags': tags,
                'nodes': nodes
            }
            all_elements.append(elem_data)

            # Type 37 = 25节点四边形单元
            #if elem_type == 37 and len(nodes) == 25:
            node_num = len(self.index_n_modes)
            if elem_type == self.elem_type_id and len(nodes) == node_num:
                quadN_elements.append(elem_data)

        if len(quadN_elements) == 0:
            raise ValueError(f'No {node_num}-node quadrilateral elements (type {self.elem_type_id}) found in mesh file')

        element_num = len(quadN_elements)
        print(f'Found {element_num} quadrilateral elements with {node_num} nodes')

        # 初始化ibool数组, 使用原始节点编号
        ibool = np.zeros((element_num, NGLLX * NGLLZ), dtype=int)

        for idx, elem in enumerate(quadN_elements):
            # Gmsh节点编号从1开始
            ibool[idx, :] = elem['nodes']

        # 重新编号节点，只保留在25节点四边形单元中使用的节点
        # 找出所有被使用的节点
        used_nodes = np.unique(ibool.flatten())
        num_used_nodes = len(used_nodes)
        
        print(f'Total nodes in mesh file: {len(self.nodes)}')
        print(f'Nodes used in quadrilateral elements: {num_used_nodes}')
        
        if num_used_nodes < len(self.nodes):
            print(f'Renumbering nodes: {len(self.nodes)} -> {num_used_nodes}')
            
            # 创建旧节点ID到新节点ID的映射 (1-based)
            old_to_new = {}
            for new_id, old_id in enumerate(used_nodes, start=1):
                old_to_new[old_id] = new_id
            
            # 重新编号ibool中的节点
            for i in range(element_num):
                for j in range(NGLLX * NGLLZ):
                    old_id = ibool[i, j]
                    ibool[i, j] = old_to_new[old_id]
            
            # 重新构建节点坐标数组，只包含使用的节点
            new_nodes = np.zeros((num_used_nodes, 3))
            for new_id, old_id in enumerate(used_nodes, start=1):
                # old_id是1-based，self.nodes是0-based索引
                new_nodes[new_id - 1, :] = self.nodes[old_id - 1, :]
            
            # 更新节点数组和坐标
            self.nodes = new_nodes
            self.coorg = new_nodes[:, :2].T
        
        # 验证节点编号范围
        if ibool.min() != 1 or ibool.max() != len(self.nodes):
            raise ValueError(f'Element node numbering is wrong! Min: {ibool.min()}, Max: {ibool.max()}, Expected max: {len(self.nodes)}')

        # 提取几何控制节点, 即4个角点
        # 取前4个节点作为几何控制点，并逆序排列
        knods = np.zeros((element_num, 4), dtype=int)
        for i in range(element_num):
            knods[i, :] = ibool[i, :4][::-1]

        # 应用节点顺序转换
        # index_change_25_nodes = [4, 14, 15, 16, 1, 13, 20, 24, 17, 5, 12, 23, 25, 21, 6, 11, 19, 22, 18, 7, 3, 10, 9, 8, 2]
        #index_change_25_nodes = np.array([4, 14, 15, 16, 1, 13, 20, 24, 17, 5, 12, 23, 25, 21, 6, 11, 19, 22, 18, 7, 3, 10, 9, 8, 2]) - 1
        #index_change_25_nodes = np.array([4, 11, 12, 1, 10, 16, 13, 5, 9, 15, 14, 6, 3, 8, 7, 2]) - 1
        
        for i in range(element_num):
            ibool[i, :] = ibool[i, self.index_n_modes]

        # 重塑ibool为(NGLLX, NGLLZ, nelem)格式
        ibool = ibool.T.reshape(NGLLX, NGLLZ, element_num)
        ibool = np.transpose(ibool, (1, 0, 2)) 

        # 转置knods为(4, nelem)
        self.knods = knods.T
        self.ibool = ibool

        # 保存所有单元信息以便后续使用
        self.elements = all_elements
        self.quadN_elements = quadN_elements
        
    def _get_index_n_modes(self,NGLL=5):
        """
        获取N阶四边形单元的节点局部编号顺序和单元类型ID
        
        返回:
            index_n_modes: ndarray, shape (NGLL^2,), (NGLL-1)阶四边形单元的局部节点编号顺序
        """
        # 根据单元阶数生成节点索引
        match NGLL:
            case 2: # 4-node quadrilateral
                self.index_n_modes = np.array([4, 1, 
                                               3, 2]) - 1  # 转为0-based索引
                self.elem_type_id = 3
            case 3: # 9-node quadrilateral
                self.index_n_modes = np.array([4, 8, 1,
                                               7, 9, 5,
                                               3, 6, 2]) - 1  # 转为0-based索引
                self.elem_type_id = 10
            case 4: # 16-node quadrilateral
                self.index_n_modes = np.array([4,  11, 12, 1,
                                               10, 16, 13, 5,
                                               9,  15, 14, 6,
                                               3,  8,  7,  2]) - 1  # 转为0-based索引
                self.elem_type_id = 36
            case 5: # 25-node quadrilateral
                self.index_n_modes = np.array([4,  14, 15, 16, 1, 
                                               13, 20, 24, 17, 5, 
                                               12, 23, 25, 21, 6, 
                                               11, 19, 22, 18, 7,
                                               3,  10, 9,  8,  2]) - 1  # 转为0-based索引
                self.elem_type_id = 37
            case 6: # 36-node quadrilateral
                self.index_n_modes = np.array([4,  17, 18, 19, 20, 1,
                                               16, 24, 31, 32, 21, 5,
                                               15, 30, 36, 33, 25, 6,
                                               14, 29, 35, 34, 26, 7,
                                               13, 23, 28, 27, 22, 8,
                                               3,  12, 11, 10, 9,  2]) - 1  # 转为0-based索引
                self.elem_type_id = 38
            case 7: # 49-node quadrilateral
                self.index_n_modes = np.array([4,  20, 21, 22, 23, 24, 1,
                                               19, 28, 38, 39, 40, 25, 5,
                                               18, 37, 44, 48, 41, 29, 6,
                                               17, 36, 47, 49, 45, 30, 7,
                                               16, 35, 43, 46, 42, 31, 8,
                                               15, 27, 34, 33, 32, 26, 9,
                                               3,  14, 13, 12, 11, 10, 2]) - 1  # 转为0-based索引
                self.elem_type_id = 47
            case 8: # 64-node quadrilateral
                self.index_n_modes = np.array([4,  23, 24, 25, 26, 27, 28, 1,
                                               22, 32, 45, 46, 47, 48, 29, 5,
                                               21, 44, 52, 59, 60, 49, 33, 6,
                                               20, 43, 58, 64, 61, 53, 34, 7,
                                               19, 42, 57, 63, 62, 54, 35, 8,
                                               18, 41, 51, 56, 55, 50, 36, 9,
                                               17, 31, 40, 39, 38, 37, 30, 10,
                                               3,  16, 15, 14, 13, 12, 11, 2]) - 1  # 转为0-based索引
                self.elem_type_id = 48
            case 9: # 81-node quadrilateral
                self.index_n_modes = np.array([4,  26, 27, 28, 29, 30, 31, 32, 1,
                                               25, 36, 52, 53, 54, 55, 56, 33, 5,
                                               24, 51, 60, 70, 71, 72, 57, 37, 6,
                                               23, 50, 69, 76, 80, 73, 61, 38, 7,
                                               22, 49, 68, 79, 81, 77, 62, 39, 8,
                                               21, 48, 67, 75, 78, 74, 63, 40, 9,
                                               20, 47, 59, 66, 65, 64, 58, 41, 10,
                                               19, 35, 46, 45, 44, 43, 42, 34, 11,
                                               3,  18, 17, 16, 15, 14, 13, 12, 2]) - 1  # 转为0-based索引
                self.elem_type_id = 49
            case _:
                raise ValueError(f'Unsupported NGLL value: {NGLL}')

        #return index_n_modes, elem_type_id

    def get_element_physical_groups(self):
        """
        获取每个25节点四边形单元所属的物理组
        
        返回:
            element_physical_group: ndarray, shape (nelem,), 每个单元的物理组ID
        """
        if not hasattr(self, 'quadN_elements'):
            raise ValueError('Elements not parsed yet. Call parse() first.')

        nelem = len(self.quadN_elements)
        element_physical_group = np.zeros(nelem, dtype=int)

        for idx, elem in enumerate(self.quadN_elements):
            # 物理组ID通常是tags的第一个元素
            if len(elem['tags']) > 0:
                element_physical_group[idx] = elem['tags'][0]
            else:
                element_physical_group[idx] = 0  # 未分配物理组

        return element_physical_group
