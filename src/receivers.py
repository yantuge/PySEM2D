import numpy as np
import os
from .utils import find_nearest_node_indices
from .io import DataIO


class Receivers:
    """
    管理接收点的位置并记录地震图数据
    """
    def __init__(self, params, mesh, output_dir='results'):
        self.params = params
        self.mesh = mesh
        self.ndim = params['NDIM']
        self.nt = params['nt']
        self.output_dir = output_dir
        self.stations_dir = os.path.join(output_dir, 'stations')
        
        # 所有接收器的汇总信息
        self.all_names = []        # 所有设置测站名称, S开头
        self.all_locations = []    # 所有测站目标坐标
        self.all_actual_locations = []  # 所有测站实际坐标(网格节点)
        self.all_indices = []      # 所有测站在网格中的索引
        self.all_errors = []       # 所有测站的定位误差(欧氏距离)
        self.rec_types = []        # 每个测站的记录类型列表
        
        # 实际测站信息（去重后，用于数据记录）
        self.actual_station_names = []    # 实际测站名称, A开头
        self.actual_station_indices = []  # 实际测站的网格节点索引
        self.actual_station_rec_types = []  # 实际测站的记录类型
        self.setup_to_actual_map = []     # 设置测站到实际测站的映射索引
        
        # 接收器组元数据
        self.group_configs = []           # 每个组的配置信息
        self.actual_station_groups = []   # 每个实际测站所属的组索引列表
        
        # 解析接收器组配置
        self._parse_receiver_groups()
        
        # 转换为numpy数组
        self.all_locations = np.array(self.all_locations)
        self.all_actual_locations = np.array(self.all_actual_locations)
        self.all_indices = np.array(self.all_indices)
        self.all_errors = np.array(self.all_errors)
        self.setup_to_actual_map = np.array(self.setup_to_actual_map)
        self.actual_station_indices = np.array(self.actual_station_indices)
        
        # 初始化数据存储, 基于实际测站数量
        num_actual_receivers = len(self.actual_station_names)
        
        # 总是记录位移数据, 即使rec_type中没有1
        self.displacement = [np.zeros((self.ndim, self.nt)) for _ in range(num_actual_receivers)]
        
        # 仅在需要时分配速度数据
        self.velocity = None
        if self._needs_velocity():
            self.velocity = [np.zeros((self.ndim, self.nt)) for _ in range(num_actual_receivers)]
        
        # 仅在需要时分配加速度数据
        self.acceleration = None
        if self._needs_acceleration():
            self.acceleration = [np.zeros((self.ndim, self.nt)) for _ in range(num_actual_receivers)]
        
        print(f"\nReceivers initialized: {len(self.all_names)} setup stations -> {num_actual_receivers} actual stations")
        # self._print_summary() # 取消自动打印摘要，以减少冗余输出
    
    def _parse_receiver_groups(self):
        """
        解析接收器组配置
        """
        params = self.params
        
        if 'receiver_groups' in params and params['receiver_groups']:
            groups = params['receiver_groups']
            station_counter = 1  # 全局测站编号计数器
            
            for group_idx, group_config in enumerate(groups):
                group_name = group_config.get('name', f'group_{group_idx}')
                group_type = group_config.get('type', 'line')
                rec_type = group_config.get('rec_type', [1])  # 默认只记录位移
                
                # 确保rec_type是列表
                if isinstance(rec_type, int):
                    rec_type = [rec_type]
                
                # 保存组配置信息
                self.group_configs.append({
                    'name': group_name,
                    'type': group_type,
                    'rec_type': rec_type,
                    'group_idx': group_idx
                })
                
                print(f"\nProcessing receiver group '{group_name}' (type: {group_type})")
                
                if group_type == 'line':
                    names, locations = self._define_from_line(group_config, station_counter)
                elif group_type == 'file':
                    names, locations = self._define_from_file(group_config)
                else:
                    raise ValueError(f"Unknown receiver group type: {group_type}")
                
                # 查找最近节点
                indices = find_nearest_node_indices(locations, self.mesh.global_nodes)
                
                # 获取实际坐标 (网格节点坐标)
                actual_locations = self.mesh.global_nodes[indices]
                
                # 计算定位误差
                errors = np.sqrt(np.sum((locations - actual_locations)**2, axis=1))
                
                # 添加到汇总列表
                self.all_names.extend(names)
                self.all_locations.extend(locations.tolist())
                self.all_actual_locations.extend(actual_locations.tolist())
                self.all_indices.extend(indices.tolist())
                self.all_errors.extend(errors.tolist())
                self.rec_types.extend([rec_type] * len(names))
                
                # 更新计数器
                station_counter += len(names)
                
                # 打印距离反馈
                self._print_location_feedback(group_name, names, locations, actual_locations, errors)
                
                # 自动保存测站文件到输出目录下的 stations/ 文件夹
                # 保存两个文件：设置测站和实际测站
                setup_file = os.path.join(self.stations_dir, f'{group_name}_setup.txt')
                actual_file = os.path.join(self.stations_dir, f'{group_name}_actual.txt')
                
                # 保存设置测站(目标坐标)
                DataIO.write_station_file(setup_file, names, locations)
                
                # 移除重复的网格节点坐标，并获取映射关系
                unique_actual_locs, unique_indices, setup_to_actual = self._remove_duplicate_stations(
                    actual_locations, indices
                )
                
                # 生成连续的实际测站名称（A0001, A0002...）
                actual_start_index = len(self.actual_station_names) + 1
                actual_names = DataIO.generate_station_names(
                    len(unique_actual_locs), start_index=actual_start_index, prefix='A'
                )
                
                # 保存实际测站信息到类属性
                self.actual_station_names.extend(actual_names)
                self.actual_station_indices.extend(unique_indices.tolist())
                self.actual_station_rec_types.extend([rec_type] * len(actual_names))
                
                # 通过反向映射记录每个实际测站所属的组
                # setup_to_actual 记录了每个设置测站对应的实际测站索引
                # 需要将其转换为绝对索引并建立组关联
                for actual_rel_idx in range(len(unique_actual_locs)):
                    actual_abs_idx = actual_start_index - 1 + actual_rel_idx
                    # 初始化该实际测站的组列表
                    while len(self.actual_station_groups) <= actual_abs_idx:
                        self.actual_station_groups.append([])
                    # 添加当前组到该实际测站的组列表
                    self.actual_station_groups[actual_abs_idx].append(group_idx)
                
                # 保存设置测站到实际测站的映射
                offset = actual_start_index - 1
                self.setup_to_actual_map.extend((setup_to_actual + offset).tolist())
                
                # 保存实际测站
                DataIO.write_station_file(actual_file, actual_names, unique_actual_locs)
                
                if len(unique_actual_locs) < len(names):
                    print(f"  Note: {len(names) - len(unique_actual_locs)} duplicate stations removed")
                    print(f"  Actual stations: {len(unique_actual_locs)} (A{actual_start_index:04d}-A{actual_start_index+len(unique_actual_locs)-1:04d})")
                
                print(f"  Added {len(names)} setup stations with rec_type={rec_type}")
        
        else:
            # 旧格式
            print("\nUsing legacy receiver configuration (rec_n + rec_z)")
            rec_n = params.get('rec_n', 101)
            rec_z = params.get('rec_z', 0.0)
            rec_type = params.get('rec_type', [1])  # 默认只记录位移
            
            if isinstance(rec_type, int):
                rec_type = [rec_type]
            
            # 保存组配置信息
            self.group_configs.append({
                'name': 'legacy_receivers',
                'type': 'line',
                'rec_type': rec_type,
                'group_idx': 0
            })
            
            # 定义水平线接收器
            locations = np.zeros((rec_n, 2))
            locations[:, 0] = np.linspace(params['xmin'], params['xmax'], rec_n)
            locations[:, 1] = rec_z
            
            # 生成测站名称
            names = DataIO.generate_station_names(rec_n, start_index=1)
            
            # 查找最近节点
            indices = find_nearest_node_indices(locations, self.mesh.global_nodes)
            
            # 获取实际坐标
            actual_locations = self.mesh.global_nodes[indices]
            
            # 计算定位误差
            errors = np.sqrt(np.sum((locations - actual_locations)**2, axis=1))
            
            # 添加到汇总列表
            self.all_names.extend(names)
            self.all_locations.extend(locations.tolist())
            self.all_actual_locations.extend(actual_locations.tolist())
            self.all_indices.extend(indices.tolist())
            self.all_errors.extend(errors.tolist())
            self.rec_types.extend([rec_type] * rec_n)
            
            # 打印距离反馈
            self._print_location_feedback('legacy_receivers', names, locations, actual_locations, errors)
            
            # 自动保存测站文件到输出目录下的 stations/ 文件夹
            setup_file = os.path.join(self.stations_dir, 'legacy_receivers_setup.txt')
            actual_file = os.path.join(self.stations_dir, 'legacy_receivers_actual.txt')
            
            # 保存设置测站
            DataIO.write_station_file(setup_file, names, locations)
            
            # 去重实际测站并获取映射
            unique_actual_locs, unique_indices, setup_to_actual = self._remove_duplicate_stations(
                actual_locations, indices
            )
            
            # 生成连续的实际测站名称: A0001, A0002...
            actual_start_index = len(self.actual_station_names) + 1
            actual_names = DataIO.generate_station_names(
                len(unique_actual_locs), start_index=actual_start_index, prefix='A'
            )
            
            # 保存实际测站信息
            self.actual_station_names.extend(actual_names)
            self.actual_station_indices.extend(unique_indices.tolist())
            self.actual_station_rec_types.extend([rec_type] * len(actual_names))
            offset = actual_start_index - 1
            self.setup_to_actual_map.extend((setup_to_actual + offset).tolist())
            
            # 记录每个实际测站所属的组
            for actual_rel_idx in range(len(unique_actual_locs)):
                actual_abs_idx = actual_start_index - 1 + actual_rel_idx
                while len(self.actual_station_groups) <= actual_abs_idx:
                    self.actual_station_groups.append([])
                self.actual_station_groups[actual_abs_idx].append(0)
            
            # 保存实际测站
            DataIO.write_station_file(actual_file, actual_names, unique_actual_locs)
            
            if len(unique_actual_locs) < rec_n:
                print(f"  Note: {rec_n - len(unique_actual_locs)} duplicate stations removed")
                print(f"  Actual stations: {len(unique_actual_locs)} (A{actual_start_index:04d}-A{actual_start_index+len(unique_actual_locs)-1:04d})")
            
            print(f"  Added {rec_n} setup stations with rec_type={rec_type}")
    
    def _remove_duplicate_stations(self, locations, indices):
        """
        移除重复的测站坐标, 并返回映射关系
        
        参数:
            locations: np.ndarray, 坐标数组 (num, 2)
            indices: np.ndarray, 网格节点索引数组 (num,)
            
        返回:
            unique_locations: np.ndarray, 去重后的坐标数组
            unique_indices: np.ndarray, 去重后的网格节点索引
            setup_to_actual_map: np.ndarray, 设置测站到实际测站的映射 (num,)
        """
        unique_locations = []
        unique_indices_list = []
        seen_coords = {}  # coord_key -> actual_station_index
        setup_to_actual_map = []
        
        for i in range(len(locations)):
            # 将坐标转换为元组用于集合比较
            coord_key = (round(locations[i, 0], 6), round(locations[i, 1], 6))
            
            if coord_key not in seen_coords:
                # 第一次见到这个坐标，添加到唯一列表
                actual_idx = len(unique_locations)
                seen_coords[coord_key] = actual_idx
                unique_locations.append(locations[i])
                unique_indices_list.append(indices[i])
                setup_to_actual_map.append(actual_idx)
            else:
                # 已见过的，使用映射
                setup_to_actual_map.append(seen_coords[coord_key])
        
        return np.array(unique_locations), np.array(unique_indices_list), np.array(setup_to_actual_map)
    
    def _print_location_feedback(self, group_name, names, target_locs, actual_locs, errors):
        """
        打印接收器定位反馈信息
        
        参数:
            group_name: str, 接收器组名称
            names: list, 测站名称列表
            target_locs: np.ndarray, 目标坐标 (num, 2)
            actual_locs: np.ndarray, 实际坐标 (num, 2)
            errors: np.ndarray, 定位误差 (num,)
        """
        num_stations = len(names)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        max_error_idx = np.argmax(errors)
        
        print(f"\n  RECEIVER LOCATION ANALYSIS - Group '{group_name}'")
        print(f"  " + "="*56)
        
        # 打印前3个和后3个测站的详细信息
        num_to_show = min(3, num_stations)
        for i in range(num_to_show):
            print(f"  {names[i]}: Target=({target_locs[i, 0]:.2f}, {target_locs[i, 1]:.2f}) → "
                  f"Actual=({actual_locs[i, 0]:.2f}, {actual_locs[i, 1]:.2f}), Error={errors[i]:.2f} m")
        
        if num_stations > 6:
            print(f"  ... ({num_stations - 6} stations omitted) ...")
            
            for i in range(num_stations - num_to_show, num_stations):
                print(f"  {names[i]}: Target=({target_locs[i, 0]:.2f}, {target_locs[i, 1]:.2f}) → "
                      f"Actual=({actual_locs[i, 0]:.2f}, {actual_locs[i, 1]:.2f}), Error={errors[i]:.2f} m")
        elif num_stations > num_to_show:
            for i in range(num_to_show, num_stations):
                print(f"  {names[i]}: Target=({target_locs[i, 0]:.2f}, {target_locs[i, 1]:.2f}) → "
                      f"Actual=({actual_locs[i, 0]:.2f}, {actual_locs[i, 1]:.2f}), Error={errors[i]:.2f} m")
        
        print(f"  " + "-"*56)
        print(f"  Summary: Mean error={mean_error:.2f} m, Max error={max_error:.2f} m ({names[max_error_idx]})")
        print(f"  " + "="*56)
    
    def _define_from_line(self, group_config, start_index):
        """
        从起止坐标和数量定义接收器
        
        参数:
            group_config: dict, 包含 'start', 'end', 'num'
            start_index: int, 测站编号起始值
            
        返回:
            names: list, 测站名称
            locations: np.ndarray, 坐标数组 (num, 2)
        """
        start = np.array(group_config['start'])
        end = np.array(group_config['end'])
        num = group_config['num']
        
        # 生成线性分布的坐标
        locations = np.zeros((num, 2))
        locations[:, 0] = np.linspace(start[0], end[0], num)
        locations[:, 1] = np.linspace(start[1], end[1], num)
        
        # 生成测站名称
        names = DataIO.generate_station_names(num, start_index=start_index)
        # 注意：此时尚未计算实际坐标，需要在调用处保存
        
        return names, locations
    
    def _define_from_file(self, group_config):
        """
        从测站文件读取接收器, 总是读取设置测站坐标
        
        参数:
            group_config: dict, 包含 'file_path'
            
        返回:
            names: list, 测站名称
            locations: np.ndarray, 目标坐标数组 (num, 2)
        """
        file_path = group_config['file_path']
        names, locations = DataIO.read_station_file(file_path)
        print(f"  Loaded {len(names)} target station locations from {file_path}")
        return names, locations
    
    def _needs_velocity(self):
        """检查是否有任何测站需要记录速度"""
        for rec_type_list in self.rec_types:
            if 2 in rec_type_list:
                return True
        return False
    
    def _needs_acceleration(self):
        """检查是否有任何测站需要记录加速度"""
        for rec_type_list in self.rec_types:
            if 3 in rec_type_list:
                return True
        return False
    
    def _print_summary(self):
        """打印接收器配置摘要"""
        print("\n" + "="*60)
        print("RECEIVER CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Total stations: {len(self.all_names)}")
        print(f"\nTarget coordinate range:")
        print(f"  X: [{self.all_locations[:, 0].min():.2f}, {self.all_locations[:, 0].max():.2f}] m")
        print(f"  Z: [{self.all_locations[:, 1].min():.2f}, {self.all_locations[:, 1].max():.2f}] m")
        print(f"\nActual coordinate range (on mesh nodes):")
        print(f"  X: [{self.all_actual_locations[:, 0].min():.2f}, {self.all_actual_locations[:, 0].max():.2f}] m")
        print(f"  Z: [{self.all_actual_locations[:, 1].min():.2f}, {self.all_actual_locations[:, 1].max():.2f}] m")
        print(f"\nLocation accuracy:")
        print(f"  Mean error: {np.mean(self.all_errors):.3f} m")
        print(f"  Max error: {np.max(self.all_errors):.3f} m")
        print(f"  Min error: {np.min(self.all_errors):.3f} m")
        
        # 统计记录类型
        rec_type_count = {1: 0, 2: 0, 3: 0}
        for rec_type_list in self.rec_types:
            for rt in rec_type_list:
                if rt in rec_type_count:
                    rec_type_count[rt] += 1
        
        print(f"\nRecording types:")
        print(f"  Displacement (type=1): {rec_type_count[1]} stations")
        print(f"  Velocity (type=2): {rec_type_count[2]} stations")
        print(f"  Acceleration (type=3): {rec_type_count[3]} stations")
        
        print(f"\nMemory allocation:")
        print(f"  Displacement: allocated")
        print(f"  Velocity: {'allocated' if self.velocity is not None else 'not allocated'}")
        print(f"  Acceleration: {'allocated' if self.acceleration is not None else 'not allocated'}")
        print("="*60 + "\n")
    
    def record(self, it, displacement, velocity=None, acceleration=None):
        """
        基于实际测站记录当前时间步的数据
        
        参数:
            it: int, 时间步索引
            displacement: np.ndarray, 位移场 (num_nodes, ndim)
            velocity: np.ndarray or None, 速度场 (num_nodes, ndim)
            acceleration: np.ndarray or None, 加速度场 (num_nodes, ndim)
        """
        # 记录位移, 使用实际测站索引
        for i, node_idx in enumerate(self.actual_station_indices):
            self.displacement[i][:, it] = displacement[node_idx, :]
        
        # 记录速度
        if velocity is not None and self.velocity is not None:
            for i, node_idx in enumerate(self.actual_station_indices):
                self.velocity[i][:, it] = velocity[node_idx, :]
        
        # 记录加速度
        if acceleration is not None and self.acceleration is not None:
            for i, node_idx in enumerate(self.actual_station_indices):
                self.acceleration[i][:, it] = acceleration[node_idx, :]
    
    @property
    def locations(self):
        """向后兼容：返回所有测站实际位置"""
        return self.all_actual_locations
    
    @property
    def target_locations(self):
        """返回所有测站的目标位置"""
        return self.all_locations
    
    @property
    def actual_locations(self):
        """返回所有测站的实际位置"""
        return self.all_actual_locations
    
    @property
    def location_errors(self):
        """返回所有测站的定位误差"""
        return self.all_errors
    
    @property
    def indices(self):
        """向后兼容：返回所有测站索引"""
        return self.all_indices
    
    @property
    def seismograms(self):
        """
        向后兼容：返回位移地震图
        格式：(num_receivers, ndim, nt)
        """
        num_receivers = len(self.displacement)
        seismograms = np.zeros((num_receivers, self.ndim, self.nt))
        for i in range(num_receivers):
            seismograms[i, :, :] = self.displacement[i]
        return seismograms