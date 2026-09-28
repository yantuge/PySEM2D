"""
数据输入输出管理模块
"""

import numpy as np
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class DataIO:
    """
    数据输入输出管理类
    """
    
    # 测站命名规则：S或A + 4位数字
    STATION_NAME_PATTERN = re.compile(r'^[SA]\d{4}$')
    COORDINATE_PRECISION = 10  # 坐标保留10位小数
    
    # 异步I/O线程池
    _io_executor: Optional[ThreadPoolExecutor] = None
    _pending_writes = []  # 待完成的异步写入任务
    
    @classmethod
    def init_async_io(cls, max_workers=2):
        """
        初始化异步I/O线程池
        
        参数:
            max_workers: int, 最大工作线程数
        """
        if cls._io_executor is None:
            cls._io_executor = ThreadPoolExecutor(max_workers=max_workers)
            cls._pending_writes = []
    
    @classmethod
    def wait_async_io(cls):
        """
        等待所有异步I/O任务完成
        """
        if cls._pending_writes:
            print(f"Waiting for {len(cls._pending_writes)} async I/O tasks to complete...")
            for future in cls._pending_writes:
                try:
                    future.result()  # 等待任务完成
                except Exception as e:
                    print(f"Async I/O error: {e}")
            cls._pending_writes.clear()
    
    @classmethod
    def shutdown_async_io(cls):
        """
        在程序结束时调用关闭异步I/O线程池
        """
        cls.wait_async_io()  # 确保所有任务完成
        if cls._io_executor is not None:
            cls._io_executor.shutdown(wait=True)
            cls._io_executor = None
    
    @staticmethod
    def validate_station_name(name):
        """
        验证测站名称格式是否符合规范
        
        参数:
            name: str, 测站名称
            
        返回:
            bool, True表示符合S0001或A0001格式
        """
        return DataIO.STATION_NAME_PATTERN.match(name) is not None
    
    @staticmethod
    def generate_station_names(num_stations, start_index=1, prefix='S'):
        """
        自动生成测站名称
        
        参数:
            num_stations: int, 测站数量
            start_index: int, 起始编号（默认1）
            prefix: str, 测站名称前缀（默认'S'）
            
        返回:
            list of str, 测站名称列表 ['S0001', 'S0002', ...] 或 ['A0001', 'A0002', ...]
        """
        names = []
        for i in range(num_stations):
            station_id = start_index + i
            if station_id > 9999:
                raise ValueError(f"Station index {station_id} exceeds maximum 9999")
            names.append(f"{prefix}{station_id:04d}")
        return names
    
    @staticmethod
    def read_station_file(file_path):
        """
        从文件读取测站信息
        
        文件格式支持：
        1. 无表头格式：
           S0001       2200.0000000000       680.0000000000
           S0002       2300.0000000000       680.0000000000
        
        2. 带表头格式：
           # Station_Name  X  Z
           S0001       2200.0000000000       680.0000000000
           S0002       2300.0000000000       680.0000000000
        
        参数:
            file_path: str, 测站文件路径
            
        返回:
            names: list of str, 测站名称列表
            coordinates: np.ndarray, 目标坐标数组 (num_stations, 2)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Station file not found: {file_path}")
        
        names = []
        coordinates = []
        
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释行
                    continue
                
                # 解析行: 名称 + 空格 + X坐标 + 空格 + Z坐标
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(
                        f"Invalid format at line {line_num}: expected at least 3 columns, got {len(parts)}"
                    )
                
                name, x_str, z_str = parts[0], parts[1], parts[2]
                
                # 验证测站名称格式
                if not DataIO.validate_station_name(name):
                    raise ValueError(
                        f"Invalid station name '{name}' at line {line_num}: "
                        f"must follow pattern 'S0001' (S + 4 digits)"
                    )
                
                # 解析坐标
                try:
                    x = float(x_str)
                    z = float(z_str)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid coordinate at line {line_num}: {e}"
                    )
                
                names.append(name)
                coordinates.append([x, z])
        
        if len(names) == 0:
            raise ValueError(f"No valid stations found in file: {file_path}")
        
        print(f"Successfully loaded {len(names)} stations from {file_path}")
        return names, np.array(coordinates)
    
    @staticmethod
    def write_station_file(file_path, names, coordinates):
        """
        将测站信息写入文件
        
        参数:
            file_path: str, 输出文件路径
            names: list of str, 测站名称列表
            coordinates: np.ndarray, 坐标数组 (num_stations, 2)
        """
        if len(names) != len(coordinates):
            raise ValueError(
                f"Mismatch: {len(names)} names but {len(coordinates)} coordinates"
            )
        
        # 验证所有测站名称
        for name in names:
            if not DataIO.validate_station_name(name):
                raise ValueError(
                    f"Invalid station name '{name}': must follow pattern 'S0001' or 'A0001'"
                )
        
        # 如果不存在, 创建目录
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', 
                    exist_ok=True)
        
        # 写入文件
        with open(file_path, 'w') as f:
            # 写入表头
            f.write("# Station_Name  X                   Z\n")
            
            # 写入数据
            for i, name in enumerate(names):
                x, z = coordinates[i]
                f.write(f"{name}       "
                       f"{x:.{DataIO.COORDINATE_PRECISION}f}       "
                       f"{z:.{DataIO.COORDINATE_PRECISION}f}\n")
        
        print(f"Station file saved: {file_path}")
    
    @staticmethod
    def write_seismogram_csv(file_path, time_array, data_x, data_z):
        """
        将单个测站的地震图数据写入CSV文件
        
        文件格式：
        Time,X,Z
        0.0000,1.23e-5,-2.34e-6
        0.0001,1.45e-5,-2.56e-6
        
        参数:
            file_path: str, 输出CSV文件路径
            time_array: np.ndarray, 时间数组 (nt,)
            data_x: np.ndarray, X方向数据 (nt,)
            data_z: np.ndarray, Z方向数据 (nt,)
        """
        if not (len(time_array) == len(data_x) == len(data_z)):
            raise ValueError(
                f"Array length mismatch: time={len(time_array)}, "
                f"X={len(data_x)}, Z={len(data_z)}"
            )
        
        # 创建目录
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', 
                    exist_ok=True)
        
        # 写入CSV
        with open(file_path, 'w') as f:
            # 写入表头
            f.write("Time,X,Z\n")
            
            # 写入数据
            for t, x, z in zip(time_array, data_x, data_z):
                f.write(f"{t:.6f},{x:.12e},{z:.12e}\n")
    
    @staticmethod
    def export_all_seismograms(receivers, output_dir, dt, nt):
        """
        导出所有测站的地震图数据为CSV文件
        
        参数:
            receivers: Receivers对象，包含测站信息和地震图数据
            output_dir: str, 输出目录
            dt: float, 时间步长
            nt: int, 时间步数
        """
        # 创建输出目录
        seismogram_dir = os.path.join(output_dir, 'seismograms','data')
        os.makedirs(seismogram_dir, exist_ok=True)
        
        # 生成时间数组
        time_array = np.arange(nt) * dt
        
        # rec_types定义：1=位移, 2=速度, 3=加速度
        type_names = {1: 'disp', 2: 'vel', 3: 'acc'}
        type_data_fields = {1: 'displacement', 2: 'velocity', 3: 'acceleration'}
        
        print("\n" + "="*60)
        print("EXPORTING SEISMOGRAM DATA TO CSV")
        print("="*60)
        
        total_files = 0
        
        # 遍历所有实际测站
        for i, station_name in enumerate(receivers.actual_station_names):
            # 遍历该测站需要输出的数据类型
            for rec_type in receivers.actual_station_rec_types[i]:
                if rec_type not in type_names:
                    print(f"Warning: Unknown rec_type {rec_type} for station {station_name}, skipping")
                    continue
                
                type_name = type_names[rec_type]
                data_field = type_data_fields[rec_type]
                
                # 获取数据
                data_dict = getattr(receivers, data_field, None)
                if data_dict is None:
                    print(f"Warning: {data_field} data not available for station {station_name}, skipping")
                    continue
                
                # data_dict[i] shape: (2, nt)
                data = data_dict[i]
                data_x = data[0, :]
                data_z = data[1, :]
                
                # 生成文件名
                csv_filename = f"{station_name}_{type_name}.csv"
                csv_filepath = os.path.join(seismogram_dir, csv_filename)
                
                # 写入CSV
                DataIO.write_seismogram_csv(csv_filepath, time_array, data_x, data_z)
                total_files += 1
        
        print(f"Total {total_files} CSV files exported to: {seismogram_dir}\n")
        #print("="*60 + "\n")
    
    # 配置文件读取
    @staticmethod
    def read_config(file_path, format='yaml'):
        """
        读取配置文件
        
        参数:
            file_path: str, 配置文件路径
            format: str, 文件格式 ('yaml', 'json', 'ini')
            
        返回:
            dict, 配置参数字典
        """
        raise NotImplementedError(
            "Configuration file reading is not yet implemented. "
            "Currently, please use dictionary-based configuration in main script."
        )
    
    # 波场快照数据输出
    @staticmethod
    def write_snapshot_data(file_path, mesh, displacement, velocity=None, acceleration=None,
                           time=0.0, timestep=0, snapshot_fields=[1], file_format='vtu',
                           async_write=False):
        """
        输出单个时间步的波场快照数据到VTU文件
        
        参数:
            file_path: str, 输出VTU文件路径 (不含扩展名)
            mesh: Mesh对象, 包含GLL节点坐标信息
            displacement: np.ndarray, 位移场 (num_nodes, 2)
            velocity: np.ndarray or None, 速度场 (num_nodes, 2)
            acceleration: np.ndarray or None, 加速度场 (num_nodes, 2)
            time: float, 当前物理时间
            timestep: int, 时间步编号
            snapshot_fields: list of int, 输出字段类型 [1=位移, 2=速度, 3=加速度]
            file_format: str, 输出格式 ('vtu' 或 'h5')
            async_write: bool, 是否使用异步I/O（默认False，同步写入）
            
        返回:
            str, 实际写入的文件路径
        """
        try:
            import meshio
        except ImportError:
            raise ImportError(
                "meshio library is required for snapshot export. "
                "Please install it: pip install meshio"
            )
        
        if file_format == 'h5':
            raise NotImplementedError(
                "HDF5 format is not yet implemented. Please use 'vtu' format."
            )
        
        # 获取GLL节点坐标
        points = mesh.global_nodes  # (num_nodes, 2)
        num_nodes = points.shape[0]
        
        # 转换为3D坐标 (meshio要求，Y坐标设为0)
        points_3d = np.column_stack([points[:, 0], np.zeros(num_nodes), points[:, 1]])
        
        # 定义cells：每个GLL点作为独立的vertex
        cells = [("vertex", [[i] for i in range(num_nodes)])]
        
        # 准备point_data字典
        point_data = {}
        
        # 字段类型映射：1=位移, 2=速度, 3=加速度
        field_map = {
            1: ('displacement', displacement, 'disp'),
            2: ('velocity', velocity, 'vel'),
            3: ('acceleration', acceleration, 'acc')
        }
        
        # 根据snapshot_fields添加数据
        for field_type in snapshot_fields:
            if field_type not in field_map:
                Warning(f"Unknown snapshot field type: {field_type}, skipping...")
                continue  # 静默跳过未知类型
            
            field_name, field_data, prefix = field_map[field_type]
            
            if field_data is None:
                Warning(f"{field_name} data not provided, skipping export for this field.")
                continue  # 静默跳过未提供的数据
            
            # 验证数据形状
            if field_data.shape != (num_nodes, 2):
                raise ValueError(
                    f"{field_name} shape mismatch: expected ({num_nodes}, 2), "
                    f"got {field_data.shape}"
                )
            
            # 计算幅值
            magnitude = np.sqrt(field_data[:, 0]**2 + field_data[:, 1]**2)
            
            # 添加到point_data
            # 移除了vector字段以减小文件大小和提升读取速度
            point_data[f"{prefix}_x"] = field_data[:, 0]
            point_data[f"{prefix}_z"] = field_data[:, 1]
            point_data[f"{prefix}_mag"] = magnitude
        
        # 如果没有任何有效字段，返回None
        if len(point_data) == 0:
            return None
        
        # 生成输出文件路径
        output_file = f"{file_path}.vtu"
        
        # 根据async_write选择同步或异步写入
        if async_write:
            # 异步写入, 提交到线程池
            if DataIO._io_executor is None:
                DataIO.init_async_io()
            
            # 在主线程中完成创建meshio Mesh对象
            mesh_data = meshio.Mesh(
                points=points_3d.copy(),  # 复制数据避免竞态条件
                cells=cells,
                point_data={k: v.copy() for k, v in point_data.items()}
            )
            
            # 提交异步写入任务
            future = DataIO._io_executor.submit(
                DataIO._write_vtu_file, output_file, mesh_data
            )
            DataIO._pending_writes.append(future)
        else:
            # 同步写入, 立即执行
            mesh_data = meshio.Mesh(
                points=points_3d,
                cells=cells,
                point_data=point_data
            )
            meshio.write(output_file, mesh_data, binary=True)
        
        return output_file
    
    @staticmethod
    def _write_vtu_file(output_file, mesh_data):
        """
        用于异步执行写入VTU文件的内部方法
        
        参数:
            output_file: str, 输出文件路径
            mesh_data: meshio.Mesh, meshio网格对象
        """
        import meshio
        meshio.write(output_file, mesh_data, binary=True)
    
    @staticmethod
    def write_pvd_collection(pvd_path, vtu_files, times):
        """
        生成ParaView时间序列集合PVD格式文件
        
        PVD文件是XML格式，引用多个VTU文件及其对应的时间值，
        使得ParaView可以作为时间序列动画播放。
        
        参数:
            pvd_path: str, PVD文件路径 (含.pvd扩展名)
            vtu_files: list of str, VTU文件路径列表 (相对或绝对路径)
            times: list of float, 每个VTU文件对应的时间值
        """
        if len(vtu_files) != len(times):
            raise ValueError(
                f"Number of VTU files ({len(vtu_files)}) must match "
                f"number of time values ({len(times)})"
            )
        
        if len(vtu_files) == 0:
            print("Warning: No VTU files to include in PVD collection")
            return
        
        # 创建目录
        os.makedirs(os.path.dirname(pvd_path) if os.path.dirname(pvd_path) else '.', 
                    exist_ok=True)
        
        # 获取PVD文件所在目录, 用于计算相对路径
        pvd_dir = os.path.dirname(os.path.abspath(pvd_path))
        
        # 写入PVD文件, XML格式
        with open(pvd_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            
            for vtu_file, time_val in zip(vtu_files, times):
                # 计算相对路径, 相对于PVD文件位置
                vtu_abs = os.path.abspath(vtu_file)
                try:
                    vtu_rel = os.path.relpath(vtu_abs, pvd_dir)
                except ValueError:
                    # 如果在不同驱动器上，使用绝对路径
                    vtu_rel = vtu_abs
                
                # 统一使用正斜杠
                vtu_rel = vtu_rel.replace('\\', '/')
                
                f.write(f'    <DataSet timestep="{time_val:.6e}" group="" part="0" file="{vtu_rel}"/>\n')
            
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')
        
        print(f"PVD collection file saved: {pvd_path}")
        print(f"  Total snapshots: {len(vtu_files)}")
        print(f"  Time range: [{times[0]:.6f}, {times[-1]:.6f}] s")
        print(f"You can open the .pvd file in ParaView to visualize the time series of wavefield.")
