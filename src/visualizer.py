import matplotlib.pyplot as plt
import numpy as np
import os

class Visualizer:
    """
    负责结果的可视化和输出，包括地震记录图和模型设置图
    """
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'snapshots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'seismograms'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'stations'), exist_ok=True)

    def plot_seismograms(self, receivers, params, data_types=None):
        """
        支持多接收器组，为每个组生成独立的PNG文件
        
        参数:
            receivers: Receivers对象，包含接收器位置和地震图数据
            params: 参数字典
            data_types: list of int, 要绘制的数据类型列表
        
        输出:
            为每个接收器组保存独立的seismograms_{group_name}.png文件
        """
        # 数据类型配置
        type_config = {
            1: {'name': 'Displacement', 'unit': 'm', 'data_field': 'displacement'},
            2: {'name': 'Velocity', 'unit': 'm/s', 'data_field': 'velocity'},
            3: {'name': 'Acceleration', 'unit': 'm/s²', 'data_field': 'acceleration'}
        }
        
        # 生成时间轴
        t = np.linspace(0, (params['nt'] - 1) * params['dt'], params['nt'])
        
        print("\n" + "="*60)
        print("SEISMOGRAM STATISTICS")
        print("="*60)
        print(f"Total receiver groups: {len(receivers.group_configs)}")
        print(f"Total actual stations: {len(receivers.actual_station_names)}")
        print(f"Time samples: {params['nt']}")
        
        # 遍历每个接收器组
        for group_config in receivers.group_configs:
            group_name = group_config['name']
            group_idx = group_config['group_idx']
            group_rec_types = group_config['rec_type']
            
            print(f"\nProcessing group '{group_name}' (rec_type={group_rec_types})")
            
            # 查找属于该组的实际测站
            group_station_indices = []  # 实际测站在列表中的索引
            for actual_idx, groups_list in enumerate(receivers.actual_station_groups):
                if group_idx in groups_list:
                    group_station_indices.append(actual_idx)
            
            if len(group_station_indices) == 0:
                print(f"  Warning: No stations found for group '{group_name}', skipping")
                continue
            
            # 获取该组实际测站的坐标
            group_locations = receivers.actual_locations[group_station_indices]
            group_names = [receivers.actual_station_names[i] for i in group_station_indices]
            
            # 确定该组使用的主要坐标轴（X或Z）作为偏移参考
            x_range = np.max(group_locations[:, 0]) - np.min(group_locations[:, 0])
            z_range = np.max(group_locations[:, 1]) - np.min(group_locations[:, 1])
            
            if x_range >= z_range:
                offset_coords = group_locations[:, 0]  # 使用X坐标作为偏移
                coord_label = 'Distance (m)'
            else:
                offset_coords = group_locations[:, 1]  # 使用Z坐标作为偏移
                coord_label = 'Depth (m)'
            
            print(f"  Stations in this group: {len(group_station_indices)}")
            print(f"  Using {'X' if x_range >= z_range else 'Z'} coordinate for trace offset")
            
            # 计算子图布局
            num_types = len(group_rec_types)
            fig, axes = plt.subplots(num_types, 2, figsize=(16, 8 * num_types))
            
            # 如果只有一种类型，确保axes是2D数组
            if num_types == 1:
                axes = axes.reshape(1, -1)
            
            # 遍历该组的每种记录类型
            for type_idx, data_type in enumerate(group_rec_types):
                if data_type not in type_config:
                    print(f"  Warning: Unknown data type {data_type}, skipping")
                    continue
                
                config = type_config[data_type]
                data_field = config['data_field']
                type_name = config['name']
                unit = config['unit']
                
                # 获取数据
                data_list = getattr(receivers, data_field, None)
                if data_list is None:
                    print(f"  Warning: {type_name} data not available, skipping")
                    continue
                
                # 提取该组测站的数据
                group_data = []
                for actual_idx in group_station_indices:
                    group_data.append(data_list[actual_idx])
                
                # 转换为数组格式 [num_stations_in_group, 2, nt]
                num_stations = len(group_data)
                seism = np.zeros((num_stations, 2, params['nt']))
                for i in range(num_stations):
                    seism[i, :, :] = group_data[i]
                
                seism_x = seism[:, 0, :]  # X分量 [num_stations, nt]
                seism_z = seism[:, 1, :]  # Z分量 [num_stations, nt]
                
                # 打印统计信息
                print(f"  {type_name}:")
                print(f"    X-component max amplitude: {np.max(np.abs(seism_x)):.6e} {unit}")
                print(f"    Z-component max amplitude: {np.max(np.abs(seism_z)):.6e} {unit}")
                
                # X分量地震图
                max_amp_x = np.max(np.abs(seism_x))
                if max_amp_x > 0 and len(offset_coords) > 1:
                    offset_range = np.max(np.abs(np.diff(np.sort(offset_coords))))
                    ampli_x = offset_range / max_amp_x
                else:
                    ampli_x = 1.0
                
                for i in range(num_stations):
                    trace_x = seism_x[i, :] * ampli_x + offset_coords[i]
                    axes[type_idx, 0].plot(t, trace_x, '-k', linewidth=1.0)
                
                axes[type_idx, 0].set_title(f'{type_name}-X component (Group: {group_name})', 
                                           fontsize=14, fontweight='bold')
                axes[type_idx, 0].set_xlabel('Time (s)', fontsize=12)
                axes[type_idx, 0].set_ylabel(coord_label, fontsize=12)
                axes[type_idx, 0].grid(True, alpha=0.3)
                axes[type_idx, 0].set_facecolor('#f5f5f5')
                
                # Z分量地震图
                max_amp_z = np.max(np.abs(seism_z))
                if max_amp_z > 0 and len(offset_coords) > 1:
                    offset_range = np.max(np.abs(np.diff(np.sort(offset_coords))))
                    ampli_z = offset_range / max_amp_z
                else:
                    ampli_z = 1.0
                
                for i in range(num_stations):
                    trace_z = seism_z[i, :] * ampli_z + offset_coords[i]
                    axes[type_idx, 1].plot(t, trace_z, '-k', linewidth=1.0)
                
                axes[type_idx, 1].set_title(f'{type_name}-Z component (Group: {group_name})', 
                                           fontsize=14, fontweight='bold')
                axes[type_idx, 1].set_xlabel('Time (s)', fontsize=12)
                axes[type_idx, 1].set_ylabel(coord_label, fontsize=12)
                axes[type_idx, 1].grid(True, alpha=0.3)
                axes[type_idx, 1].set_facecolor('#f5f5f5')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存该组的图形
            filepath = os.path.join(self.output_dir, 'seismograms', 
                                   f'seismograms_{group_name}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Seismograms saved to: {filepath}")
            plt.close()
        
        #print("="*60 + "\n")

    def plot_model_setup(self, mesh, material, source, receivers, config, save=True):
        """
        绘制模型设置，显示材料属性、震源和接收器位置，供用户仿真前检查
        """
        # 获取全局节点坐标和材料属性
        coord = mesh.global_nodes  # [num_nodes, 2], (x, z)
        vp = material.vp
        vs = material.vs
        
        # 提取震源和接收器信息
        src_locations = source.locations  # [n_sources, 2]
        # receivers.locations 现在返回实际网格节点坐标
        rec_locations = receivers.locations  # [n_receivers, 2] - actual locations
        
        # 创建图形，包含两个子图：Vp和Vs
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左子图：Vp速度分布
        scatter_vp = axes[0].scatter(coord[:, 0]/1000, coord[:, 1]/1000, 
                                     c=vp, s=15, cmap='viridis')
        # 标记震源
        for src in src_locations:
            axes[0].scatter(src[0]/1000, src[1]/1000, c='blue', s=300, 
                           marker='*', edgecolors='white', linewidths=2, 
                           label='Source' if src is src_locations[0] else '')
        # 标记接收器
        axes[0].scatter(rec_locations[:, 0]/1000, rec_locations[:, 1]/1000, 
                       c='red', s=50, marker='v', alpha=0.6, label='Receivers')
        axes[0].set_xlabel('Distance (km)', fontsize=12)
        axes[0].set_ylabel('Depth (km)', fontsize=12)
        axes[0].set_title('P-wave Velocity (Vp)', fontsize=14, fontweight='bold')
        axes[0].set_xlim(config['xmin']/1000, config['xmax']/1000)
        axes[0].set_ylim(config['zmin']/1000, config['zmax']/1000)
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')
        cbar_vp = plt.colorbar(scatter_vp, ax=axes[0], label='Vp (m/s)')
        
        # 右子图：Vs速度分布
        scatter_vs = axes[1].scatter(coord[:, 0]/1000, coord[:, 1]/1000, 
                                     c=vs, s=15, cmap='plasma')
        # 标记震源
        for src in src_locations:
            axes[1].scatter(src[0]/1000, src[1]/1000, c='blue', s=300, 
                           marker='*', edgecolors='white', linewidths=2,
                           label='Source' if src is src_locations[0] else '')
        # 标记接收器
        axes[1].scatter(rec_locations[:, 0]/1000, rec_locations[:, 1]/1000, 
                       c='red', s=50, marker='v', alpha=0.6, label='Receivers')
        axes[1].set_xlabel('Distance (km)', fontsize=12)
        axes[1].set_ylabel('Depth (km)', fontsize=12)
        axes[1].set_title('S-wave Velocity (Vs)', fontsize=14, fontweight='bold')
        axes[1].set_xlim(config['xmin']/1000, config['xmax']/1000)
        axes[1].set_ylim(config['zmin']/1000, config['zmax']/1000)
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right')
        cbar_vs = plt.colorbar(scatter_vs, ax=axes[1], label='Vs (m/s)')
        
        plt.tight_layout()
        
        # 打印模型信息到控制台
        print("\n" + "="*60)
        print("MODEL SETUP INFORMATION")
        print("="*60)
        print(f"Model dimensions: X=[{config['xmin']/1000:.1f}, {config['xmax']/1000:.1f}] km, " + 
              f"Z=[{config['zmin']/1000:.1f}, {config['zmax']/1000:.1f}] km")
        
        # 检查是否是规则网格
        if 'nelem_x' in config and 'nelem_z' in config:
            print(f"Grid: {config['nelem_x']} x {config['nelem_z']} elements, " +
                  f"Polynomial order: {config['P']}")
        else:
            print(f"Grid: {mesh.num_elements} elements (irregular mesh), " +
                  f"Polynomial order: {config['P']}")
        
        print(f"Total nodes: {mesh.num_nodes}")
        
        # Print material properties based on model type
        if mesh.physical_groups is not None and mesh.element_physical_group is not None:
            # Physical group based material model
            print(f"\nMaterial properties (physical groups):")
            
            # Count elements in each physical group
            group_element_counts = {}
            for phys_id in mesh.physical_groups.keys():
                count = np.sum(mesh.element_physical_group == phys_id)
                group_element_counts[phys_id] = count
            
            # Print information for each physical group
            for phys_id, info in mesh.physical_groups.items():
                group_name = info['name']
                elem_count = group_element_counts[phys_id]
                
                if elem_count > 0:
                    # Find elements belonging to this group
                    elem_indices = np.where(mesh.element_physical_group == phys_id)[0]
                    
                    # Get material properties from first element's nodes
                    first_elem = elem_indices[0]
                    node_indices = mesh.connectivity[first_elem, :, :].flatten()
                    
                    # Get representative values (should be constant within group)
                    rho_val = material.rho[node_indices[0]]
                    vp_val = material.vp[node_indices[0]]
                    vs_val = material.vs[node_indices[0]]
                    
                    print(f"  {group_name} (ID={phys_id}):")
                    print(f"    Elements: {elem_count}")
                    print(f"    rho: {rho_val:.1f} kg/m^3")
                    print(f"    Vp:  {vp_val:.1f} m/s")
                    print(f"    Vs:  {vs_val:.1f} m/s")
        else:
            # Non-physical group model (homogeneous or varying)
            print(f"\nMaterial properties:")
            print(f"  Density (rho): min={np.min(material.rho):.1f}, max={np.max(material.rho):.1f}, mean={np.mean(material.rho):.1f} kg/m^3")
            print(f"  Vp: min={np.min(material.vp):.1f}, max={np.max(material.vp):.1f}, mean={np.mean(material.vp):.1f} m/s")
            print(f"  Vs: min={np.min(material.vs):.1f}, max={np.max(material.vs):.1f}, mean={np.mean(material.vs):.1f} m/s")
        
        
        if save:
            filepath = os.path.join(self.output_dir, 'model_setup.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Model setup diagram saved to: {filepath}")
            plt.close()
        else:
            plt.show()