"""
从YAML配置文件读取参数并运行谱元法仿真
用法:
    python main.py                          # 使用默认配置文件 config/config.yaml
    python main.py -c config/my_config.yaml # 使用指定的配置文件
    python main.py --config custom.yaml     # 使用指定的配置文件
"""

import yaml
import os
import argparse
from src.solver import SEMSolver
from src.mesh import Mesh
from src.material import Material
from src.boundary_conditions import BoundaryConditions
from src.source import Source
from src.receivers import Receivers
from src.visualizer import Visualizer
from src.gll import GLL
from src.assembly import SEMAssembler
from src.utils import check_stability_before, check_wavefiled_accuracy, abaqus_inp_to_gmsh_msh, print_program_banner
from src.taichi_config import TaichiConfig, validate_taichi_config
from datetime import datetime

def load_yaml_config(yaml_path='config/config.yaml'):
    """
    从YAML文件加载配置
    
    参数:
        yaml_path: str, YAML配置文件路径
        
    返回:
        dict, 解析后的配置字典
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
    
    print(f"Loading configuration from: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    # 显示仿真标题
    sim_config = yaml_config.get('simulation', {})
    sim_title = sim_config.get('title', None)
    if sim_title:
        #print("\n" + "="*70)
        print(f"\nSimulation Title: {sim_title}")
        #print("="*70)
    
    return yaml_config


def map_yaml_to_internal_config(yaml_config):
    """
    将YAML配置格式映射到程序内部使用的config字典格式
    
    参数:
        yaml_config: dict, 从YAML文件加载的配置
        
    返回:
        dict, 转换后的内部配置格式
    """
    config = {}
    
    # 1. 基础仿真参数
    sim = yaml_config.get('simulation', {})
    config['title'] = sim.get('title', 'PySEM Simulation')  # 仿真标题
    config['NDIM'] = sim.get('ndim', 2)
    config['P'] = sim.get('polynomial_order', 4)
    config['dt'] = sim.get('dt', 1e-4)
    config['nt'] = sim.get('nt', 10001)
    
    # 2. 网格配置
    mesh_cfg = yaml_config.get('mesh', {})
    config['mesh_format'] = mesh_cfg.get('format', 'gmsh')
    config['msh_file'] = mesh_cfg.get('msh_file', None)
    config['inp_file'] = mesh_cfg.get('inp_file', None)
    
    # 3. 边界条件配置
    boundary = yaml_config.get('boundary', {})
    pml = boundary.get('pml', {}) if 'pml' in boundary else boundary
    
    # PML配置
    config['use_pml_or_not'] = True
    config['pml_x_thick'] = boundary.get('x_thickness', pml.get('x_thickness', 300))
    config['pml_z_thick'] = boundary.get('z_thickness', pml.get('z_thickness', 300))
    config['use_pml_x_left'] = boundary.get('enable_PML_left', pml.get('enable_PML_left', True))
    config['use_pml_x_right'] = boundary.get('enable_PML_right', pml.get('enable_PML_right', True))
    config['use_pml_z_bottom'] = boundary.get('enable_PML_bottom', pml.get('enable_PML_bottom', True))
    config['use_pml_z_top'] = boundary.get('enable_PML_top', pml.get('enable_PML_top', False))
    
    # 4. 震源配置
    sources_cfg = yaml_config.get('sources', {})
    if 'groups' in sources_cfg:
        # 新YAML格式：sources.groups
        config['sources'] = sources_cfg['groups']
    elif isinstance(sources_cfg, list):
        # 直接是列表格式
        config['sources'] = sources_cfg
    else:
        raise ValueError("Invalid source configuration format. Expected 'sources.groups' or 'sources' as list.")
    
    # 5. 接收器配置
    receivers_cfg = yaml_config.get('receivers', {})
    if 'groups' in receivers_cfg:
        config['receiver_groups'] = receivers_cfg['groups']
    else:
        # 默认为空列表
        config['receiver_groups'] = []
    
    # 6. 输出配置
    output = yaml_config.get('output', {})
    config['output_dir'] = output.get('directory', 'results')
    
    # 终端输出配置
    terminal = output.get('terminal', {})
    config['progress_print_interval'] = terminal.get('progress_print_interval', 100)
    
    # 快照配置
    snapshot = output.get('snapshot', {})
    config['enable_snapshot'] = snapshot.get('enabled', True)
    config['snapshot_interval'] = snapshot.get('interval', 100)
    config['snapshot_format'] = snapshot.get('format', 'vtu')
    config['snapshot_field'] = snapshot.get('fields', [1, 2, 3])
    config['async_snapshot'] = snapshot.get('async_write', True)
    
    # 地震图配置
    seismogram = output.get('seismogram', {})
    config['save_seismograms'] = seismogram.get('enabled', True)
    
    # 实时可视化配置
    visualization = yaml_config.get('visualization', {})
    config['enable_realtime_viz'] = visualization.get('enabled', False)
    config['use_taichi_gui'] = visualization.get('use_taichi_gui', True)
    config['viz_interval'] = visualization.get('interval', 500)
    config['viz_field_type'] = visualization.get('field_type', 1)
    config['viz_colormap'] = visualization.get('colormap', 'jet')
    
    # 7. 求解器配置
    solver_cfg = yaml_config.get('solver', {})
    
    # 快速计算模式
    config['fast_mode'] = solver_cfg.get('fast_mode', False)
    
    # 根据 taichi.arch 自动推断 solver_type
    # 如果 arch 是 'cpu'，则为 CPU 模式；否则为 GPU 模式
    taichi_cfg = solver_cfg.get('taichi', {})
    arch = taichi_cfg.get('arch', 'cpu')
    config['solver_type'] = 'cpu' if arch == 'cpu' else 'gpu'
    
    # 保留完整的solver配置(包括taichi配置)供后续使用
    config['solver'] = solver_cfg
    
    # 提取Taichi后端配置（新架构使用）
    config['taichi_config'] = solver_cfg.get('taichi', {})
    
    # 8. 材料配置
    material_cfg = yaml_config.get('material', {})
    config['material_config'] = material_cfg  # 保留原始格式供后续处理
    
    # 提取物理模型类型（新架构使用）
    # 从 material.type 读取，映射到内部的 physics_type
    config['physics_type'] = material_cfg.get('type', 'elastic_isotropic')
    
    return config


def main(config_file='config/config.yaml'):
    """
    主函数：从YAML配置文件读取参数并运行SEM仿真
    
    参数:
        config_file: str, YAML配置文件路径，默认: config/config.yaml
    """
    # 打印程序横幅
    print_program_banner()
    
    # 1. 加载并映射配置
    yaml_config = load_yaml_config(config_file)
    config = map_yaml_to_internal_config(yaml_config)
    
    # 2. 验证Taichi配置
    is_valid, errors = validate_taichi_config(config)
    if not is_valid:
        print("\n" + "!"*70)
        print("CONFIGURATION VALIDATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print("!"*70)
        print("\nPlease fix the configuration errors and try again.")
        return
    
    # 3. 初始化Taichi后端，必须在导入gpu_kernels之前
    # 只有当solver_type为'gpu'或'cpu'且需要Taichi加速时才初始化
    solver_type = config.get('solver_type', 'cpu')
    if solver_type in ['gpu', 'cpu']:
        try:
            taichi_actual_config = TaichiConfig.initialize(config)
            print(f"Taichi backend successfully initialized: {taichi_actual_config['arch']}")
        except Exception as e:
            print(f"\nERROR: Failed to initialize Taichi backend: {e}")
            print("Simulation cannot continue without Taichi.")
            return
    
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"  Polynomial order (P):     {config['P']}")
    print(f"  Time step (dt):           {config['dt']} s")
    print(f"  Number of time steps:     {config['nt']}")
    print(f"  Mesh format:              {config['mesh_format']}")
    print(f"  Solver type:              {config['solver_type'].upper()}")
    print(f"  Fast mode:                {'ENABLED' if config['fast_mode'] else 'DISABLED'}")
    print(f"  PML enabled:              {config['use_pml_or_not']}")
    print(f"  Number of sources:        {len(config['sources'])}")
    print(f"  Number of receiver groups: {len(config['receiver_groups'])}")
    print("="*70 + "\n")
    
    # 4. 创建 GLL 求积器
    gll = GLL(order=config['P'])
    
    # 5. 创建并加载网格
    mesh = Mesh(config, gll)
    
    if config['mesh_format'] == 'abaqus':
        # ABAQUS格式：先转换为Gmsh格式
        if not config.get('inp_file'):
            raise ValueError("ABAQUS format requires 'inp_file' in mesh configuration")
        
        inp_file = config['inp_file']
        # 生成msh文件路径
        if not config.get('msh_file'):
            base_name = os.path.splitext(os.path.basename(inp_file))[0]
            config['msh_file'] = base_name + '.msh'
        
        print(f"Converting ABAQUS mesh: {inp_file} -> {config['msh_file']}")
        # 传入输出目录参数，将msh和vtk文件保存到结果目录
        actual_msh_file = abaqus_inp_to_gmsh_msh(
            inp_file, 
            config['msh_file'], 
            order=config['P'],
            output_dir=config['output_dir']
        )
        mesh.read_from_gmsh(actual_msh_file)
        
        # 更新config中的msh_file为实际的完整路径（用于后续的稳定性检查等）
        config['msh_file'] = actual_msh_file
        
    elif config['mesh_format'] == 'gmsh':
        # Gmsh格式：直接读取
        if not config.get('msh_file'):
            raise ValueError("Gmsh format requires 'msh_file' in mesh configuration")
        
        print(f"Loading Gmsh mesh: {config['msh_file']}")
        mesh.read_from_gmsh(config['msh_file'])
        
    else:
        raise ValueError(f"Unsupported mesh format: {config['mesh_format']}")
    
    # 从网格中获取边界信息并更新config
    config['xmin'] = mesh.global_nodes[:, 0].min()
    config['xmax'] = mesh.global_nodes[:, 0].max()
    config['zmin'] = mesh.global_nodes[:, 1].min()
    config['zmax'] = mesh.global_nodes[:, 1].max()
    
    print(f"\nMesh boundaries: x=[{config['xmin']:.2f}, {config['xmax']:.2f}], "
          f"z=[{config['zmin']:.2f}, {config['zmax']:.2f}]")
    
    # 6. 设置材料属性
    material = Material(mesh)
    material_config = config['material_config']
    physics_type = config.get('physics_type', 'elastic_isotropic').lower()
    
    if 'physical_groups' in material_config and mesh.physical_groups:
        # 基于物理组设置材料
        print("\nSetting material properties based on physical groups...")
        physical_group_properties = material_config['physical_groups']
        
        # 根据物理类型选择设置方法
        if physics_type == 'elastic_anisotropic':
            # 检查是否提供了各向异性参数
            first_group = list(physical_group_properties.values())[0]
            if 'c11' in first_group:
                # 各向异性材料配置
                material.set_anisotropic_by_physical_groups(physical_group_properties)
            else:
                raise ValueError(
                    "Anisotropic material requires c11, c13, c33, c44, rho parameters. "
                    "Got: " + str(list(first_group.keys()))
                )
        else:
            # 各向同性材料配置
            material.set_by_physical_groups(physical_group_properties)

    else:
        # 默认均匀介质
        print("\nWarning:No material configuration found, using default homogeneous material")
        material.set_homogeneous(rho=2700, vp=5000, vs=5000/1.732)
    
    # 7. 创建边界条件
    bc = BoundaryConditions(config)
    
    # 8. 创建震源
    source_params = {
        'sources': config['sources'],
        'dt': config['dt'],
        'nt': config['nt']
    }
    source = Source(source_params)
    
    # 9. 创建物理模型
    if physics_type == 'elastic_anisotropic':
        from src.physics.elastic import AnisotropicElastic
        physics_model = AnisotropicElastic(material, bc)
    else:
        from src.physics.elastic import IsotropicElastic
        physics_model = IsotropicElastic(material, bc)
    print(f"\nPhysics model created: {physics_model.get_model_name()}")
    
    # 10. 创建组装器
    assembler = SEMAssembler(gll, mesh, physics_model=physics_model)
    
    # 11. 设置PML边界条件
    bc.setup_pml(assembler, material)
    
    # 12. 创建接收器
    receivers = Receivers(config, mesh, output_dir=config['output_dir'])
    
    # 13. 创建可视化器
    visualizer = Visualizer(output_dir=config['output_dir'])
    
    # 14. 稳定性检查
    check_stability_before(config, material, mesh, msh_file=config.get('msh_file'))
    
    # 15. 波场精度检查
    check_wavefiled_accuracy(config, material, mesh, msh_file=config.get('msh_file'))
    
    # 16. 绘制模型设置
    print("\nGenerating model setup plot...")
    visualizer.plot_model_setup(mesh, material, source, receivers, config, save=True)
    
    # 17. 创建求解器
    solver = SEMSolver(config, mesh, material, bc, source, receivers, assembler, visualizer)
    
    # 18. 组装矩阵
    taichi_arch = config.get('solver', {}).get('taichi', {}).get('arch', 'cpu').lower()
    use_gpu_stiffness = taichi_arch in ['cuda', 'vulkan', 'metal']
    if use_gpu_stiffness:
        print(f"\n  GPU stiffness assembly enabled (backend: {taichi_arch.upper()})")
    else:
        print(f"\n  CPU stiffness assembly (backend: {taichi_arch.upper()})")
    solver.assemble_matrices(use_gpu=use_gpu_stiffness)
    
    # 19. 运行仿真
    solver.run()
    
    # 获取当前时间和日期
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    print("\n" + "*"*70)
    print("SIMULATION COMPLETED SUCCESSFULLY!")
    print(f"All results are saved in the folder: {visualizer.output_dir}")
    print(f"Completion Date: {date_str} {time_str}")
    print("*"*70 + "\n")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='PySEM - Python Spectral Element Method Solver',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Use default config: config/config.yaml
  python main.py -c config/gmsh_config.yaml  # Use specified config file
  python main.py --config my_simulation.yaml # Use specified config file
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config/config.yaml',
        help='Path to YAML configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='PySEM v1.0.0'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print(f"Please specify a valid configuration file using -c or --config option")
        exit(1)
    
    # 运行主程序
    main(config_file=args.config)