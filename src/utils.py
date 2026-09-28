import numpy as np
from scipy.spatial.distance import cdist
import re
import gmsh
import os
from datetime import datetime


def print_program_banner():
    """
    打印程序标识符横幅, 显示程序名称、开发者信息和当前运行时间
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    print("\n" + "*"*70)
    print("""*                                                                    *
*           ██████  ██    ██ ███████ ███████ ███    ███              * 
*           ██   ██  ██  ██  ██      ██      ████  ████              *  
*           ██████    ████   ███████ █████   ██ ████ ██              * 
*           ██         ██         ██ ██      ██  ██  ██              *  
*           ██         ██    ███████ ███████ ██      ██              *  
*                                                                    *
*                                                                    * """)
    print(f"*       PySEM: Python-based Spectral Element Method Solver for       *")
    print(f"*              2D Seismic Wave Simulation                            *")
    print(f"*  Version: 1.0 (GPU/CPU Acceleration Supported By Taichi Framework) *")
    print("*"*70)
    print(f"*  Developer:  Hui Shen (Assistant Professor)                        *")
    print(f"*  Email:      hshen@whut.edu.cn                                     *")
    print(f"*  Institute:  Wuhan University of Technology                        *")
    print(f"*  department: School of Civil Engineering and Architecture          *")
    print(f"*  Date:       {date_str} {time_str}                                   *")
    print("*"*70 + "\n")


def find_nearest_node_indices(points, node_coords):
    """
    为一组点找到最近的网格节点索引
    """
    dist_matrix = cdist(points, node_coords)
    return np.argmin(dist_matrix, axis=1)

def estimate_min_element_edge_length(params, mesh, msh_file=None):
    """
    估算网格最小单元边长, 优先从gmsh文件获取精确值，如果失败则使用网格坐标估算
    
    参数:
        params: dict, 仿真参数, 包含xmin, xmax, zmin, zmax等
        mesh: Mesh, 网格对象
        msh_file: str, 可选，网格文件路径
    
    返回:
        h_min: float, 最小单元边长
        source: str, 数据来源说明（'gmsh'或'estimated'）
    """
    h_min = None
    
    # 尝试从gmsh获取最小边长
    if msh_file and os.path.exists(msh_file):
        try:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)  # 禁止gmsh终端输出
            gmsh.open(msh_file)
            
            # 获取2D单元
            types, tags, _ = gmsh.model.mesh.getElements(dim=2)
            if len(tags) > 0 and len(tags[0]) > 0:
                # 获取所有单元的最小边长
                qualities = gmsh.model.mesh.getElementQualities(tags[0], qualityName="minEdge")
                h_min = np.min(qualities)
                gmsh.finalize()
                return h_min, 'gmsh'
            
            gmsh.finalize()
        except Exception as e:
            print(f"Warning: Failed to get element quality from gmsh: {e}")
            try:
                gmsh.finalize()
            except:
                pass
    
    # 如果无法从gmsh获取，使用网格坐标估算
    domain_x = params['xmax'] - params['xmin']
    domain_z = params['zmax'] - params['zmin']
    domain_area = domain_x * domain_z
    
    # 估算单元特征尺寸
    estimated_elem_size = np.sqrt(domain_area / mesh.num_elements)
    return estimated_elem_size, 'estimated'

def check_stability_before(params, material, mesh, msh_file=None):
    """
    模拟前检查CFL稳定性条件
    
    参数:
        params: dict, 仿真参数
        material: Material, 材料对象
        mesh: Mesh, 网格对象
        msh_file: str, 可选，网格文件路径
    """
    print("\n" + "="*70)
    print("CFL STABILITY CHECK")
    print("="*70)
    
    # 1. 计算最大P波速度
    vp_max = np.max(material.vp)
    print(f"Maximum P-wave velocity (V_p_max): {vp_max:.2f} m/s")
    
    # 2. 获取多项式阶数
    N = params['P']
    print(f"Polynomial order (N): {N}")
    
    # 3. 估算最小GLL节点间距
    h_min, source = estimate_min_element_edge_length(params, mesh, msh_file)
    if source == 'gmsh':
        print(f"Minimum element edge length (h_min): {h_min:.6f} m (from gmsh)")
    else:
        print(f"Estimated minimum element size (h_min): {h_min:.6f} m (rough estimate)")
        print(f"  Note: Could not read from gmsh file. Using domain size estimation.")
    
    # 估算最小GLL节点间距
    delta_x_min = h_min / N
    print(f"Estimated minimum GLL spacing (Δx_min ≈ h_min/N): {delta_x_min:.6f} m")
    
    # 4. 计算CFL稳定时间步
    C_courant = 0.5  # 2D SEM的标准Courant数
    dt_stable = C_courant * (delta_x_min / vp_max)
    print(f"\nCourant number (C): {C_courant}")
    print(f"CFL stability limit (dt_stable): {dt_stable:.8f} s")
    
    # 应用安全系数
    safety_factor = 0.9
    dt_recommended = safety_factor * dt_stable
    print(f"Recommended time step (with {safety_factor} safety factor): {dt_recommended:.8f} s")
    
    # 5. 与用户设定的时间步比较
    user_dt = params['dt']
    print(f"\nUser-specified time step (dt): {user_dt:.8f} s")
    
    if user_dt > dt_stable:
        print("\n[WARNING]: TIME STEP EXCEEDS CFL STABILITY LIMIT!")
        print(f"   Current dt is {user_dt/dt_stable:.2f}x larger than the stability limit.")
        print(f"   The simulation is LIKELY TO BECOME UNSTABLE (blow up).")
        print(f"   Recommended action: Reduce dt to {dt_recommended:.8f} or smaller.")
        print("\nSimulation aborted due to CFL stability violation.")
        raise ValueError(f"Time step {user_dt:.8f} exceeds CFL stability limit {dt_stable:.8f}. Reduce dt to {dt_recommended:.8f} or smaller.")
    elif user_dt > dt_recommended:
        print("\n[CAUTION]: Time step is close to stability limit.")
        print(f"   For better stability, consider reducing dt to {dt_recommended:.8f} or smaller.")
    else:
        margin = (dt_recommended - user_dt) / dt_recommended * 100
        print(f"\nTime step satisfies CFL condition with {margin:.1f}% safety margin.")
    
    print("="*70 + "\n")
    
def check_wavefiled_accuracy(params, material, mesh, msh_file=None):
    """
    根据波传播精度要求, 在模拟前检查波场精度条件

    参数:
        params: dict, 仿真参数
        material: Material, 材料对象
        mesh: Mesh, 网格对象
        msh_file: str, 可选，网格文件路径
    """
    print("\n" + "="*70)
    print("WAVEFIELD ACCURACY CHECK")
    print("="*70)
    
    # 1. 计算最小S波速度
    vs_min = np.min(material.vs)
    if vs_min == 0:
        print("Warning: Minimum S-wave velocity is 0, using minimum non-zero value")
        vs_min = np.min(material.vs[material.vs > 0])
    print(f"Minimum S-wave velocity (V_s_min): {vs_min:.2f} m/s")
    
    # 2. 从震源配置中提取最高频率成分
    sources = params.get('sources', [])
    if not sources:
        print("\n[INFO]: No sources defined, skipping accuracy check")
        print("="*70 + "\n")
        return
    
    # 提取所有震源的f0频率
    f0_list = [float(src['f0']) for src in sources]
    f_max = max(f0_list)
    print(f"Maximum source frequency (f_max): {f_max:.2f} Hz (from {len(sources)} source(s))")
    
    # 3. 计算最短波长
    lambda_min = vs_min / f_max
    print(f"Minimum wavelength (λ_min = V_s_min / f_max): {lambda_min:.2f} m")
    
    # 4. 获取每个波长所需的采样点数（SEM中通常>=5）
    N_ppw = 5  # 谱元法推荐的每波长节点数
    print(f"Points per wavelength (N_ppw): {N_ppw} (SEM recommended minimum)")
    
    # 5. 计算所需的最大网格间距
    delta_x_required = lambda_min / N_ppw
    print(f"\nRequired maximum grid spacing (Δx ≤ λ_min / N_ppw): {delta_x_required:.2f} m")
    
    # 6. 估算实际网格间距（与CFL检查使用相同方法）
    N = params['P']
    h_min, source = estimate_min_element_edge_length(params, mesh, msh_file)
    if source == 'gmsh':
        print(f"Minimum element edge length (h_min): {h_min:.2f} m (from gmsh)")
    else:
        print(f"Estimated minimum element size (h_min): {h_min:.2f} m (rough estimate)")
    
    # 估算最小GLL节点间距
    delta_x_actual = h_min / N
    print(f"Estimated actual grid spacing (Δx = h_min / N): {delta_x_actual:.2f} m")
    
    # 7. 比较实际网格间距与所需值
    if delta_x_actual > delta_x_required:
        ratio = delta_x_actual / delta_x_required
        print("\n" + "!"*70)
        print("[WARNING]: GRID SPACING EXCEEDS ACCURACY REQUIREMENT!")
        print("!"*70)
        print(f"   Actual grid spacing is {ratio:.2f}x LARGER than required.")
        print(f"   The simulation will suffer from NUMERICAL DISPERSION and INACCURACY.")
        print(f"   Wave shapes will be distorted, especially at high frequencies.")
        print(f"\n   Recommended actions:")
        print(f"   1. Refine mesh: reduce element size to ≤ {delta_x_required * N:.2f} m")
        print(f"   2. Increase polynomial order (current P={N})")
        print(f"   3. Reduce source frequency to ≤ {vs_min / (N_ppw * delta_x_actual):.1f} Hz")
        print("!"*70)
        print("\nSimulation aborted due to insufficient grid resolution.")
        raise ValueError(f"Grid spacing {delta_x_actual:.2f}m exceeds accuracy requirement {delta_x_required:.2f}m.")
    else:
        margin = (delta_x_required - delta_x_actual) / delta_x_required * 100
        print(f"\nGrid spacing satisfies accuracy condition with {margin:.1f}% margin.")
        # 计算可以准确模拟的最高频率
        f_accurate_max = vs_min / (N_ppw * delta_x_actual)
        print(f"  Maximum frequency for accurate simulation: {f_accurate_max:.1f} Hz")
        
def abaqus_inp_to_gmsh_msh(inp_file, msh_file, order=4, output_dir=None):
    """
    将 ABAQUS .inp 文件转换为 Gmsh .msh 文件, 并升阶到指定阶数的四节点四边形单元

    参数：
        inp_file (str): 输入的 ABAQUS .inp 文件路径
        msh_file (str): 输出的 Gmsh .msh 文件路径
        order (int): 单元阶数，默认为4
        output_dir (str): 输出目录路径，如果指定则将msh和vtk文件保存到该目录
    """
    # 数据容器
    nodes = {}           # node_id -> (x, y)
    elements = {}        # elem_id -> [node_ids]
    elsets = {}          # elset_name -> [elem_ids]
    nsets = {}           # nset_name -> [node_ids]

    # 支持的ABAQUS四节点四边形单元类型
    supported_quad_types = {
        "CPS4", "CPS4R",       # 平面应力
        "CPE4", "CPE4R",       # 平面应变
    }

    # 状态标志
    node_section = False
    elem_section = False
    current_elset = None
    current_nset = None

    with open(inp_file, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("**"):
            i += 1
            continue

        # --- Node Section ---
        if line.lower().startswith("*node"):
            node_section = True
            elem_section = False
            current_elset = None
            current_nset = None
            i += 1
            continue

        # --- Element Section ---
        elif line.lower().startswith("*element"):
            node_section = False
            current_nset = None

            # 提取元素类型
            m_type = re.search(r"type=([^,]+)", line, re.IGNORECASE)
            elem_type = m_type.group(1).strip().upper() if m_type else "CPS4"
            
            # 检查是否为支持的四边形单元类型
            if elem_type not in supported_quad_types:
                supported_list = ", ".join(sorted(supported_quad_types))
                raise ValueError(
                    f"Unsupported element type: {elem_type}. "
                    f"Only 4-node quadrilateral elements are supported.\n"
                    f"Supported types: {supported_list}"
                )

            # 检查元素是否指定了elset
            m_elset = re.search(r"elset=([^,]+)", line, re.IGNORECASE)
            if m_elset:
                current_elset = m_elset.group(1).strip()
                # 初始化该elset的元素列表
                if current_elset not in elsets:
                    elsets[current_elset] = []
            else:
                # 如果未指定elset，使用"All_Elements"作为默认分组
                current_elset = "All_Elements"
                if current_elset not in elsets:
                    elsets[current_elset] = []

            elem_section = True
            i += 1
            continue

        # --- Elset Section ---
        elif line.lower().startswith("*elset"):
            node_section = False
            elem_section = False
            current_nset = None

            m_elset = re.search(r"elset=([^,]+)", line, re.IGNORECASE)
            current_elset = m_elset.group(1).strip() if m_elset else "Unnamed"
            generate = "generate" in line.lower()

            elsets[current_elset] = []

            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line or next_line.startswith("*"):
                    i -= 1
                    break
                parts = [x.strip() for x in next_line.split(",") if x.strip()]
                if generate and len(parts) == 3:
                    start, end, step = map(int, parts)
                    elsets[current_elset].extend(range(start, end + 1, step))
                else:
                    elsets[current_elset].extend(map(int, parts))
                i += 1
            i += 1
            continue

        # --- Nset Section ---
        elif line.lower().startswith("*nset"):
            node_section = False
            elem_section = False
            current_elset = None

            m_nset = re.search(r"nset=([^,]+)", line, re.IGNORECASE)
            current_nset = m_nset.group(1).strip() if m_nset else "Unnamed"
            generate = "generate" in line.lower()

            nsets[current_nset] = []

            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line or next_line.startswith("*"):
                    i -= 1
                    break
                parts = [x.strip() for x in next_line.split(",") if x.strip()]
                if generate and len(parts) == 3:
                    start, end, step = map(int, parts)
                    nsets[current_nset].extend(range(start, end + 1, step))
                else:
                    nsets[current_nset].extend(map(int, parts))
                i += 1
            i += 1
            continue

        # --- Node Data ---
        if node_section and not line.startswith("*"):
            parts = [x.strip() for x in line.split(",") if x.strip()]
            nid = int(parts[0])
            x, y = map(float, parts[1:3])
            nodes[nid] = (x, y)
            i += 1
            continue

        # --- Element Data ---
        if elem_section and not line.startswith("*"):
            parts = [x.strip() for x in line.split(",") if x.strip()]
            eid = int(float(parts[0]))
            conn = [int(float(x)) for x in parts[1:]]
            elements[eid] = conn
            if current_elset:
                elsets[current_elset].append(eid)
            i += 1
            continue

        i += 1

    print(f"Nodes read: {len(nodes)}")
    print(f"Elements read: {len(elements)}")
    print(f"Elsets: {list(elsets.keys())}")
    print(f"Nsets: {list(nsets.keys())}")

    # --- 写入 Gmsh .msh 文件 ---
    # 如果指定了输出目录，将文件保存到该目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        msh_filename = os.path.basename(msh_file)
        msh_file = os.path.join(output_dir, msh_filename)
    
    # 检查是否存在文件, 如果存在先删除, 防止是遗留文件
    if os.path.exists(msh_file):
        os.remove(msh_file)
    # 创建临时msh文件, 后续升阶
    msh_file_temp = msh_file.replace(".msh", "_temp.msh")
    
    # 开始写入
    with open(msh_file_temp, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # --- 节点 ---
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for nid, (x, y) in nodes.items():
            f.write(f"{nid} {x:.6f} {y:.6f} 0.0\n")
        f.write("$EndNodes\n")

        # --- 单元 ---
        msh_elems = []
        pid_counter = 1
        phys_groups = {}

        # Physical Surfaces (来自Elset)
        for name, eids in elsets.items():
            # 跳过内部默认分组
            if name == "All_Elements":
                continue
            if not eids:
                continue
            phys_groups[name] = pid_counter
            pid_counter += 1
            for eid in eids:
                if eid not in elements:
                    continue
                conn = elements[eid]
                if len(conn) == 4:  # 四边形
                    # 反向节点顺序以确保正的Jacobian
                    # ABAQUS中的节点顺序需要反向才能得到正确的逆时针排列
                    conn_reversed = conn[::-1]
                    conn_right = [conn_reversed[i] for i in (3,0,1,2)]
                    msh_elems.append((eid, 3, phys_groups[name], conn_right))  # gmsh 3=四边形

        # Physical Lines (来自Nset)
        #for name, nids in nsets.items():
        #    if len(nids) < 2:
        #        continue
        #    phys_groups[name] = pid_counter
        #    pid_counter += 1
        #    for j in range(len(nids) - 1):
        #        msh_elems.append((100000 + pid_counter * 1000 + j, 1, phys_groups[name],
        #                          [nids[j], nids[j + 1]]))  # gmsh 1=线段

        # --- 输出单元 ---
        f.write("$Elements\n")
        f.write(f"{len(msh_elems)}\n")
        for eid, etype, pid, conn in msh_elems:
            conn_str = " ".join(str(c) for c in conn)
            f.write(f"{eid} {etype} 2 {pid} {pid} {conn_str}\n")
        f.write("$EndElements\n")

        # --- Physical groups ---
        f.write("$PhysicalNames\n")
        f.write(f"{len(phys_groups)}\n")
        for name, pid in phys_groups.items():
            dim = 2 if pid < pid_counter else 1
            f.write(f"{dim} {pid} \"{name}\"\n")
        f.write("$EndPhysicalNames\n")
    
    # 升阶到指定阶数
    upgrade_to_order(msh_file_temp, msh_file, order, output_dir)
    # 删除临时文件
    os.remove(msh_file_temp)
    print(f"Conversion complete: {msh_file}\n")
    
    return msh_file
    
def upgrade_to_order(msh_in, msh_out=None, order=4, output_dir=None):
    """
    Upgrade mesh to N order quadrilateral elements
    
    参数:
        msh_in (str): Input mesh file path
        msh_out (str): Output mesh file path (optional)
        order (int): Element order (default: 4)
        output_dir (str): Output directory for VTK preview file (optional)
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.open(msh_in)

    # 生成VTK预览文件
    base, ext = os.path.splitext(msh_in)
    base_name = os.path.basename(base)
    # 去掉 _temp 字符串
    base_name = base_name.replace("_temp", "")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vtk_file = os.path.join(output_dir, f"{base_name}.vtk")
    else:
        vtk_file = f"{base}.vtk"
    
    gmsh.write(vtk_file)  # 先输出 VTK 格式以便可视化检查
    
    print(f"[GMSH] Setting element order to {order}...")
    try:
        gmsh.model.mesh.setOrder(order)
    except Exception as e:
        print(f"[GMSH ERROR] {e}")
        print(">>> Try refining the mesh or checking for inconsistent curve connectivity.")
        gmsh.finalize()
        raise

    # 生成高阶节点
    gmsh.model.mesh.createEdges()
    gmsh.model.mesh.createFaces()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    if not msh_out:
        #base, ext = os.path.splitext(msh_in)
        msh_out = f"{base}_order{order}{ext}"

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # 4.1 格式
    gmsh.write(msh_out)
    gmsh.finalize()
    print(f"[GMSH] High-order mesh saved: {msh_out}")
    print(f"[GMSH] Model mesh preview saved: {vtk_file}, you can open it with Paraview.")