import numpy as np
from .gll import lagrange_any


class Source:
    """
    管理多震源信息, 所有震源必须是同一类型, 不能混合使用
    """
    
    def __init__(self, params):
        """
        初始化震源对象
        
        参数:
            params: dict, 包含震源配置的参数字典
        """
        self.params = params
        
        # 检查是否使用多震源格式
        if 'sources' not in params:
            raise ValueError(
                "Configuration must include 'sources' list. "
                "See documentation for format examples."
            )
        
        self.sources = params['sources']
        self.num_sources = len(self.sources)
        
        if self.num_sources == 0:
            raise ValueError("At least one source must be defined.")
        
        # 确定震源类型并验证一致性
        self.source_type = self._validate_and_get_source_type()
        
        # 提取所有震源位置
        self.locations = np.array([src['position'] for src in self.sources])
        
        # 为每个震源生成时间函数
        self.stf_list = []
        self.time_delays = []
        
        # 根据震源类型初始化特定参数
        if self.source_type == 'force':
            self.angles = []  # 力源角度列表
        else:  # moment_tensor
            self.moment_tensors = []  # 矩张量列表
            self.source_arrays = None  # 延迟初始化，需要mesh和assembler信息
            self.source_elements = None  # 震源所在单元索引
            self.source_local_indices = None  # 震源在单元内的局部索引
        
        # 处理每个震源
        for i, src in enumerate(self.sources):
            # 生成时间函数
            stf = self._generate_stf_for_source(src, i)
            self.stf_list.append(stf)
            
            # 时间延迟, 转换为时间步数
            time_delay_sec = float(src.get('time_delay', 0.0))
            time_delay_steps = int(time_delay_sec / params['dt'])
            self.time_delays.append(time_delay_steps)
            
            # 根据类型处理特定参数
            if self.source_type == 'force':
                angle_deg = float(src.get('force_angle', 0.0))
                self.angles.append(angle_deg * np.pi / 180.0)
            else:  # moment_tensor
                Mxx = float(src.get('Mxx', 0.0))
                Mzz = float(src.get('Mzz', 0.0))
                Mxz = float(src.get('Mxz', 0.0))
                self.moment_tensors.append({'Mxx': Mxx, 'Mzz': Mzz, 'Mxz': Mxz})
        
        # 打印震源配置信息
        self._print_source_info()
    
    def _validate_and_get_source_type(self):
        """
        验证所有震源类型一致并返回震源类型
        
        返回:
            str: 震源类型 ('force' 或 'moment_tensor')
        """
        source_types = set()
        
        for i, src in enumerate(self.sources):
            # 获取震源类型
            src_type = src.get('source_type', 'force')
            
            if src_type not in ['force', 'moment_tensor']:
                raise ValueError(
                    f"Source {i+1}: Invalid source_type '{src_type}'. "
                    f"Supported types: 'force', 'moment_tensor'."
                )
            
            source_types.add(src_type)
        
        # 检查是否混合了不同类型
        if len(source_types) > 1:
            raise ValueError(
                "All sources must be of the same type. "
                "Cannot mix 'force' and 'moment_tensor' sources in one simulation. "
                f"Found types: {source_types}"
            )
        
        return source_types.pop()
    
    def _generate_stf_for_source(self, src, src_id):
        """
        为单个震源生成时间函数
        
        参数:
            src: dict, 震源参数字典
            src_id: int, 震源编号
        
        返回:
            stf: numpy.ndarray, 震源时间函数序列
        """
        p = self.params
        t = np.linspace(0, (p['nt'] - 1) * p['dt'], p['nt'])
        
        time_func_type = src['time_func_type']
        amplitude = float(src['amplitude'])
        f0 = float(src['f0'])
        
        if time_func_type == 'ricker':
            # Ricker wavelet: second derivative of a Gaussian
            t0 = float(src.get('t0', 1.325 / f0))
            tau = np.pi * f0 * (t - t0)
            stf = amplitude * (1.0 - 2.0 * tau**2) * np.exp(-tau**2)
            return stf
            
        elif time_func_type == 'gaussian':
            # Gaussian wavelet
            t0 = float(src.get('t0', 1.18 / f0))
            stf = amplitude * np.exp(-(np.pi * f0 * (t - t0))**2)
            return stf
        
        elif time_func_type == 'dgaussian':
            # Derivative of Gaussian wavelet
            t0 = float(src.get('t0', 1.18 / f0))
            tau = np.pi * f0 * (t - t0)
            stf = -2.0 * amplitude * tau * np.exp(-tau**2)
            return stf
            
        elif time_func_type == 'gabor':
            # Gabor wavelet: modulated Gaussian
            gamma = float(src.get('gamma', 4.0))
            theta = float(src.get('theta', np.pi / 2))
            t0 = float(src.get('t0', 0.48 * gamma / f0))
            
            w0 = 2 * np.pi * f0
            envelope = np.exp(-((w0 / gamma) * (t - t0))**2)
            carrier = np.cos(w0 * (t - t0) + theta)
            stf = amplitude * envelope * carrier
            return stf
            
        else:
            raise ValueError(
                f"Source {src_id+1}: time function type '{time_func_type}' is not implemented. "
                f"Supported types: 'ricker', 'gaussian', 'dgaussian', 'gabor'."
            )
    
    def _print_source_info(self):
        """打印震源配置信息"""
        print(f"\nSource Configuration:")
        print(f"  Source type: {self.source_type}")
        print(f"  Number of sources: {self.num_sources}")
        
        for i, src in enumerate(self.sources):
            print(f"  Source {i+1}:")
            print(f"    Position: {src['position']}")
            
            if self.source_type == 'force':
                print(f"    Force angle: {src.get('force_angle', 0.0)}°")
            else:  # moment_tensor
                mt = self.moment_tensors[i]
                print(f"    Moment tensor: Mxx={mt['Mxx']}, Mzz={mt['Mzz']}, Mxz={mt['Mxz']}")
            
            print(f"    Amplitude: {src['amplitude']}")
            print(f"    Frequency f0: {src['f0']} Hz")
            print(f"    Time function: {src['time_func_type']}")
            
            if self.time_delays[i] > 0:
                print(f"    Time delay: {src.get('time_delay', 0.0)} s ({self.time_delays[i]} steps)")
    
    def get_stf_at_time(self, src_idx, it):
        """
        获取指定震源在指定时间步的STF值
        
        参数:
            src_idx: int, 震源索引
            it: int, 当前时间步
        
        返回:
            float, STF值
        """
        actual_time = it - self.time_delays[src_idx]
        if actual_time < 0 or actual_time >= len(self.stf_list[src_idx]):
            return 0.0
        return self.stf_list[src_idx][actual_time]
    
    def initialize_moment_tensor_sources(self, mesh, assembler, gll):
        """
        初始化矩震源数组
        
        参数:
            mesh: Mesh对象，包含网格信息
            assembler: SEMAssembler对象，包含雅可比信息
            gll: GLL对象，包含GLL点和权重
        """
        if self.source_type != 'moment_tensor':
            return
        
        print("\nInitializing moment tensor source arrays...")
        
        NGLL = gll.ngll
        NDIM = 2
        num_sources = self.num_sources
        
        # 初始化数组
        self.source_arrays = np.zeros((num_sources, NDIM, NGLL, NGLL))
        self.source_elements = np.zeros(num_sources, dtype=np.int32)
        self.source_local_indices = np.zeros((num_sources, 2), dtype=np.int32)
        
        xigll = gll.points
        
        for src_idx in range(num_sources):
            src_pos = self.locations[src_idx]
            
            # 查找包含震源的单元及其局部索引
            ispec, i_src, j_src = self._find_source_element(
                src_pos, mesh, assembler, gll
            )
            
            self.source_elements[src_idx] = ispec
            self.source_local_indices[src_idx] = [i_src, j_src]
            
            # 震源在参考坐标系中的位置
            xi_source = xigll[i_src]
            gamma_source = xigll[j_src]
            
            # 计算震源数组
            source_array = self._compute_moment_tensor_source_array(
                src_idx, ispec, xi_source, gamma_source,
                assembler, gll
            )
            
            self.source_arrays[src_idx] = source_array
            
            print(f"  Source {src_idx+1}: element={ispec}, local_idx=({i_src},{j_src})")
        
        print("  Moment tensor source arrays initialized.")
    
    def _find_source_element(self, src_pos, mesh, assembler, gll):
        """
        查找包含震源位置的单元及震源最近的GLL点索引
        
        参数:
            src_pos: numpy.ndarray, 震源位置 [x, z]
            mesh: Mesh对象
            assembler: SEMAssembler对象
            gll: GLL对象
        
        返回:
            ispec: int, 单元索引
            i_source: int, xi方向的局部GLL点索引
            j_source: int, gamma方向的局部GLL点索引
        """
        NGLL = gll.ngll
        
        # 查找最近的全局节点
        distances = np.linalg.norm(mesh.global_nodes - src_pos, axis=1)
        nearest_node = np.argmin(distances)
        
        # 查找包含该节点的单元
        for ispec in range(mesh.num_elements):
            for j in range(NGLL):
                for i in range(NGLL):
                    if mesh.connectivity[ispec, j, i] == nearest_node:
                        return ispec, i, j
        
        raise ValueError(f"Cannot find element containing source at {src_pos}")
    
    def _compute_moment_tensor_source_array(self, src_idx, ispec, xi_source, gamma_source,
                                            assembler, gll):
        """
        计算单个矩震源在其所在单元各GLL点的贡献数组
        
        参数:
            src_idx: int, 震源索引
            ispec: int, 震源所在单元索引
            xi_source: float, 震源在参考坐标系xi方向的位置
            gamma_source: float, 震源在参考坐标系gamma方向的位置
            assembler: SEMAssembler对象，包含雅可比信息
            gll: GLL对象
        
        返回:
            source_array: numpy.ndarray, 形状 (NDIM, NGLL, NGLL)
        """
        NGLL = gll.ngll
        NDIM = 2
        xigll = gll.points
        
        # 获取矩张量分量
        mt = self.moment_tensors[src_idx]
        Mxx = mt['Mxx']
        Mzz = mt['Mzz']
        Mxz = mt['Mxz']
        
        # 计算震源位置的拉格朗日插值函数及其导数
        hxis, hpxis = lagrange_any(xi_source, NGLL, xigll)
        hgammas, hpgammas = lagrange_any(gamma_source, NGLL, xigll)
        
        # 计算参考坐标到物理坐标的导数映射
        dxis_dx = 0.0
        dxis_dz = 0.0
        dgammas_dx = 0.0
        dgammas_dz = 0.0
        
        for m in range(NGLL):
            for k in range(NGLL):
                xixd = assembler.xix[ispec, k, m]
                xizd = assembler.xiz[ispec, k, m]
                gammaxd = assembler.gammax[ispec, k, m]
                gammazd = assembler.gammaz[ispec, k, m]
                
                hlagrange = hxis[k] * hgammas[m]
                
                dxis_dx += hlagrange * xixd
                dxis_dz += hlagrange * xizd
                dgammas_dx += hlagrange * gammaxd
                dgammas_dz += hlagrange * gammazd
        
        # 计算震源数组
        # 矩张量分量 Mxx, Mzz, Mxz 应设置为归一化值, 实际震源强度由 amplitude 参数控制
        source_array = np.zeros((NDIM, NGLL, NGLL))
        
        for m in range(NGLL):
            for k in range(NGLL):
                dsrc_dx = (hpxis[k] * dxis_dx) * hgammas[m] + hxis[k] * (hpgammas[m] * dgammas_dx)
                dsrc_dz = (hpxis[k] * dxis_dz) * hgammas[m] + hxis[k] * (hpgammas[m] * dgammas_dz)
                
                source_array[0, k, m] += Mxx * dsrc_dx + Mxz * dsrc_dz
                source_array[1, k, m] += Mxz * dsrc_dx + Mzz * dsrc_dz
        
        # The physical basis gradients carry inverse-length units.
        return source_array
    
    def get_source_array(self, src_idx):
        """
        获取指定矩震源的空间分布数组
        
        参数:
            src_idx: int, 震源索引
        
        返回:
            numpy.ndarray: 形状 (NDIM, NGLL, NGLL) 的震源数组
        """
        if self.source_type != 'moment_tensor':
            raise ValueError("get_source_array only available for moment_tensor sources")
        
        if self.source_arrays is None:
            raise RuntimeError(
                "Moment tensor source arrays not initialized. "
                "Call initialize_moment_tensor_sources() first."
            )
        
        return self.source_arrays[src_idx]
    
    def get_source_element(self, src_idx):
        """
        获取指定矩震源所在的单元索引
        
        参数:
            src_idx: int, 震源索引
        
        返回:
            int: 单元索引
        """
        if self.source_type != 'moment_tensor':
            raise ValueError("get_source_element only available for moment_tensor sources")
        
        return self.source_elements[src_idx]