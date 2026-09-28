"""
Taichi配置管理模块
管理Taichi后端的初始化和硬件资源检测
"""
import taichi as ti
import warnings
import os
#import psutil


class TaichiConfig:
    """
    Taichi配置管理器
    """
    _initialized = False
    _actual_config = {}
    
    @classmethod
    def get_np_float_type(cls):
        """获取当前配置对应的NumPy浮点类型"""
        import numpy as np
        if not cls._initialized:
            return np.float64
        return np.float32 if cls._actual_config.get('precision') == 'f32' else np.float64
    
    @classmethod
    def initialize(cls, config):
        """
        根据配置初始化Taichi后端
        
        参数:
            config: dict, 从YAML解析的配置字典
                - solver.taichi.arch: 'cuda', 'vulkan', 'metal', 'cpu'
                - solver.taichi.default_fp: 'f32' or 'f64'
                - solver.taichi.gpu.device_memory_gb: float
                - solver.taichi.gpu.device_id: int
                - solver.taichi.cpu.max_num_threads: int
                - solver.taichi.cpu.default_vector_width: int
        
        返回:
            dict: 实际使用的配置
        """
        if cls._initialized:
            warnings.warn("Taichi has already been initialized. Ignoring repeated initialization.")
            return cls._actual_config
        
        print("\n" + "="*70)
        print("TAICHI BACKEND CONFIGURATION")
        print("="*70)
        
        # 提取配置
        solver_config = config.get('solver', {})
        taichi_config = solver_config.get('taichi', {})
        
        # 根据 taichi.arch 自动推断 solver_type
        # 如果 arch 是 'cpu', 则为 CPU 模式; 否则为 GPU 模式
        arch = taichi_config.get('arch', 'cpu')
        solver_type = 'cpu' if arch == 'cpu' else 'gpu'
        
        # 默认配置
        DEFAULT_FP = 'f64'
        DEFAULT_GPU_MEMORY = 8.0  # GB
        DEFAULT_CPU_THREADS = 4
        
        # 读取用户配置
        requested_arch = arch  # 已经从 taichi_config 读取
        requested_fp = taichi_config.get('default_fp', DEFAULT_FP)
        
        gpu_config = taichi_config.get('gpu', {})
        requested_gpu_memory = gpu_config.get('device_memory_gb', DEFAULT_GPU_MEMORY)
        requested_device_id = gpu_config.get('device_id', 0)
        
        cpu_config = taichi_config.get('cpu', {})
        requested_cpu_threads = cpu_config.get('max_num_threads', DEFAULT_CPU_THREADS)
        requested_cpu_vector_width = cpu_config.get('default_vector_width', None)
        
        # 显示用户请求的配置
        print(f"Requested Configuration:")
        print(f"  Solver Type:     {solver_type}")
        print(f"  Architecture:    {requested_arch}")
        print(f"  Precision:       {requested_fp}")
        
        # 确定最终使用的架构和参数
        final_arch = requested_arch
        final_params = {}
        
        # 转换精度字符串为Taichi类型
        if requested_fp == 'f64':
            final_params['default_fp'] = ti.f64
            precision_str = 'f64 (double precision)'
        elif requested_fp == 'f32':
            final_params['default_fp'] = ti.f32
            precision_str = 'f32 (single precision)'
        else:
            warnings.warn(f"Unknown precision '{requested_fp}', using f64 (double precision)")
            final_params['default_fp'] = ti.f64
            precision_str = 'f64 (double precision)'
        
        # GPU后端配置
        if requested_arch in ['cuda', 'vulkan', 'metal', 'opengl', 'dx11']:
            if requested_arch == 'opengl':
                warnings.warn(
                    "OpenGL backend is deprecated and may have limited features. "
                    "Consider using 'vulkan' for cross-platform GPU support instead.",
                    DeprecationWarning
                )
                print(f"  WARNING: OpenGL backend is legacy. Consider using Vulkan instead.")
            elif requested_arch == 'dx11':
                warnings.warn(
                    "DX11 backend is deprecated and may have limited features. "
                    "Consider using 'vulkan' or 'cuda' instead.",
                    DeprecationWarning
                )
                print(f"  WARNING: DX11 backend is legacy. Consider using Vulkan or CUDA instead.")
            
            success, final_arch, gpu_params = cls._configure_gpu_backend(
                requested_arch, requested_gpu_memory, requested_device_id
            )
            
            if success:
                final_params.update(gpu_params)
            else:
                # GPU配置失败，回退到CPU
                print("\n" + "!"*70)
                print("WARNING: GPU backend configuration failed!")
                print("Falling back to CPU backend for computation.")
                print("!"*70 + "\n")
                final_arch = 'cpu'
                final_params.update(cls._configure_cpu_backend(
                    requested_cpu_threads, requested_cpu_vector_width
                ))
        
        # CPU后端配置
        elif requested_arch == 'cpu':
            final_params.update(cls._configure_cpu_backend(
                requested_cpu_threads, requested_cpu_vector_width
            ))
        
        else:
            warnings.warn(f"Unknown architecture '{requested_arch}', falling back to CPU")
            final_arch = 'cpu'
            final_params.update(cls._configure_cpu_backend(
                requested_cpu_threads, requested_cpu_vector_width
            ))
        
        # 将架构字符串转换为Taichi类型
        arch_map = {
            'cuda': ti.cuda,
            'vulkan': ti.vulkan,
            'metal': ti.metal,
            'cpu': ti.cpu,
            'opengl': ti.opengl,
            'dx11': ti.dx11
        }
        final_params['arch'] = arch_map[final_arch]
        
        # 初始化Taichi
        print("\n" + "-"*70)
        print("Initializing Taichi with final configuration:")
        print(f"  Architecture:    {final_arch}")
        print(f"  Precision:       {precision_str}")
        
        if final_arch in ['cuda', 'vulkan', 'metal', 'opengl', 'dx11']:
            print(f"  GPU Memory:      {final_params.get('device_memory_GB', 'default')} GB")
            if 'device_id' in final_params:
                print(f"  GPU Device ID:   {final_params['device_id']}")
        elif final_arch == 'cpu':
            print(f"  CPU Threads:     {final_params.get('cpu_max_num_threads', 'default')}")
            if 'cpu_default_vector_width' in final_params:
                print(f"  Vector Width:    {final_params['cpu_default_vector_width']}")
        
        try:
            ti.init(**final_params)
            cls._initialized = True
            print("\nTaichi initialization successful!")
            print("="*70 + "\n")
            
            # 保存实际配置
            cls._actual_config = {
                'arch': final_arch,
                'precision': requested_fp,
                'params': final_params
            }
            
            return cls._actual_config
            
        except Exception as e:
            print(f"\nERROR: Taichi initialization failed: {e}")
            print("Attempting fallback to CPU backend...")
            
            # 最后的回退尝试
            try:
                fallback_params = {
                    'arch': ti.cpu,
                    'default_fp': final_params['default_fp']
                }
                ti.init(**fallback_params)
                cls._initialized = True
                print("Fallback to CPU successful!")
                print("="*70 + "\n")
                
                cls._actual_config = {
                    'arch': 'cpu',
                    'precision': requested_fp,
                    'params': fallback_params
                }
                return cls._actual_config
                
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize Taichi even with CPU fallback: {e2}")
    
    @classmethod
    def _configure_gpu_backend(cls, arch, requested_memory_gb, device_id):
        """
        配置GPU后端，包括可用性检查和资源限制
        
        返回:
            tuple: (success: bool, final_arch: str, params: dict)
        """
        print(f"\nConfiguring GPU backend ({arch})...")
        
        # 检查架构是否可用
        if not cls._check_gpu_available(arch):
            return False, arch, {}
        
        params = {}
        
        # 检查并设置显存
        max_memory = cls._get_max_gpu_memory()
        if max_memory is not None:
            if requested_memory_gb > max_memory:
                warnings.warn(
                    f"Requested GPU memory ({requested_memory_gb} GB) exceeds available memory "
                    f"({max_memory:.2f} GB). Setting to {max_memory:.2f} GB."
                )
                params['device_memory_GB'] = max_memory
                print(f"  GPU Memory:      {max_memory:.2f} GB (auto-adjusted from {requested_memory_gb} GB)")
            else:
                params['device_memory_GB'] = requested_memory_gb
                print(f"  GPU Memory:      {requested_memory_gb} GB")
        else:
            # 无法检测显存，使用用户设置
            params['device_memory_GB'] = requested_memory_gb
            print(f"  GPU Memory:      {requested_memory_gb} GB (user-specified, detection unavailable)")
        
        # 设置设备ID
        if device_id != 0:
            params['device_id'] = device_id
            print(f"  GPU Device ID:   {device_id}")
        
        return True, arch, params
    
    @classmethod
    def _configure_cpu_backend(cls, requested_threads, requested_vector_width):
        """
        配置CPU后端，包括线程数和向量宽度检查
        
        Returns:
            dict: CPU配置参数
        """
        print(f"\nConfiguring CPU backend...")
        params = {}
        
        # 检查并设置CPU线程数
        max_threads = os.cpu_count()
        if max_threads is not None:
            if requested_threads > max_threads:
                warnings.warn(
                    f"Requested CPU threads ({requested_threads}) exceeds available cores "
                    f"({max_threads}). Setting to {max_threads}."
                )
                params['cpu_max_num_threads'] = max_threads
                print(f"  CPU Threads:     {max_threads} (auto-adjusted from {requested_threads})")
            else:
                params['cpu_max_num_threads'] = requested_threads
                print(f"  CPU Threads:     {requested_threads}")
        else:
            # 无法检测CPU核心数，使用用户设置
            params['cpu_max_num_threads'] = requested_threads
            print(f"  CPU Threads:     {requested_threads} (user-specified, detection unavailable)")
        
        # 设置向量宽度
        if requested_vector_width is not None:
            params['cpu_default_vector_width'] = requested_vector_width
            print(f"  Vector Width:    {requested_vector_width}")
        
        return params
    
    @classmethod
    def _check_gpu_available(cls, arch):
        """
        检查指定的GPU后端是否可用
        
        返回:
            bool: GPU后端是否可用
        """
        try:
            # 尝试检测GPU
            if arch == 'cuda':
                import subprocess
                try:
                    # 尝试运行nvidia-smi检测NVIDIA GPU
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        gpu_names = result.stdout.strip().split('\n')
                        print(f"  Detected CUDA GPU(s): {', '.join(gpu_names)}")
                        return True
                    else:
                        print(f"  CUDA GPU not detected (nvidia-smi failed)")
                        return False
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    print(f"  Cannot verify CUDA availability (nvidia-smi not found)")
                    # 仍然允许尝试，因为可能在某些环境下nvidia-smi不可用但GPU可用
                    return True
            
            elif arch == 'vulkan':
                # Vulkan通常更通用，假设可用
                print(f"  Vulkan backend requested (cross-platform GPU)")
                return True
            
            elif arch == 'metal':
                # Metal仅适用于macOS
                import platform
                if platform.system() == 'Darwin':
                    print(f"  Metal backend detected (macOS)")
                    return True
                else:
                    print(f"  Metal backend only available on macOS")
                    return False
            
            else:
                # 其他GPU后端
                return True
                
        except Exception as e:
            print(f"  GPU detection error: {e}")
            return False
    
    @classmethod
    def _get_max_gpu_memory(cls):
        """
        获取GPU最大可用显存
        
        返回:
            float or None: 最大显存 (单位GB)，如果无法检测则返回None
        """
        try:
            import subprocess
            # 尝试使用nvidia-smi获取显存信息
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # 获取第一个GPU的显存（MB）
                memory_mb = float(result.stdout.strip().split('\n')[0])
                memory_gb = memory_mb / 1024.0
                # 留20%的安全冗余
                safe_memory = memory_gb * 0.8
                return safe_memory
        except Exception:
            pass
        
        return None
    
    @classmethod
    def is_initialized(cls):
        """检查Taichi是否已初始化"""
        return cls._initialized
    
    @classmethod
    def get_config(cls):
        """获取实际使用的配置"""
        return cls._actual_config.copy()
    
    @classmethod
    def reset(cls):
        """重置初始化状态"""
        cls._initialized = False
        cls._actual_config = {}


def validate_taichi_config(config):
    """
    验证Taichi配置的合法性
    
    参数:
        config: dict, 配置字典
    
    返回:
        tuple: (is_valid: bool, error_messages: list)
    """
    errors = []
    
    solver_config = config.get('solver', {})
    taichi_config = solver_config.get('taichi', {})
    
    # 检查arch
    arch = taichi_config.get('arch', None)
    valid_archs = ['cuda', 'vulkan', 'metal', 'cpu', 'opengl', 'dx11']
    if arch and arch not in valid_archs:
        errors.append(
            f"Invalid solver.taichi.arch: '{arch}'. "
            f"Valid options: {', '.join(valid_archs)}"
        )
    
    # 检查精度
    fp = taichi_config.get('default_fp', 'f64')
    if fp not in ['f32', 'f64']:
        errors.append(f"Invalid solver.taichi.default_fp: '{fp}'. Must be 'f32' or 'f64'.")
    
    # 检查GPU配置
    gpu_config = taichi_config.get('gpu', {})
    if 'device_memory_gb' in gpu_config:
        mem = gpu_config['device_memory_gb']
        if not isinstance(mem, (int, float)) or mem <= 0:
            errors.append(f"Invalid solver.taichi.gpu.device_memory_gb: {mem}. Must be positive number.")
    
    if 'device_id' in gpu_config:
        dev_id = gpu_config['device_id']
        if not isinstance(dev_id, int) or dev_id < 0:
            errors.append(f"Invalid solver.taichi.gpu.device_id: {dev_id}. Must be non-negative integer.")
    
    # 检查CPU配置
    cpu_config = taichi_config.get('cpu', {})
    if 'max_num_threads' in cpu_config:
        threads = cpu_config['max_num_threads']
        if not isinstance(threads, int) or threads <= 0:
            errors.append(f"Invalid solver.taichi.cpu.max_num_threads: {threads}. Must be positive integer.")
    
    if 'default_vector_width' in cpu_config:
        width = cpu_config['default_vector_width']
        if not isinstance(width, int) or width <= 0:
            errors.append(f"Invalid solver.taichi.cpu.default_vector_width: {width}. Must be positive integer.")
    
    return len(errors) == 0, errors
