"""
使用Taichi的GUI系统实现高性能的实时波场可视化
"""

import taichi as ti
import numpy as np
from src.taichi_config import TaichiConfig
from .base import VisualizationStrategy


@ti.data_oriented
class TaichiGUIVisualizer(VisualizationStrategy):
    """
    Taichi GUI实时可视化器
    """
    
    def __init__(self, mesh, config, height=800, colormap='jet'):
        """
        初始化Taichi GUI可视化器
        
        参数:
            mesh: Mesh, 网格对象
            config: dict, 配置参数
            height: int, 窗口高度（像素，默认800）
            colormap: str, 配色方案 ('jet' 或 'viridis'，默认'jet')
        """
        self.mesh = mesh
        self.config = config
        self.gui = None
        self.running = False
        self.colormap = colormap.lower()  # 配色方案
        
        # 几何边界, 用于归一化坐标
        self.xmin = config['xmin']
        self.xmax = config['xmax']
        self.zmin = config['zmin']
        self.zmax = config['zmax']
        
        # 计算模型的实际宽高比
        model_width = self.xmax - self.xmin
        model_height = self.zmax - self.zmin
        aspect_ratio = model_width / model_height
        
        # 固定高度，根据宽高比计算宽度
        self.height = height
        self.width = int(height * aspect_ratio)
        
        # 确保宽度在合理范围内
        min_width = 400
        max_width = 2560
        if self.width < min_width:
            self.width = min_width
            print(f"  Note: Window width adjusted to minimum {min_width}px")
        elif self.width > max_width:
            self.width = max_width
            print(f"  Note: Window width adjusted to maximum {max_width}px")
        
        print(f"  Window size: {self.width}x{self.height} (aspect ratio: {aspect_ratio:.2f})")
        print(f"  Model size: {model_width:.1f} x {model_height:.1f} m")
        print(f"  Colormap: {self.colormap}")
        
        # Taichi fields
        self.coords_field = None  # 节点坐标 [num_nodes, 2]
        self.pixels = None  # 像素缓冲区 [width, height, 3]
        self.mag_field = None  # 幅值缓冲区 [num_nodes]
    
    @ti.kernel
    def _compute_max_amplitude_kernel(self, wavefield: ti.template()) -> float:
        """
        在GPU上计算波场最大幅值
        
        参数:
            wavefield: Taichi Vector field [num_nodes, 2]
        
        返回:
            最大幅值标量
        """
        max_val = 0.0
        for i in wavefield:
            # 计算幅值
            amp = ti.sqrt(wavefield[i][0]**2 + wavefield[i][1]**2)
            ti.atomic_max(max_val, amp)
        return max_val
    
    @ti.kernel
    def _compute_field_magnitude(self, 
                                  field: ti.template(),
                                  mag: ti.template()):
        """
        计算波场幅值 |field| = sqrt(field_x^2 + field_z^2)
        
        参数:
            field: ti.Vector.field(2, dtype=float), 波场数据
            mag: ti.field(dtype=float), 输出幅值
        """
        for i in mag:
            mag[i] = ti.sqrt(field[i][0]**2 + field[i][1]**2)
    
    @ti.func
    def _jet_colormap(self, val: float) -> ti.math.vec3:
        """
        Jet配色方案
        输入: val ∈ [0, 1]
        输出: RGB ∈ [0, 1]^3
        """
        r = float(0.0)
        g = float(0.0)
        b = float(0.0)
        
        if val < float(0.0):
            val = float(0.0)
        if val > float(1.0):
            val = float(1.0)
        
        if val < float(0.125):
            b = float(0.5) + float(0.5) * (val / float(0.125))
        elif val < float(0.375):
            b = float(1.0)
            g = (val - float(0.125)) / float(0.25)
        elif val < float(0.625):
            g = float(1.0)
            b = float(1.0) - (val - float(0.375)) / float(0.25)
        elif val < float(0.875):
            g = float(1.0)
            r = (val - float(0.625)) / float(0.25)
        else:
            r = float(1.0)
            g = float(1.0) - (val - float(0.875)) / float(0.125)
        
        # 使用数学函数简化计算
        #val = ti.max(0.0, ti.min(1.0, val))
        #r = ti.min(ti.max(1.5 - abs(6 * val - 3), 0), 1)
        #g = ti.min(ti.max(1.5 - abs(6 * val - 2), 0), 1)
        #b = ti.min(ti.max(1.5 - abs(6 * val - 1), 0), 1)
        
        return ti.Vector([r, g, b])
    
    @ti.func
    def _viridis_colormap(self, val: float) -> ti.math.vec3:
        """
        Viridis配色方案
        输入: val ∈ [0, 1]
        输出: RGB ∈ [0, 1]^3
        """
        r = float(0.0)
        g = float(0.0)
        b = float(0.0)
        
        # 限制输入范围
        if val < float(0.0):
            val = float(0.0)
        if val > float(1.0):
            val = float(1.0)
        
        # Viridis配色的分段线性近似
        if val < float(0.25):
            # 深紫 → 紫蓝 (0.0 <= val < 0.25)
            t = val / float(0.25)
            r = float(0.267) + t * float(0.015)
            g = float(0.005) + t * float(0.210)
            b = float(0.329) + t * float(0.230)
        elif val < float(0.5):
            # 紫蓝 → 青色 (0.25 <= val < 0.5)
            t = (val - float(0.25)) / float(0.25)
            r = float(0.282) - t * float(0.154)
            g = float(0.215) + t * float(0.351)
            b = float(0.559) - t * float(0.008)
        elif val < float(0.75):
            # 青色 → 绿色 (0.5 <= val < 0.75)
            t = (val - float(0.5)) / float(0.25)
            r = float(0.128) + t * float(0.265)
            g = float(0.566) + t * float(0.210)
            b = float(0.551) - t * float(0.295)
        else:
            # 绿色 → 黄色 (0.75 <= val <= 1.0)
            t = (val - float(0.75)) / float(0.25)
            r = float(0.393) + t * float(0.600)
            g = float(0.776) + t * float(0.130)
            b = float(0.256) - t * float(0.112)
        
        return ti.Vector([r, g, b])
    
    @ti.kernel
    def _render_field_to_pixels_jet(self,
                                     coords: ti.template(),
                                     field_mag: ti.template(),
                                     x_min: float, x_max: float,
                                     z_min: float, z_max: float,
                                     vmin: float, vmax: float):
        """
        将波场数据渲染到像素缓冲区
        
        参数:
            coords: ti.Vector.field(2, dtype=float), 节点坐标 [num_nodes, 2]
            field_mag: ti.field(dtype=float), 波场幅值 [num_nodes]
            x_min, x_max, z_min, z_max: 边界范围
            vmin, vmax: 颜色映射范围
        """
        # 清空像素
        for i, j in self.pixels:
            #self.pixels[i, j] = ti.Vector([float(0.341), float(0.341), float(0.431)]) # 深灰色背景
            self.pixels[i, j] = ti.Vector([float(0.0), float(0.0), float(0.0)]) # 黑色背景
        
        # 渲染每个节点
        for node_idx in field_mag:
            x = coords[node_idx][0]
            z = coords[node_idx][1]
            
            # 归一化坐标到[0, 1]
            x_norm = (x - x_min) / (x_max - x_min)
            z_norm = (z - z_min) / (z_max - z_min)
            
            # 转换到像素坐标
            px = ti.cast(x_norm * self.width, ti.i32)
            pz = ti.cast(z_norm * self.height, ti.i32)
            
            # 边界检查
            if px >= 0 and px < self.width and pz >= 0 and pz < self.height:
                # 归一化场值到[0, 1]
                val = field_mag[node_idx]
                val_norm = float((val - vmin) / (vmax - vmin)) if vmax > vmin else float(0.0)
                
                # 应用jet配色
                color = self._jet_colormap(val_norm)
                self.pixels[px, pz] = color
    
    @ti.kernel
    def _render_field_to_pixels_viridis(self,
                                         coords: ti.template(),
                                         field_mag: ti.template(),
                                         x_min: float, x_max: float,
                                         z_min: float, z_max: float,
                                         vmin: float, vmax: float):
        """
        将波场数据渲染到像素缓冲区
        
        参数:
            coords: ti.Vector.field(2, dtype=float), 节点坐标 [num_nodes, 2]
            field_mag: ti.field(dtype=float), 波场幅值 [num_nodes]
            x_min, x_max, z_min, z_max: 边界范围
            vmin, vmax: 颜色映射范围
        """
        # 清空像素, 白色背景
        for i, j in self.pixels:
            self.pixels[i, j] = ti.Vector([float(0.0), float(0.0), float(0.0)])
        
        # 渲染每个节点
        for node_idx in field_mag:
            x = coords[node_idx][0]
            z = coords[node_idx][1]
            
            # 归一化坐标到[0, 1]
            x_norm = (x - x_min) / (x_max - x_min)
            z_norm = (z - z_min) / (z_max - z_min)
            
            # 转换到像素坐标
            px = ti.cast(x_norm * self.width, ti.i32)
            pz = ti.cast(z_norm * self.height, ti.i32)
            
            # 边界检查
            if px >= 0 and px < self.width and pz >= 0 and pz < self.height:
                # 归一化场值到[0, 1]
                val = field_mag[node_idx]
                val_norm = float((val - vmin) / (vmax - vmin)) if vmax > vmin else float(0.0)
                
                # 应用viridis配色
                color = self._viridis_colormap(val_norm)
                self.pixels[px, pz] = color
    
    
    def initialize(self):
        """初始化Taichi GUI窗口和缓冲区"""
        try:
            # 创建GUI窗口
            self.gui = ti.GUI(
                'PySEM2D: Real-time Wavefield Visualization',
                res=(self.width, self.height)
            )
            
            # 创建坐标field, 将网格节点坐标存储在GPU上
            num_nodes = self.mesh.num_nodes
            self.coords_field = ti.Vector.field(2, dtype=float, shape=num_nodes)
            self.coords_field.from_numpy(self.mesh.global_nodes.astype(TaichiConfig.get_np_float_type()))
            
            # 创建像素缓冲区
            self.pixels = ti.Vector.field(3, dtype=float, shape=(self.width, self.height))
            
            # 创建幅值缓冲区
            self.mag_field = ti.field(dtype=float, shape=num_nodes)
            
            self.running = True
            print("  Taichi GUI initialized successfully")
            
        except Exception as e:
            print(f"  WARNING: Failed to initialize Taichi GUI: {e}")
            print("  Real-time visualization will be disabled")
            self.running = False
    
    def render(self, wavefield_field, time, timestep, field_type):
        """
        渲染波场到Taichi GUI
        
        参数:
            wavefield_field: Taichi Vector field [num_nodes, 2]
            time: float, 当前物理时间
            timestep: int, 当前时间步
            field_type: int, 波场类型 (1=位移, 2=速度, 3=加速度)
        
        返回:
            float: 最大幅值
        """
        if not self.running or not self.gui.running:
            self.running = False
            return None
        
        try:
            # 在GPU上计算最大幅值
            max_amplitude = self._compute_max_amplitude_kernel(wavefield_field)
            
            # 在GPU上计算幅值场
            self._compute_field_magnitude(wavefield_field, self.mag_field)
            
            # 确定颜色映射范围
            vmin = 0.0
            vmax = max_amplitude if max_amplitude > 1e-20 else 1e-20
            
            # 在GPU上渲染到像素缓冲区
            if self.colormap == 'viridis':
                self._render_field_to_pixels_viridis(
                    self.coords_field, self.mag_field,
                    self.xmin, self.xmax, self.zmin, self.zmax,
                    vmin, vmax
                )
            else:  # 默认使用jet
                self._render_field_to_pixels_jet(
                    self.coords_field, self.mag_field,
                    self.xmin, self.xmax, self.zmin, self.zmax,
                    vmin, vmax
                )
            
            # 零拷贝显示
            self.gui.set_image(self.pixels)
            
            # 添加文本信息
            field_names = {1: 'Displacement', 2: 'Velocity', 3: 'Acceleration'}
            units = {1: 'm', 2: 'm/s', 3: 'm/s²'}
            field_name = field_names.get(field_type, 'Unknown')
            unit = units.get(field_type, '')
            
            # 顶部标题
            self.gui.text(
                content=f'{field_name} Magnitude',
                pos=(0.02, 0.96),
                font_size=20,
                color=0xFF0000
            )
            
            # 时间和幅值信息
            time_text = f'Time: {time:.4f} s  |  Step: {timestep}'
            max_text = f'Max: {max_amplitude:.3e} {unit}'
            self.gui.text(
                content=f'{time_text}  |  {max_text}',
                pos=(0.02, 0.92),
                font_size=16,
                color=0xFF0000
            )
            
            # 底部提示
            self.gui.text(
                content='Press ESC to exit',
                pos=(0.02, 0.02),
                font_size=14,
                color=0xFF0000
            )
            
            # 显示画面
            self.gui.show()
            
            # 返回最大幅值
            return max_amplitude
            
        except Exception as e:
            print(f"  WARNING: Rendering error: {e}")
            self.running = False
            return None
    
    def finalize(self):
        """关闭GUI窗口"""
        if self.gui:
            try:
                self.gui.close()
                print("  Taichi GUI closed")
            except:
                pass
        self.running = False
    
    def is_running(self):
        """检查GUI是否仍在运行"""
        return self.running and self.gui and self.gui.running
