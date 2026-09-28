"""
波场可视化抽象基类
"""

from abc import ABC, abstractmethod


class VisualizationStrategy(ABC):
    """
    可视化抽象基类
    """
    
    @abstractmethod
    def initialize(self):
        """
        初始化可视化环境
        """
        pass
    
    @abstractmethod
    def render(self, wavefield, time, timestep, field_type):
        """
        渲染波场
        
        参数:
            wavefield: 波场数据（Taichi field或NumPy数组）
            time: float, 当前物理时间
            timestep: int, 当前时间步
            field_type: int, 波场类型 (1=位移, 2=速度, 3=加速度)
        """
        pass
    
    @abstractmethod
    def finalize(self):
        """
        清理资源，关闭窗口
        """
        pass
    
    @abstractmethod
    def is_running(self):
        """
        检查可视化是否仍在运行
        
        返回:
            bool, True表示窗口未关闭
        """
        pass
