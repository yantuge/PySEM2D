"""
弹性波物理模型

该模块实现了弹性波动方程的变体：
- isotropic: 各向同性弹性波
- anisotropic: 横向各向同性(VTI)弹性波
"""

from .isotropic import IsotropicElastic
from .anisotropic import AnisotropicElastic

__all__ = [
    'IsotropicElastic',
    'AnisotropicElastic',
]
