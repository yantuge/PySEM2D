"""
核心抽象层模块

"""

from .physics_model import PhysicsModelBase, WaveFieldType
from .compute_backend import ComputeBackendBase
from .material_properties import MaterialPropertiesBase

__all__ = [
    'PhysicsModelBase',
    'WaveFieldType',
    'ComputeBackendBase',
    'MaterialPropertiesBase',
]
