from .simple_transform import SimpleTransform
from .simple_transform_3d_smpl import SimpleTransform3DSMPL
from .simple_transform_3d_smpl_cam import SimpleTransform3DSMPLCam
from .simple_transform_cam import SimpleTransformCam

# SMPL-X preset import initializes SMPL-X assets eagerly. Keep it optional so
# SMPL-only scripts still work when SMPL-X model files are absent.
try:
    from .simple_transform_3d_smplx import SimpleTransform3DSMPLX
except Exception:
    SimpleTransform3DSMPLX = None

__all__ = [
    'SimpleTransform', 'SimpleTransform3DSMPL', 'SimpleTransform3DSMPLCam', 'SimpleTransformCam']

if SimpleTransform3DSMPLX is not None:
    __all__.append('SimpleTransform3DSMPLX')
