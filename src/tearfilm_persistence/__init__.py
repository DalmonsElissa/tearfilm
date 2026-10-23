"""Two-timescale tear-film measurement from robotic anterior-segment streams.

The package is organised around the measurement chain rather than around the
usual train/eval split: an acquisition enters through :mod:`streams`, becomes a
deformation field through :mod:`deformation`, acquires a structural reading
through :mod:`structure`, and leaves as a calibrated descriptor through
:mod:`descriptors`. Everything else supports that chain.
"""

from tearfilm_persistence.streams.acquisition import AcquisitionClip

__all__ = ["AcquisitionClip"]
__version__ = "1.0.0"
