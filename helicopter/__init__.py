import os
import sys

_current_dir = os.path.dirname(__file__)
_build_dir = os.path.abspath(os.path.join(_current_dir, "..", "cmake-build-default"))

if os.path.exists(_build_dir) and _build_dir not in sys.path:
    sys.path.append(_build_dir)

import helicopter_cpp
sys.modules[f"{__name__}.helicopter_cpp"] = helicopter_cpp
