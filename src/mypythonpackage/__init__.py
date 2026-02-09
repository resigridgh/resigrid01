from .model.regression import LinearRegression
from .model.regression import run_hydropower
from .regression import CauchyRegression
from .matrix import rowswap, rowscale, rowreplacement, rref
from .two_layer_binary_classification import binary_classification
from .weight_animation import WeightMatrixAnime, animate_weight_heatmap
from .largewt_animation import LargeWeightMatrixAnime, animate_large_heatmap




__all__ = ["LinearRegression", "run_hydropower"]
__all__ = ['CauchyRegression']
__all__ = ['rowswap', 'rowscale', 'rowreplacement', 'rref']
__all__ = ["binary_classification"]

__version__ = "0.2.0"
