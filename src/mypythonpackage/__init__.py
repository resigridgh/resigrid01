from .model.regression import LinearRegression
from .model.regression import run_hydropower
from .regression import CauchyRegression
from .matrix import rowswap, rowscale, rowreplacement, rref


__all__ = ["LinearRegression", "run_hydropower"]
__all__ = ['CauchyRegression']
__all__ = ['rowswap', 'rowscale', 'rowreplacement', 'rref']

__version__ = "0.2.0"
