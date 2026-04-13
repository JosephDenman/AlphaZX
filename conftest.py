# Root conftest.py — ensures pytest discovers the project correctly
# with --import-mode=importlib.
#
# Shared fixtures can be added here as needed.

# Restore NumPy type aliases removed in NumPy 2.0 (np.float_, np.int_, etc.)
# before any test imports networkx/pyzx.  See alphazx/_numpy_compat.py.
import alphazx._numpy_compat  # noqa: F401
