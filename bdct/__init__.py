import warnings

# Numpy emits a warning for log 0, but we might get it a lot during likelihood calculations
warnings.filterwarnings('ignore', r'divide by zero encountered in log')
