import numpy as np

def diamond(center: tuple[float, float], bDims: tuple[int, int]) -> np.ndarray:
  (cx, cy) = center
  (width, height) = bDims
  X, Y = np.meshgrid(np.arange(width), np.arange(height))
  mask = (np.abs(X - cx) + np.abs(Y - cy)) <= min(width, height) / 4
  return mask

def circle(center: tuple[float, float], bDims: tuple[int, int], r: float) -> np.ndarray:
  (cx, cy) = center
  (width, height) = bDims
  X, Y = np.meshgrid(np.arange(width), np.arange(height))
  distances = np.sqrt((X - cx)**2 + (Y - cy)**2)
  mask = (distances <= r)
  return mask