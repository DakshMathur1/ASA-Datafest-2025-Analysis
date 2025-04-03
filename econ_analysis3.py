import numpy as np
import matplotlib.pyplot as plt

# Define a range for L (labor)
L = np.linspace(1, 100, 400)

# Isoquant for q=140 remains K = 196 / L.
K_isoquant_140 = 196 / L

# Isocost line for TC=1120: 20L + 80K = 1120 => K = -0.25L + 14
K_isocost_1120 = -0.25 * L + 14

plt.figure(figsize=(8,6))
plt.plot(L, K_isoquant_140, label="Isoquant (q=140): K = 196/L")
plt.plot(L, K_isocost_1120, label="Isocost (TC=1120): K = -0.25L + 14")
plt.plot(28, 7, 'ro', label="Optimal Point (28, 7)")
plt.xlabel("Labor (L)")
plt.ylabel("Capital (K)")
plt.title("Graph (c): q = 140 - Isoquant & Isocost (Long Run)")
plt.legend()
plt.grid(True)
plt.xlim(0, 60)
plt.ylim(0, 20)
plt.show()
