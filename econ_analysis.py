import numpy as np
import matplotlib.pyplot as plt

# Define a range for L (labor)
L = np.linspace(1, 40, 400)

# Isoquant for q=100: K = 100 / L
K_isoquant_100 = 100 / L

# Isocost line for TC=800: 20L + 80K = 800 => K = -0.25L + 10
K_isocost_800 = -0.25 * L + 10

plt.figure(figsize=(8,6))
plt.plot(L, K_isoquant_100, label="Isoquant (q=100): K = 100/L")
plt.plot(L, K_isocost_800, label="Isocost (TC=800): K = -0.25L + 10")
plt.plot(20, 5, 'ro', label="Optimal Point (20, 5)")
plt.xlabel("Labor (L)")
plt.ylabel("Capital (K)")
plt.title("Graph (a): q = 100 - Isoquant & Isocost")
plt.legend()
plt.grid(True)
plt.xlim(0, 40)
plt.ylim(0, 12)
plt.show()
