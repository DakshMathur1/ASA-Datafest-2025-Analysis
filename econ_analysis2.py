import numpy as np
import matplotlib.pyplot as plt
# Define a range for L (labor)
L = np.linspace(1, 100, 400)

# Isoquant for q=140: K = 196 / L  (since 10√(L·K)=140 implies L*K=196)
K_isoquant_140 = 196 / L

plt.figure(figsize=(8,6))
plt.plot(L, K_isoquant_140, label="Isoquant (q=140): K = 196/L")
plt.axhline(y=5, color='orange', linestyle='--', label="Fixed Capital: K = 5")
plt.plot(39.2, 5, 'ro', label="Required Point (≈39.2, 5)")
plt.xlabel("Labor (L)")
plt.ylabel("Capital (K)")
plt.title("Graph (b): q = 140 with Fixed Capital (K = 5)")
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.ylim(0, 50)
plt.show()
