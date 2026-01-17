import numpy as np

import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = x**4

plt.figure(figsize=(8, 6))
plt.plot(x, y, linewidth=2, color='#2E86AB')
plt.fill_between(x, y, alpha=0.3, color='#2E86AB')
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('x⁴', fontsize=12)
plt.title('f(x) = x⁴', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()