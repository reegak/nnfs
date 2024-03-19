import numpy as np
import matplotlib.pyplot as plt

n = 100;
R = 1;
x0 = 0; # Center of the circle in the x direction.
y0 = 0; # Center of the circle in the y direction.
# Now create the set of points.
t = 2 * np.pi * np.random.rand(n);
r = R * np.sqrt(np.random.rand(n));

x = [];y = []
for i in range(n):
    x.append(x0 + r * np.cos(t))
    y.append(y0 + r * np.sin(t))

plt.scatter(x,y)
plt.show()
