import numpy as np
import cmath
import matplotlib.pyplot as plt

def f(pi, la):
    pi = pi.reshape(-1, 1)
    la = la.reshape(1, -1)
    Gt = (3-pi)*(1-la)/(0.99*(1-0.99*la))
    return (5*Gt-13)
def eig(y, l):
    y = y.reshape(-1, 1)
    x = 0.99*(1 - l) / (1 - l * 0.99)
    x = x.reshape(1, -1)
    g = 9* x**2 * y**2 \
        -48* x**2 *y\
        +64*x**2\
        +42*x*y\
        -108*x\
        +45 +0j

    lam = abs(0.25 * (0.1* np.sqrt(g)
                  -3*0.1*x*y +8*0.1*x -7*0.1+4
                  ))
    return lam

p = np.array([0, 0.1, 0.2,0.5, 1])
p1 = np.linspace(0, 1, 1000)
l = np.array([0, 0.8, 0.9, 0.95, 1])
l1 = np.linspace(0,1,1000)
results = f(p,l)
results2 = eig(p1,l1)


print(results2<=1)
plt.imshow(results2,extent=(0,1,1,0))
plt.colorbar()
i, j = np.where(np.isclose(results2, 1))
plt.plot(j/1000, i/1000, color="red")
plt.show()