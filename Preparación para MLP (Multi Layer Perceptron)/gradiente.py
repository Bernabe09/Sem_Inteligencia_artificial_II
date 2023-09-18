import numpy as np
from math import e # importamos euler

def F(x1, x2): # La función a evaluar
    return 10 - e ** -((x1**2)+(x2**2))
def G(x, y): # Función gradiente descendiente
    return np.array([(1 - 2*(x**2))*np.exp(-x**2-y**2), - 2*x*y*np.exp(-x**2-y**2)])

xl = [-3, -3] # límite inferior
xu = [3, 3] # límite superior
xi = [-1, 1] # cordenada inicial

h = .1 # Valor que amplifica el resultado del gradiente en la iteración i
for i in range(50):
    xi = xi - h * G(xi[0], xi[1])

print("Mínimo global en:", xi)
