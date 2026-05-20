import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def dft_1d(vector):
    N = len(vector)
    resultado = np.zeros(N, dtype=complex)

    for k in range(N):
        suma = 0 + 0j
        for n in range(N):
            angulo = -2j * np.pi * k * n / N
            suma += vector[n] * np.exp(angulo)
        resultado[k] = suma

    return resultado

def dft_2d(imagen):
    M, N = imagen.shape

    # Primero DFT por filas
    temporal = np.zeros((M, N), dtype=complex)
    for i in range(M):
        temporal[i, :] = dft_1d(imagen[i, :])

    # Luego DFT por columnas
    resultado = np.zeros((M, N), dtype=complex)
    for j in range(N):
        resultado[:, j] = dft_1d(temporal[:, j])

    return resultado

def centrar_espectro(F):
    M, N = F.shape
    F_centrada = np.zeros_like(F)

    for i in range(M):
        for j in range(N):
            nuevo_i = (i + M // 2) % M
            nuevo_j = (j + N // 2) % N
            F_centrada[nuevo_i, nuevo_j] = F[i, j]

    return F_centrada

# -------------------------------
# Cargar imagen
# -------------------------------
ruta = "pupila5.jpeg"

imagen = Image.open(ruta).convert("L")
imagen = imagen.resize((128, 128))  # Recomendado para que no tarde demasiado
imagen = np.array(imagen, dtype=float)

# -------------------------------
# Aplicar DFT 2D manual
# -------------------------------
F = dft_2d(imagen)

# Centrar espectro
F_centrada = centrar_espectro(F)

# Magnitud en escala logarítmica
espectro = np.log(1 + np.abs(F_centrada))

# -------------------------------
# Mostrar resultados
# -------------------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(imagen, cmap="gray")
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(espectro, cmap="gray")
plt.title("Espectro de Fourier")
plt.axis("off")

plt.show()