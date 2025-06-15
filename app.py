#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: 14/06/2025
Licencia: GPL v3

Descripción:  
"""
#❯ cat app.py
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Obtener embedding usando Ollama
def get_embedding(text, model="deepseek-r1:1.5b"):
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": model, "prompt": text}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Frase y palabras
frase_completa = "El gato duerme tranquilo"
palabras = frase_completa.split()

# Obtener embeddings
embeddings = {word: get_embedding(word) for word in palabras}
embedding_frase = get_embedding(frase_completa)

# Tomar las primeras 3 dimensiones de cada embedding
embeddings_3d = {word: emb[:3] for word, emb in embeddings.items()}
embedding_frase_3d = embedding_frase[:3]

# Crear gráfico 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Dibujar los ejes (cruz que separa los cuadrantes)
ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='Eje X')
ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Eje Y')
ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Eje Z')

# Dibujar vectores para cada palabra y la frase completa
colors = ['cyan', 'magenta', 'yellow', 'orange', 'purple']  # Colores para el, gato, duerme, tranquilo, frase completa
labels = ['"El"', '"gato"', '"duerme"', '"tranquilo"', '"Frase completa"']

for i, (word, emb_3d) in enumerate(embeddings_3d.items()):
    ax.quiver(0, 0, 0, emb_3d[0], emb_3d[1], emb_3d[2], color=colors[i], label=labels[i], linewidth=2)

# Dibujar el vector de la frase completa
ax.quiver(0, 0, 0, embedding_frase_3d[0], embedding_frase_3d[1], embedding_frase_3d[2], color=colors[-1], label=labels[-1], linewidth=2)

# Configurar límites y cuadrícula
max_coord = max(abs(min([min(emb) for emb in embeddings_3d.values()] + [min(embedding_frase_3d)])), 
                abs(max([max(emb) for emb in embeddings_3d.values()] + [max(embedding_frase_3d)])))
ax.set_xlim([-max_coord - 1, max_coord + 1])
ax.set_ylim([-max_coord - 1, max_coord + 1])
ax.set_zlim([-max_coord - 1, max_coord + 1])
ax.grid(True)

# Etiquetas y título
ax.set_xlabel('Dimensión 1')
ax.set_ylabel('Dimensión 2')
ax.set_zlabel('Dimensión 3')
ax.set_title('Vectores 3D de Embeddings de Palabras y Frase (DeepSeek R-1:1.5b)')
ax.legend()

# Mostrar gráfico
plt.show()
