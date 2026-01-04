import pandas as pd

# Construir lista de columnas
columns = ['class']  # La primera columna suele ser la etiqueta
for i in range(33):
    columns += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

# Crear DataFrame vacío con esas columnas
df = pd.DataFrame(columns=columns)

# Guardarlo como coords.csv
df.to_csv('coords.csv', index=False)
print("Archivo 'coords.csv' generado con éxito.")