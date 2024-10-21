#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Profesor: Boris Panes </em><br>
# 
# </div>
# 
# # Análisis de datos
# *27 de Octubre de 2024*
# 
# **Nombre Estudiante(s)**: Mariangel Arrieta, Giuseppe Lavarello, Ingrid Solís, Rosario Valderrama

# **1. Importación de Librerias**

# Importar librerias
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.linear_model import LinearRegression



# **2. Lectura de archivo**

# Descripción del contexto: Determinar la demanda global de vinos dada la venta histórica

# Especifica la ruta del archivo Excel
file_path = r'..\data\VentaHistoricaVM.xlsx'

# Lee el archivo Excel
df = pd.read_excel(file_path, sheet_name='Hoja4')

# Mostrar las primeras filas del DataFrame
df.head(10)


# Cambiar el nombre de la columna 'ano' a 'año'
df.rename(columns={'ano': 'año'}, inplace=True)

# Mostrar las primeras filas del DataFrame
df.head(10)


# Mostrar las últimas filas del DataFrame
df.tail(10)


# **3. Análisis exploratorio**

# Recopilación de información básica sobre el conjunto de datos
df.info()


# Resumen estadistico de las variables numericas
df.describe()


# Dimensiones del df
df.shape


# Total de elementos únicos por cada columna.
df.nunique()


# Convertir todos los nombres de países a mayúsculas
df['Pais'] = df['Pais'].str.upper()


# **4. Limpieza de datos**

# **4.1 Busqueda de Valores Nulos**

# Filtrar y mostrar las filas con NaN
nan_rows = df[df.isna().any(axis=1)]

#print("Listado de filas con NaN:")
nan_rows.head(10)


# Contar cuántos NaN hay en cada columna
nan_count = df.isna().sum()
print("Valores NaN por columna:")
print(nan_count)


#pip install missingno
import missingno as msgn


# Se visualiza los Nulos
msgn.matrix(df)


# **4.2 Tratamiento de valores Nulos: campos 'mes' , 'CodCliente' y 'CodProducto'** 

# Filtrar las filas donde 'mes' es nulo
mes_nulos = df[df['mes'].isna()]

# Mostrar los años correspondientes a los valores nulos en 'mes'
años_con_mes_nulo = mes_nulos['año'].unique()

print("Años correspondientes a los valores nulos en 'mes':")
print(años_con_mes_nulo)


# Filtrar las filas donde 'CodCliente' es nulo
cod_cliente_nulos = df[df['CodCliente'].isna()]

# Mostrar las primeras filas
print(cod_cliente_nulos.head(10))

# Mostrar cuántos registros tienen 'CodCliente' nulo
print(f"Total de registros con 'CodCliente' nulo: {len(cod_cliente_nulos)}")


# Decisión: Se eliminan registros con valores NAN asociados a 'mes' y 'CodCliente' porque la mayoria pertecen a los registros con periodo 1999 hasta 2004 y representan el 0.5% del total de filas

# Eliminar filas con valores nulos en 'mes' y 'CodCliente'
df = df.dropna(subset=['mes', 'CodCliente', 'codproducto'])

# Mostrar la cantidad de filas después de la limpieza
print(f"Total de filas después de eliminar valores nulos: {len(df)}")


# **4.3 Tratamiento de valores Nulos: campo 'Cosecha'**

# Desición: Lo que buscamos es determinar la demanda global de vinos dada la venta histórica, es por ello que eliminamos la variable cosecha.

# Eliminar la columna 'cosecha'
df.drop(columns=['cosecha'], inplace=True)

# Verificar que la columna ha sido eliminada
df.head()


# **4.4 Tratamiento de valores Nulos: campo 'Pais'**

# Completar los valores nulos de la columna Pais basándonos en el 'mercado' y 'CodCliente' 

# Crear un diccionario basado en mercado, CodCliente y Pais (sin valores nulos)
mercado_cliente_pais = df.dropna(subset=['Pais']).groupby(['mercado', 'CodCliente'])['Pais'].first().to_dict()

# Función para rellenar los valores nulos de 'Pais'
def completar_pais(row):
    if pd.isna(row['Pais']):
        # Buscar en el diccionario el país basado en mercado y CodCliente
        return mercado_cliente_pais.get((row['mercado'], row['CodCliente']), None)
    else:
        return row['Pais']

# Aplicar la función a las filas con 'Pais' nulo
df['Pais'] = df.apply(completar_pais, axis=1)

# Mostrar las primeras filas para verificar los cambios
print(df[['mercado', 'CodCliente', 'Pais']].head(10))

# Verificar si quedan valores nulos
print(f"Valores nulos restantes en la columna 'Pais': {df['Pais'].isna().sum()}")


#evaluamos cual es el país que aun contiene NAN
df[df['Pais'].isna()]


# Filtrar y mostrar las filas donde 'CodCliente' es igual a '2002-8'
codigo_cliente_filas = df[df['CodCliente'] == '2002-8']

# Mostrar las filas correspondientes
print(codigo_cliente_filas)


# Desición: se elimina el unico registro con NAN de Pais porque no existen registros anteriores para el cliente: '2002-8' con los cuales comparar.

# Eliminar las filas donde 'CodCliente' es igual a '2002-8'
df= df[df['CodCliente'] != '2002-8']

# Verificar si las filas fueron eliminadas
print(f"Total de filas después de eliminar '2002-8': {len(df)}")


# Contar cuántos NaN hay en cada columna
nan_count = df.isna().sum()
print("Valores NaN por columna:")
print(nan_count)


# **5. Ingenieria de Columnas**

# **5.1 Eliminar campo MontoCLP,** Solo nos quedaremos con el campo MontoUSD, que corresponde a las ventas en moneda dólar

# Eliminar la columna 'MontoCLP'
df.drop(columns=['MontoCLP'], inplace=True)

# Verificar que la columna ha sido eliminada
df.info()


# Convertir la columna 'mes' y 'año' a tipo entero
df['año'] = df['año'].astype(int)
df['mes'] = df['mes'].astype(int)


# Verificar si hay valores infinitos
print(df[['año', 'mes']].replace([np.inf, -np.inf], np.nan).isna().sum())


# **5.2 Crear la columna Periodo,** combinando 'año' y 'mes', establecerla como indice y elimninar 'año' y 'mes'

# Crear la columna 'periodo' combinando 'año' y 'mes' con un separador
df['periodo'] = df['año'].astype(str) + '-' + df['mes'].astype(int).astype(str).str.zfill(2)

# Mostrar las primeras filas para verificar el resultado
print(df[['año', 'mes', 'periodo']].head())


# Colocar la columna 'periodo' como la primera y eliminar 'año' y 'mes'
df = df[['periodo'] + [col for col in df.columns if col not in ['año', 'mes', 'periodo']]]

# Verificar el nuevo DataFrame
df.head()


# **6. Serie temporales**

# Lo que buscamos es determinar la demanda global de vinos dada la venta histórica, es por ello que eliminamos variables tales como Codcliente, Codproducto, mercado y país

# Eliminar las columnas 'CodCliente', 'codproducto', 'mercado' y 'Pais'
df.drop(columns=['CodCliente', 'codproducto', 'mercado', 'Pais'], inplace=True)

# Verificar que las columnas han sido eliminadas
print(df.head())


# Agrupar por Periodo

# Agrupar por 'periodo' y sumar los valores de 'Cajas9Lts' y 'MontoUSD'
df = df.groupby('periodo').sum().reset_index()

# Mostrar las primeras filas del DataFrame agrupado
print(df.head())


# Agregamos variable Fecha

# Convertir 'periodo' a fecha usando el formato correcto 'YYYY-MM'
df['fecha'] = pd.to_datetime(df['periodo'].astype(str), format='%Y-%m')

# Reordenar las columnas para que 'fecha' esté en primer lugar
columnas = ['fecha'] + [col for col in df.columns if col != 'fecha']
df = df[columnas]

# Verificar las primeras filas
print(df.head())


# Normalizar las variables númericas

from sklearn.preprocessing import MinMaxScaler

# Crear un objeto MinMaxScaler
scaler = MinMaxScaler()

# Normalizar las columnas 'Cajas9Lts' y 'MontoUSD' y crear nuevas columnas para los datos normalizados
df['Cajas9Lts_norm'] = scaler.fit_transform(df[['Cajas9Lts']])
df['MontoUSD_norm'] = scaler.fit_transform(df[['MontoUSD']])

# Verificar las nuevas columnas normalizadas
print(df[['Cajas9Lts', 'Cajas9Lts_norm', 'MontoUSD', 'MontoUSD_norm']].head())


df['periodo'] = pd.to_datetime(df['periodo'], format='%Y-%m')
# Agrupar por año y mes, sumando 'MontoUSD'
df_monthly = df.groupby('periodo', as_index=False)['MontoUSD'].sum()

# Graficar las ventas totales por mes
plt.figure(figsize=(10, 6))
plt.plot(df_monthly['periodo'], df_monthly['MontoUSD'], marker='o')

# Añadir etiquetas y título
plt.title('Total MontoUSD Vendido Cada Mes')
plt.xlabel('Mes')
plt.ylabel('MontoUSD')

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45)

# Ajustar automáticamente las etiquetas del eje x para evitar superposición
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Mostrar cada interval meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Formato: Año-Mes

plt.tight_layout()
plt.show()


# Crear shifts
df_monthly['t-1']=df_monthly['MontoUSD'].shift(1) 
df_monthly['t-12']=df_monthly['MontoUSD'].shift(12)
# Botar Nans
df_monthly.dropna(inplace=True)

df_monthly.head()


# Calcular correlaciones
corr_t1 = df_monthly['MontoUSD'].corr(df_monthly['t-1'])
corr_t12 = df_monthly['MontoUSD'].corr(df_monthly['t-12'])

# Crear gráfico de dispersión para t-1 (mes anterior)
plt.figure(figsize=(10, 6))
plt.scatter(df_monthly['t-1'], df_monthly['MontoUSD'], alpha=0.7)
plt.title('Diagrama de Dispersión: MontoUSD vs t-1 (Mes Anterior)')
plt.xlabel('MontoUSD (t-1)')
plt.ylabel('MontoUSD (Mes Actual)')
plt.grid(True)

# Agregar el valor de la correlación en la esquina superior izquierda
plt.text(0.05, 0.95, f'Correlación: {corr_t1:.2f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, 
         verticalalignment='top')

plt.tight_layout()
plt.show()

# Crear gráfico de dispersión para t-12 (mismo mes el año pasado)
plt.figure(figsize=(10, 6))
plt.scatter(df_monthly['t-12'], df_monthly['MontoUSD'], alpha=0.7)
plt.title('Diagrama de Dispersión: MontoUSD vs t-12 (Mismo Mes Año Pasado)')
plt.xlabel('MontoUSD (t-12)')
plt.ylabel('MontoUSD (Mes Actual)')
plt.grid(True)

# Agregar el valor de la correlación en la esquina superior izquierda
plt.text(0.05, 0.95, f'Correlación: {corr_t12:.2f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, 
         verticalalignment='top')

plt.tight_layout()
plt.show()


# Creamos una segunda columna relacionada al MontoUSD utilizando shift

# Crear una nueva columna con los valores de 'MontoUSD' desplazados en 12 períodos (mismo mes del año anterior)
df['MontoUSD_t_12'] = df['MontoUSD'].shift(12)

# Verificar el resultado
print(df.head(15))  # Mostrar más filas para ver el efecto


# Eliminamos valores NAN

# Eliminar las filas que contienen valores NaN en cualquier columna
df = df.dropna()

df.info()


# MontoUSD es la variable dependiente principal en este análisis de series temporales. Representa el valor en dólares de las ventas de vino a lo largo del tiempo. Esta columna captura la evolución de las ventas en diferentes países y mercados, y está directamente afectada por la cantidad de cajas vendidas y otros factores relacionados con la demanda y la oferta en el mercado del vino. Al modelar esta variable, se busca entender los patrones de comportamiento de las ventas para prever la demanda futura de vinos, específicamente en el año 2025.
