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

# Descripción del contexto: Determinar la demanda global de vinos basado en la venta histórica

# ### 1. Importe de Librerias

#pip install statsmodels


# Importar librerias
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


# ### 2. Lectura de archivo

# Especifica la ruta del archivo Excel
file_path = r'..\data\VentaHistoricaVM.xlsx'

# Lee el archivo Excel
df = pd.read_excel(file_path, sheet_name='Hoja4')

# Mostrar las primeras filas del DataFrame
df.head(10)


# ### 3. Análisis exploratorio

# Cambiar el nombre de la columna 'ano' a 'año'
df.rename(columns={'ano': 'año'}, inplace=True)

# Mostrar las primeras filas del DataFrame
df.head(10)


# Mostrar las últimas filas del DataFrame
df.tail(10)


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


# ### 4. Limpieza de datos

# #### 4.1 Busqueda de Valores Nulos

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


# #### 4.2 Tratamiento de valores Nulos: campos 'mes' , 'CodCliente' y 'CodProducto'

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


# #### 4.3 Tratamiento de valores Nulos: campo 'Cosecha'

# Desición: Lo que buscamos es determinar la demanda global de vinos dada la venta histórica, es por ello que eliminamos la variable cosecha.

# Eliminar la columna 'cosecha'
df.drop(columns=['cosecha'], inplace=True)

# Verificar que la columna ha sido eliminada
df.head()


# #### 4.4 Tratamiento de valores Nulos: campo 'Pais'

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


# **Desición:** se elimina el unico registro con NAN de Pais porque no existen registros anteriores para el cliente: '2002-8' con los cuales comparar.

# Eliminar las filas donde 'CodCliente' es igual a '2002-8'
df= df[df['CodCliente'] != '2002-8']

# Verificar si las filas fueron eliminadas
print(f"Total de filas después de eliminar '2002-8': {len(df)}")


# Contar cuántos NaN hay en cada columna
nan_count = df.isna().sum()
print("Valores NaN por columna:")
print(nan_count)


# ### 5. Ingenieria de Columnas

# #### 5.1 Eliminar campo MontoCLP
# Solo nos quedaremos con el campo MontoUSD, que corresponde a las ventas en moneda dólar

# Eliminar la columna 'MontoCLP'
df.drop(columns=['MontoCLP'], inplace=True)

# Verificar que la columna ha sido eliminada
df.info()


# Convertir la columna 'mes' y 'año' a tipo entero
df['año'] = df['año'].astype(int)
df['mes'] = df['mes'].astype(int)


# Verificar si hay valores infinitos
print(df[['año', 'mes']].replace([np.inf, -np.inf], np.nan).isna().sum())


# ### 5.2 Crear la columna Periodo 
# Combinando 'año' y 'mes', establecerla como indice y elimninar 'año' y 'mes'

# Crear la columna 'periodo' combinando 'año' y 'mes' con un separador
df['periodo'] = df['año'].astype(str) + '-' + df['mes'].astype(int).astype(str).str.zfill(2)

# Mostrar las primeras filas para verificar el resultado
print(df[['año', 'mes', 'periodo']].head())


# Colocar la columna 'periodo' como la primera y eliminar 'año' y 'mes'
df = df[['periodo'] + [col for col in df.columns if col not in ['año', 'mes', 'periodo']]]

# Verificar el nuevo DataFrame
df.head()


# ### 6. Serie temporal

# Lo que buscamos es determinar la demanda global de vinos dada la venta histórica, es por ello que eliminamos variables tales como Codcliente, Codproducto, mercado y país

# Eliminar las columnas 'CodCliente', 'codproducto', 'mercado' y 'Pais'
df.drop(columns=['CodCliente', 'codproducto', 'mercado', 'Pais'], inplace=True)

# Verificar que las columnas han sido eliminadas
print(df.head())


# Agrupamos por Periodo y calculamos el monto total vendido por mes por mes

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


# #### 6.1 Modelo Autorregresivo de primer orden (AR1)
# 
# - *Modelo genérico AR(1)*: 
#   $
#   \ Y_{t,i} = \beta_0 + \beta_1 Y_{t-1,i} + u_t
#   $
#   
# 
#   
# - *Selección de $Y_t$*: 
#   $
#     Y_t \leftarrow MontoUSD_t
#   $
# 
# - *Regresión Lineal Simple*: 
#   $
#   X \leftarrow Y_{t-1}, \quad Y \leftarrow Y_t
#   $

# 
# 
# Se eligió *MontoUSD* como la variable dependiente principal en este análisis de series temporales. Esta variable representa el valor en dólares de las ventas de vino a lo largo del tiempo, capturando la evolución de las ventas en diferentes países y mercados. Está directamente influenciada por la cantidad de cajas vendidas, así como por otros factores relacionados con la demanda y la oferta en el mercado del vino. Al modelar esta variable, se busca comprender los patrones de comportamiento de las ventas con el objetivo de prever la demanda futura de vinos, específicamente para el año 2025.
# 

# 
# Por lo tanto, aplicando las fórmulas a nuestro objetivo, tendríamos:
# 
# - **1. Modelo genérico AR(1)**:
#   $
#   Y_{t,i} = \beta_0 + \beta_1 Y_{t-1,i} + u_t
#   $
# 
# - Donde:
# 
#   $
#   Y_{t,i} 
#   $
#   es el valor predicho de las ventas en dólares (*MontoUSD*) en el tiempo $ t $.
# 
#   $
#   Y_{t-1,i} 
#   $
#   es el valor de las ventas en dólares en el mes anterior.
# 
#   $
#   \beta_0 
#   $
#   es el intercepto o término constante, que captura el nivel base de las ventas.
# 
#   $
#   \beta_1 
#   $
#   es el coeficiente de regresión que mide el impacto de las ventas del mes anterior ($Y_{t-1}$) sobre las ventas actuales ($Y_t$).
# 
#   $
#   u_t 
#   $
#   es el término de error aleatorio en el tiempo $t$, que captura la parte de las ventas que no puede ser explicada por el modelo (factores externos o imprevistos).
# 
# 
# 

# Análisis visual. Graficar las ventas totales por mes

df['periodo'] = pd.to_datetime(df['periodo'], format='%Y-%m')
# Agrupar por año y mes, sumando 'MontoUSD'
df_monthly = df.groupby('periodo', as_index=False)['MontoUSD'].sum().copy()

# Graficar las ventas totales por mes
plt.figure(figsize=(12, 6))
plt.plot(df_monthly['periodo'], df_monthly['MontoUSD'])

# Añadir etiquetas y título
plt.title('Total MontoUSD Vendido Cada Mes')
plt.xlabel('Mes')
plt.ylabel('MontoUSD')

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45)

# Ajustar automáticamente las etiquetas del eje x para evitar superposición
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Mostrar cada interval meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato: Año-Mes

plt.grid(True)

plt.tight_layout()
plt.show()


# Podemos observar algunos patrones que sugieren posibles problemas de no estacionariedad:
# 
# - *Tendencia*: Hay una tendencia al alza entre el 2004 y el añp 2012, y una a la baja entre el año 2018 y 2024.
# 
# - *Varianza No constante*: La varianza parece aumentar y luego disminuir en diferentes partes de la serie. En un periodo (2008-2012) las fluctuaciones parecen ser más amplias, mientras que en otras épocas (2016-2018) las ventas parecen estabilizarse un poco. Este tipo de comportamiento es un indicativo de que la serie puede no ser estacionaria.
# 
# - *Estacionalidad:* Es probable que haya estacionalidad, ya que hay picos recurrentes en intervalos regulares. Esto sugiere que ciertos meses o épocas del año experimentan más ventas de manera repetitiva.

# Crear shifts
df_monthly['t-1']=df_monthly['MontoUSD'].shift(1) 
df_monthly['t-12']=df_monthly['MontoUSD'].shift(12)

# Botar Nans
df_monthly.dropna(inplace=True)

df_monthly.head()


# Calcular la correlación entre ontoUSD y MontoUSD_t-1
correlacion = np.corrcoef(df_monthly['MontoUSD'], df_monthly['t-1'])[0, 1]

# Crear el gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(df_monthly['t-1'], df_monthly['MontoUSD'], alpha=0.7)

# Añadir etiquetas y título
plt.title('Scatter Plot entre MontoUSD y t-1')
plt.xlabel('MontoUSD (t-1)')
plt.ylabel('MontoUSD (t)')
plt.annotate(xy=(0.8,0.8), text=f'Correlación: {correlacion:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5), xycoords='axes fraction')

# Mostrar el gráfico
plt.grid(True)
plt.show()


ar1= LinearRegression(fit_intercept=True)
ar1.fit(X=df_monthly[['t-1']], y=df_monthly['MontoUSD'])

print("b_1 =",ar1.coef_[0],"b_0 =",ar1.intercept_)
print("R^2 =",ar1.score(df_monthly[['t-1']], df_monthly['MontoUSD']))



# Calcular la correlación entre Delta_MontoUSD y Delta_MontoUSD_t-1
correlacion = np.corrcoef(df_monthly['MontoUSD'], df_monthly['t-12'])[0, 1]

# Crear el gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(df_monthly['t-12'], df_monthly['MontoUSD'], alpha=0.7)


# Añadir etiquetas y título
plt.title('Scatter Plot entre Delta_MontoUSD y Delta_MontoUSD_t-12')
plt.xlabel('Delta_MontoUSD (t-12)')
plt.ylabel('Delta_MontoUSD (t)')
plt.annotate(xy=(0.8,0.8), text=f'Correlación: {correlacion:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5), xycoords='axes fraction')

# Mostrar el gráfico
plt.grid(True)
plt.show()


ar12= LinearRegression(fit_intercept=True)
ar12.fit(X=df_monthly[['t-12']], y=df_monthly['MontoUSD'])
print("b_1 =",ar12.coef_[0],"b_0 =",ar12.intercept_)
print("R^2 =",ar12.score(df_monthly[['t-12']], df_monthly['MontoUSD']))


# Si bien dado que $\beta_1 = 0.58$ nos permite confirmar que existe una estacionalidad.

# #### 6.2 Modelo Autorregresivo de primer a cuarto orden (AR1 ... AR4)
# 
# - *Modelo genérico AR(4)*: 
#   $
#   \ Y_{t,i} = \beta_0 + \beta_1 Y_{t-1,i} + \beta_2 Y_{t-2,i} + \beta_3 Y_{t-3,i} + \beta_4 Y_{t-4,i} + u_t
#   $
#   
# 
#   
# - *Selección de*
#   $
#    Y_t: \: Y_t \leftarrow \Delta MontoUSD_t
#   $
# 
# - *Regresión Lineal Multiple*: 
#   $
#   X_{t-1} \leftarrow Y_{t-1}, \: X_{t-2} \leftarrow Y_{t-2}, \: X_{t-3} \leftarrow Y_{t-3}, \: X_{t-4} \leftarrow Y_{t-4}; \quad Y \leftarrow Y_t
#   $

# 
# 
# Se eligió *$\Delta$MontoUSD* como la variable dependiente principal en este análisis de series temporales. Esta variable representa el cambio en el valor en dólares de las ventas de vino a lo largo del tiempo, capturando la variacion de las ventas totales.
# 

# 
# Por lo tanto, aplicando las fórmulas a nuestro objetivo, tendríamos:
# 
# - **2. Modelo genérico AR(4)**:
#   $
#   \ Y_{t,i} = \beta_0 + \beta_1 Y_{t-1,i} + \beta_2 Y_{t-2,i} + \beta_3 Y_{t-3,i} + \beta_4 Y_{t-4,i} + u_t
#   $
# 
# - Donde:
# 
#   $
#   Y_{t,i} 
#   $
#   es el valor predicho de las ventas en dólares (*MontoUSD*) en el tiempo $ t $.
# 
#   $
#   Y_{t-1,i} 
#   $
#   es el valor de las ventas en dólares en el mes anterior.
# 
#   $
#   \beta_0 
#   $
#   es el intercepto o término constante, que captura el nivel base de las ventas.
# 
#   $
#   \beta_{1\dots 4} 
#   $
#   es el coeficiente de regresión que mide el impacto del cambio de las ventas del mes correspondiente ($Y_{t-1\dots 4}$) sobre el cambio actual ($Y_t$).
# 
#   $
#   u_t 
#   $
#   es el término de error aleatorio en el tiempo $t$, que captura la parte de la diferencia que no puede ser explicada por el modelo (factores externos o imprevistos).
# 
# 
# 

# Definición de una diferencia o cambio en una serie temporal
# - *Definición de Delta_MontoUSD_t*:  
# 
#   $
#   \Delta MontoUSD_t = MontoUSD_t - MontoUSD_{t-1}
#   $

# Calcular la diferencia entre las ventas actuales y las del mes anterior (Delta_MontoUSD)
df_monthly['Delta_MontoUSD'] = df_monthly['MontoUSD'] - df_monthly['t-1']

# Crear la columna Delta_MontoUSD desplazada (rezago de un periodo, AR(1))
df_monthly['Delta_MontoUSD_t-1'] = df_monthly['Delta_MontoUSD'].shift(1)
df_monthly['Delta_MontoUSD_t-2'] = df_monthly['Delta_MontoUSD'].shift(2)
df_monthly['Delta_MontoUSD_t-3'] = df_monthly['Delta_MontoUSD'].shift(3)
df_monthly['Delta_MontoUSD_t-4'] = df_monthly['Delta_MontoUSD'].shift(4)
df_monthly.head()



# Eliminar las filas con NaN que se generan debido al desplazamiento
df_monthly = df_monthly.dropna()


# Mostrar las primeras filas del DataFrame limpio
df_monthly.head()


# Graficar la diferencia en MontoUSD (Delta MontoUSD)
plt.figure(figsize=(12, 6))
plt.plot(df_monthly['periodo'], df_monthly['Delta_MontoUSD'])
plt.title('Cambio Mes a Mes en las Ventas de Vino (Delta MontoUSD)')
plt.xlabel('Fecha')
plt.ylabel('Delta MontoUSD')

# Ajustar automáticamente las etiquetas del eje x para evitar superposición

# Rotar las etiquetas del eje x para mejorar la legibilidad
plt.xticks(rotation=45)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))  # Mostrar cada interval meses
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Formato: Año

plt.grid(True)
plt.show()


#Observamos la estructura de la tabla
df_monthly.shape


# Grafico de dispersión

# Calcular la correlación entre Delta_MontoUSD y Delta_MontoUSD_t-1
correlacion = np.corrcoef(df_monthly['Delta_MontoUSD'], df_monthly['Delta_MontoUSD_t-1'])[0, 1]

# Crear el gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(df_monthly['Delta_MontoUSD_t-1'], df_monthly['Delta_MontoUSD'], alpha=0.7)

# Añadir etiquetas y título
plt.title('Scatter Plot entre Delta_MontoUSD y Delta_MontoUSD_t-1')
plt.xlabel('Delta_MontoUSD (t-1)')
plt.ylabel('Delta_MontoUSD (t)')
plt.annotate(xy=(0.8,0.8), text=f'Correlación: {correlacion:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5), xycoords='axes fraction')

# Mostrar el gráfico
plt.grid(True)
plt.show()


# La correlación de -0.56 indica que existe una correlación negativa moderada entre los cambios en las ventas de un mes y los cambios del mes anterior. Esto sugiere que, en general, si el cambio en las ventas en el mes anterior fue positivo, es probable que el cambio en el mes actual sea negativo, y viceversa. Es decir, tiende a cambiar de dirección en tiempos secuenciales.

# ##### 6.2.1 Modelo Autorregresivo de primer orden (AR1)
# - Este modelo se crea para poder comparar eventualmente con el de 4to orden
# 
# 

# Extraer los datos
X = df_monthly[['Delta_MontoUSD_t-1']].dropna()  # Valores de Delta_MontoUSD_t-1 (independiente)
Y = df_monthly['Delta_MontoUSD'].dropna()  # Valores de Delta_MontoUSD (dependiente)

# Alinear los datos eliminando las filas NaN en ambas columnas
aligned_data = pd.concat([X, Y], axis=1).dropna()
X = aligned_data['Delta_MontoUSD_t-1'].values.reshape(-1, 1)
Y = aligned_data['Delta_MontoUSD'].values

# Calcular la correlación entre Delta_MontoUSD y Delta_MontoUSD_t-1
correlacion = np.corrcoef(Y, X[:, 0])[0, 1]

# Ajustar el modelo de regresión lineal
reg = LinearRegression().fit(X, Y)
Y_pred = reg.predict(X)  # Predicciones de la regresión lineal

# Crear el gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(X, Y, alpha=0.7, label='Datos')
plt.plot(X, Y_pred, color='black', label='Línea de Regresión', alpha=0.7)

# Añadir etiquetas y título
plt.title('Scatter Plot con Regresión Lineal: Delta_MontoUSD y Delta_MontoUSD_t-1')
plt.xlabel('Delta_MontoUSD (t-1)')
plt.ylabel('Delta_MontoUSD (t)')
plt.text(X.min(), Y.min(), f'Correlación: {correlacion:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Mostrar la leyenda
plt.legend()

# Mostrar el gráfico
plt.grid(True)
plt.show()


# - Calculo de R2

# Ajustar el modelo de regresión lineal
reg = LinearRegression().fit(X, Y)

# Calcular R^2
r_squared_1 = reg.score(X, Y)

# Mostrar el valor de R^2
print(f"R^2: {r_squared_1:.4f}")


# $R^2=0.3130$ indica que el modelo explica aproximadamente el $31.3%$ de la variabilidad en los cambios en las ventas $(\Delta MontoUSD)$ en función de los cambios del mes anterior  $(\Delta MontoUSD_{t-1})$. Esto significa que hay factores adicionales que afectan las ventas que no están capturados únicamente por el cambio en las ventas del mes anterior.

# Definir la longitud del conjunto de entrenamiento (80% de los datos)
n_train = int(len(df_monthly) * 0.8)

# Extraer los datos de Delta_MontoUSD y Delta_MontoUSD_t-1
X = df_monthly[['Delta_MontoUSD_t-1']].dropna().values.reshape(-1, 1)
Y = df_monthly['Delta_MontoUSD'].dropna().values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# Ajustar el modelo AR(1) usando regresión lineal
model = LinearRegression().fit(X_train, Y_train)

# Realizar predicciones sobre el conjunto de prueba
Y_pred = model.predict(X_test)

# Crear un vector de fechas para la predicción
fechas = df_monthly['periodo'].values

# Graficar los resultados
plt.figure(figsize=(12, 6))

# Gráfica del conjunto de entrenamiento (Región de ajuste MCO)
plt.plot(fechas[:n_train], Y_train, linestyle='-', label='Región de ajuste MCO')
plt.axvline(x=fechas[n_train], color='gray', linestyle='--', label='Límite entre ajuste y predicción')

# Gráfica de la predicción
plt.plot(fechas[n_train:], Y_pred, color='#ee3940', linestyle='-', label='Predicción')
plt.plot(fechas[n_train:], Y_test, color='g', linestyle='-', label='Predicción', alpha=0.5)

# Calcular R^2 del conjunto de entrenamiento
r_squared_1_pred = model.score(X_test, Y_test)

# Calcular el RMSE
rmse_1 = root_mean_squared_error(Y_test, Y_pred)


# Añadir etiquetas y título
plt.title('Predicción AR(1) de Delta_MontoUSD')
plt.xlabel('Fecha')
plt.ylabel('Delta MontoUSD')
plt.text(fechas[int(n_train/1.5)], 800000, f'$R^2_{{\t{{predicción}}}} = {r_squared_1_pred:.2f}$ \n$RMSE = {rmse_1:.0f}$', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

# Añadir la leyenda
plt.legend()

# Mostrar la cuadrícula y el gráfico
plt.grid(True)
plt.show()


# Identificar Estacionariedad

# Graficar la serie temporal de Delta_MontoUSD
plt.figure(figsize=(12, 6))
plt.plot(df_monthly['periodo'], df_monthly['Delta_MontoUSD'], linestyle='-')

# Añadir etiquetas y título
plt.title('Serie Temporal de Delta_MontoUSD')
plt.xlabel('Fecha')
plt.ylabel('Cambio en MontoUSD (Delta_MontoUSD)')
# plt.legend(loc='upper right')
plt.grid(True)

# Mostrar el gráfico
plt.show()


# Verificar si el modelo es estacionario:
# 
# Intercepto $\beta_0: -966.97$  
# Coeficiente $\beta_1: -0.5622 \Rightarrow |\beta_1| = 0.5622 < 1$

# ##### 6.2.2 Modelo Autorregresivo de segundo orden (AR2)

# Calcular la correlación entre Delta_MontoUSD y Delta_MontoUSD_t-1
correlacion = np.corrcoef(df_monthly['Delta_MontoUSD'], df_monthly['Delta_MontoUSD_t-2'])[0, 1]

# Crear el gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(df_monthly['Delta_MontoUSD_t-2'], df_monthly['Delta_MontoUSD'], alpha=0.7)

# Añadir etiquetas y título
plt.title('Scatter Plot entre Delta_MontoUSD y Delta_MontoUSD_t-2')
plt.xlabel('Delta_MontoUSD (t-2)')
plt.ylabel('Delta_MontoUSD (t)')
plt.annotate(xy=(0.8,0.8), text=f'Correlación: {correlacion:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5), xycoords='axes fraction')

# Mostrar el gráfico
plt.grid(True)
plt.show()


# Extraer los datos
X = df_monthly[['Delta_MontoUSD_t-1','Delta_MontoUSD_t-2']]  # Valores de Delta_MontoUSD_t-1, t-2 (independiente)
Y = df_monthly['Delta_MontoUSD']  # Valores de Delta_MontoUSD (dependiente)


# Ajustar el modelo de regresión lineal
reg = LinearRegression().fit(X, Y)
Y_pred = reg.predict(X)  # Predicciones de la regresión lineal

r2 = reg.score(X, Y)
n = X_test.shape[0]    
p = X_test.shape[1]  

adjusted_r2_2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"R² Ajustado: {adjusted_r2_2}")


# Definir la longitud del conjunto de entrenamiento (80% de los datos)
n_train = int(len(df_monthly) * 0.8)

# Extraer los datos de Delta_MontoUSD y Delta_MontoUSD_t-1
X = df_monthly[['Delta_MontoUSD_t-1','Delta_MontoUSD_t-2']].values
Y = df_monthly['Delta_MontoUSD'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# Ajustar el modelo AR(1) usando regresión lineal
model = LinearRegression().fit(X_train, Y_train)

# Realizar predicciones sobre el conjunto de prueba
Y_pred = model.predict(X_test)

# Crear un vector de fechas para la predicción
fechas = df_monthly['periodo'].values

# Graficar los resultados
plt.figure(figsize=(12, 6))

# Gráfica del conjunto de entrenamiento (Región de ajuste MCO)
plt.plot(fechas[:n_train], Y_train, linestyle='-', label='Región de ajuste MCO')
plt.axvline(x=fechas[n_train], color='gray', linestyle='--', label='Límite entre ajuste y predicción')

# Gráfica de la predicción
plt.plot(fechas[n_train:], Y_pred, color='#ee3940', linestyle='-', label='Predicción')
plt.plot(fechas[n_train:], Y_test, color='g', linestyle='-', label='Real', alpha=0.5)
# Calcular R^2 del conjunto de entrenamiento
r_squared = model.score(X_test, Y_test)
adj_r2_2 = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
# Calcular RMSE
rmse_2 = root_mean_squared_error(Y_test, Y_pred)

# Añadir etiquetas y título
plt.title('Predicción AR(4) de Delta_MontoUSD')
plt.xlabel('Fecha')
plt.ylabel('Delta MontoUSD')
plt.text(fechas[int(n_train/1.5)], 850000, f'$R^2_{{\t{{predicción}}}}= {adj_r2_2:.2f}$ \n$RMSE = {rmse_2:.0f}$', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

# Añadir la leyenda
plt.legend()

# Mostrar la cuadrícula y el gráfico
plt.grid(True)
plt.show()


coef = model.coef_

print(f"Coeficientes del modelo: β₀ = {model.intercept_:.0f}, β₁ = {coef[0]:.4f}, β₂ = {coef[1]:.4f}")


# Verificar si el modelo es estacionario:
# 
# Intercepto $\beta_0: 3745$   
# 
# Coeficientes:  
# - $\beta_1: -0.7636$
# - $\beta_2: -0.3239$
# 
# Todos con valor absoluto menor a $1$

# ##### 6.2.3 Modelo Autorregresivo de tercer orden (AR3)

# Calcular la correlación entre Delta_MontoUSD y Delta_MontoUSD_t-1
correlacion = np.corrcoef(df_monthly['Delta_MontoUSD'], df_monthly['Delta_MontoUSD_t-3'])[0, 1]

# Crear el gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(df_monthly['Delta_MontoUSD_t-3'], df_monthly['Delta_MontoUSD'], alpha=0.7)

# Añadir etiquetas y título
plt.title('Scatter Plot entre Delta_MontoUSD y Delta_MontoUSD_t-3')
plt.xlabel('Delta_MontoUSD (t-3)')
plt.ylabel('Delta_MontoUSD (t)')
plt.annotate(xy=(0.8,0.8), text=f'Correlación: {correlacion:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5), xycoords='axes fraction')

# Mostrar el gráfico
plt.grid(True)
plt.show()


# Extraer los datos
X = df_monthly[['Delta_MontoUSD_t-1','Delta_MontoUSD_t-2', 'Delta_MontoUSD_t-3']]  # Valores de Delta_MontoUSD_t-1, ..., t-3 (independiente)
Y = df_monthly['Delta_MontoUSD']  # Valores de Delta_MontoUSD (dependiente)


# Ajustar el modelo de regresión lineal
reg = LinearRegression().fit(X, Y)
Y_pred = reg.predict(X)  # Predicciones de la regresión lineal

r2 = reg.score(X, Y)
n = X_test.shape[0]    
p = X_test.shape[1]  

adjusted_r2_3 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"R² Ajustado: {adjusted_r2_3}")


# Definir la longitud del conjunto de entrenamiento (80% de los datos)
n_train = int(len(df_monthly) * 0.8)

# Extraer los datos de Delta_MontoUSD y Delta_MontoUSD_t-1
X = df_monthly[['Delta_MontoUSD_t-1','Delta_MontoUSD_t-2','Delta_MontoUSD_t-3']].values
Y = df_monthly['Delta_MontoUSD'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# Ajustar el modelo AR(1) usando regresión lineal
model = LinearRegression().fit(X_train, Y_train)

# Realizar predicciones sobre el conjunto de prueba
Y_pred = model.predict(X_test)

# Crear un vector de fechas para la predicción
fechas = df_monthly['periodo'].values

# Graficar los resultados
plt.figure(figsize=(12, 6))

# Gráfica del conjunto de entrenamiento (Región de ajuste MCO)
plt.plot(fechas[:n_train], Y_train, linestyle='-', label='Región de ajuste MCO')
plt.axvline(x=fechas[n_train], color='gray', linestyle='--', label='Límite entre ajuste y predicción')

# Gráfica de la predicción
plt.plot(fechas[n_train:], Y_pred, color='#ee3940', linestyle='-', label='Predicción')
plt.plot(fechas[n_train:], Y_test, color='g', linestyle='-', label='Real', alpha=0.5)
# Calcular R^2 del conjunto de entrenamiento
r_squared = model.score(X_test, Y_test)
adj_r2_3 = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
# Calcular RMSE
rmse_3 = root_mean_squared_error(Y_test, Y_pred)

# Añadir etiquetas y título
plt.title('Predicción AR(4) de Delta_MontoUSD')
plt.xlabel('Fecha')
plt.ylabel('Delta MontoUSD')
plt.text(fechas[int(n_train/1.5)], 850000, f'$R^2_{{\t{{predicción}}}}= {adj_r2_3:.2f}$ \n$RMSE = {rmse_3:.0f}$', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

# Añadir la leyenda
plt.legend()

# Mostrar la cuadrícula y el gráfico
plt.grid(True)
plt.show()


coef = model.coef_

print(f"Coeficientes del modelo: β₀ = {model.intercept_:.0f}, β₁ = {coef[0]:.4f}, β₂ = {coef[1]:.4f}, β₃ = {coef[2]:.4f}")


# Verificar si el modelo es estacionario:
# 
# Intercepto $\beta_0: 5549$   
# 
# Coeficientes:  
# - $\beta_1: -0.8497$
# - $\beta_2: -0.5304$
# - $\beta_3: -0.2713$
# 
# Todos con valor absoluto menor a $1$

# ##### 6.2.4 Modelo Autorregresivo de cuarto orden (AR4)

# Calcular la correlación entre Delta_MontoUSD y Delta_MontoUSD_t-1
correlacion = np.corrcoef(df_monthly['Delta_MontoUSD'], df_monthly['Delta_MontoUSD_t-4'])[0, 1]

# Crear el gráfico de dispersión
plt.figure(figsize=(12, 6))
plt.scatter(df_monthly['Delta_MontoUSD_t-4'], df_monthly['Delta_MontoUSD'], alpha=0.7)

# Añadir etiquetas y título
plt.title('Scatter Plot entre Delta_MontoUSD y Delta_MontoUSD_t-4')
plt.xlabel('Delta_MontoUSD (t-4)')
plt.ylabel('Delta_MontoUSD (t)')
plt.annotate(xy=(0.8,0.8), text=f'Correlación: {correlacion:.2f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5), xycoords='axes fraction')

# Mostrar el gráfico
plt.grid(True)
plt.show()


# Extraer los datos
X = df_monthly[['Delta_MontoUSD_t-1','Delta_MontoUSD_t-2','Delta_MontoUSD_t-3','Delta_MontoUSD_t-4']]  # Valores de Delta_MontoUSD_t-1...4 (independiente)
Y = df_monthly['Delta_MontoUSD'].dropna()  # Valores de Delta_MontoUSD (dependiente)


# Ajustar el modelo de regresión lineal
reg = LinearRegression().fit(X, Y)
Y_pred = reg.predict(X)  # Predicciones de la regresión lineal

r2 = reg.score(X, Y)
n = X_test.shape[0]    
p = X_test.shape[1]  

adjusted_r2_4 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"R² Ajustado: {adjusted_r2_4}")


# Definir la longitud del conjunto de entrenamiento (80% de los datos)
n_train = int(len(df_monthly) * 0.8)

# Extraer los datos de Delta_MontoUSD y Delta_MontoUSD_t-1
X = X = df_monthly[['Delta_MontoUSD_t-1','Delta_MontoUSD_t-2','Delta_MontoUSD_t-3','Delta_MontoUSD_t-4']].values
Y = df_monthly['Delta_MontoUSD'].values

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test = X[:n_train], X[n_train:]
Y_train, Y_test = Y[:n_train], Y[n_train:]

# Ajustar el modelo AR(1) usando regresión lineal
model = LinearRegression().fit(X_train, Y_train)

# Realizar predicciones sobre el conjunto de prueba
Y_pred = model.predict(X_test)

# Crear un vector de fechas para la predicción
fechas = df_monthly['periodo'].values

# Graficar los resultados
plt.figure(figsize=(12, 6))

# Gráfica del conjunto de entrenamiento (Región de ajuste MCO)
plt.plot(fechas[:n_train], Y_train, linestyle='-', label='Región de ajuste MCO')
plt.axvline(x=fechas[n_train], color='gray', linestyle='--', label='Límite entre ajuste y predicción')

# Gráfica de la predicción
plt.plot(fechas[n_train:], Y_pred, color='#ee3940', linestyle='-', label='Predicción')
plt.plot(fechas[n_train:], Y_test, color='g', linestyle='-', label='Real', alpha=0.5)
# Calcular R^2 del conjunto de entrenamiento
r_squared = model.score(X_test, Y_test)
adj_r2_4 = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
# Calcular RMSE
rmse_4 = root_mean_squared_error(Y_test, Y_pred)

# Añadir etiquetas y título
plt.title('Predicción AR(4) de Delta_MontoUSD')
plt.xlabel('Fecha')
plt.ylabel('Delta MontoUSD')
plt.text(fechas[int(n_train/1.5)], 850000, f'$R^2_{{\t{{predicción}}}}= {adj_r2_4:.2f}$ \n$RMSE = {rmse_4:.0f}$', fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

# Añadir la leyenda
plt.legend()

# Mostrar la cuadrícula y el gráfico
plt.grid(True)
plt.show()


coef = model.coef_

print(f"Coeficientes del modelo: β₀ = {model.intercept_:.0f}, β₁ = {coef[0]:.4f}, β₂ = {coef[1]:.4f}, β₃ = {coef[2]:.4f}, β₄ = {coef[3]:.4f}")



# Verificar si el modelo es estacionario:
# 
# Intercepto $\beta_0: 6494$  
# Coeficientes:  
# - $\beta_1: -0.8782$
# - $\beta_2: -0.5869$
# - $\beta_3: -0.3621$
# - $\beta_4: -0.1070$
# 
# Todos con valor absoluto menor a $1$

# 

Resultados = pd.DataFrame({
    'AR(1)': {'RMSE': rmse_1, 'R² ajustado': r_squared_1, 'R² ajustado predicción': r_squared_1_pred},
    'AR(2)': {'RMSE': rmse_2, 'R² ajustado': adjusted_r2_2, 'R² ajustado predicción': adj_r2_2},
    'AR(3)': {'RMSE': rmse_3, 'R² ajustado': adjusted_r2_3, 'R² ajustado predicción': adj_r2_3},
    'AR(4)': {'RMSE': rmse_4, 'R² ajustado': adjusted_r2_4, 'R² ajustado predicción': adj_r2_4}
})
Resultados

