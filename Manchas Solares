#Importamos librerías
import numpy as np #Para las matrices
from scipy.fft import fft, fftfreq #Para la transformada de fourier y sus frecuencias
import pandas as pd #Para manejar los datos
import matplotlib.pyplot as plt #Para las gráficas

'''
Leemos los datos
'''
Datos = pd.read_csv('SunSpots1749to2015.csv', sep = '\s+')  #Leemos el archivo como un csv con espacios en lugar de comas
años = (Datos.loc[:,'YEAR']-1749)*12 #Creamos un Series (de pandas) que nos convierte los años a meses.
mes = Datos.loc[:,'MON'] #Creamos otro Series (de pandas) que obtiene los meses.
indice = años + mes #Sumamos el año y el mes para obtener el número de mes en el que nos encontramos
indice.name = 't [monts]' #Le ponemos nombre
Datos.index = indice #Cambiamos el índice que se tenía, por el tiempo en nuestros datos.
#Sé que no era necesario lo de los meses, pero me dí cuenta ya que lo había hecho, y decidí dejarlo.

'''
Análisis
'''
Datos = Datos.loc[:, 'SSN'] #Obtenemos los datos que nos interesan.
SSN = Datos.to_numpy() #Los pasamos a una matriz unidimensional.

fourier = np.abs(fft(SSN)) #Calculamos la transformada de fourier para ese conjunto de datos.
densidad_espectral = pd.Series(fourier**2) #La elevamos al cuadrado y la convertimos a un Series de Pandas, para graficar más fácil.
densidad_espectral = 2 * densidad_espectral.iloc[1 : int(len(densidad_espectral) / 2) - 1] #Obtenemos la mitad de los datos (pues solo nos interesa la mitad).

max = densidad_espectral.idxmax() #Buscamos la frecuencia que predomina.
frecuencias = fftfreq(len(densidad_espectral)) #Vemos el espectro de frecuencias que nos dió la FFT.
frecuencia = frecuencias[max] #Obtenemos la frecuencia que predomina
periodo = (1/frecuencia)/12 #Obtenemos el periodo en meses, por lo que lo divido entre 12 para obtener los años
print(periodo) #Imprimimos el periodo.

'''
Gráficas
'''

plt.figure(figsize = ( 12, 6)) #Creamos una imagen.

plt.subplot(1, 2, 1) #Vamos a la primera gráfica
grafica = Datos.plot(x = 't', y = 'SSN') #Serán los datos, con el tiempo en el eje x, y las manchas en el eje y.
grafica.plot() #Creamos la gráfica
plt.xlabel('Meses') #Le damos nombre al eje x
plt.ylabel('# Manchas Solares') #Le samos nombre al eje y
plt.title('Datos') #Le damos nombre a nuestra gráfica.

plt.subplot(1, 2, 2) #Vamos a la segunda gráfica
grafica2 = densidad_espectral.iloc[1:200].plot() #Selecionamos el conjunto de datos a graficar, (de 1 a 200)
grafica2.plot() #Creamos la gráfica.
plt.plot([max, max], [0, densidad_espectral.max() + 0.25 * 10 ** 9], linestyle = 'dashed') #Creamos una línea vertical donde está el máximo.
plt.xlabel('Frecuencia') #Damos nombre al eje x
plt.ylabel('Peso') #Damos nombre al eje y
plt.legend(['Transformada de Fourier', 'Frecuencia máxima']) #Damos nombre a la línea y la transformada.
plt.title('Frecuencias') #Damos nombre a la gráfica.

plt.show() #Mostramos las gráficas
