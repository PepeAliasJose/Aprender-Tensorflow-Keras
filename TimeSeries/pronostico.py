import numpy as np
import matplotlib.pyplot as plt

def mostrar (tiempo, serie, formato="-",principio=0,fin=None, nombre=None):
    plt.plot(tiempo[principio:fin],serie[principio:fin],formato,label=nombre)
    plt.xlabel = ("Tiempo")
    plt.ylabel = ("Valor")
    if nombre:
        plt.legend(fontsize=14)
    plt.grid(True)

def tendencia(tiempo,pendiente):
    '''
    La pendiente es el factor de crecimiento en horizontal respecto al vertical
    cuanto mas bajo sea mas horizontal sera la pendiente
    '''
    return pendiente * tiempo

def patron(epacio_entre_patron):
    return np.where(epacio_entre_patron < 0.4, np.cos(epacio_entre_patron * 2 * np.pi*1.2),
     1 / np.exp ( 2 * epacio_entre_patron))

def seasonality(tiempo,periodo,amplitud=1,fase=0):
    epacio_entre_patron = ((tiempo + fase) % periodo) / periodo
    return amplitud * patron(epacio_entre_patron)

def ruido (tiempo, nivel_ruido=0.5, semilla = None):
    r = np.random.RandomState(semilla)
    return r.randn(len(tiempo)) * nivel_ruido

tiempo = np.arange(4 * 365 + 1)
for i in tiempo:
    print(i)
base = 20 
amplitud = 40
pendiente = 0.05
N_ruido = ruido(tiempo,semilla = 42)
serie = base + tendencia(tiempo, pendiente) + seasonality(tiempo, periodo=365, amplitud=amplitud) + ruido(tiempo,5,semilla = 42)
plt.figure(figsize=(10, 6))
mostrar(tiempo, serie)
plt.show()

separacion = 1000
tiempo_entrenar = tiempo[:separacion]
x_entrenar = serie[:separacion]
tiempo_valid = tiempo[separacion:]
x_validar = serie[separacion:]

pronostico = serie[separacion -1 : -1]
plt.figure(figsize=(10, 6))
mostrar(tiempo_valid, x_validar, nombre="Serie")
mostrar(tiempo_valid, pronostico, nombre="Pronostico")
plt.show()

# zoom
plt.figure(figsize=(10, 6))
mostrar(tiempo_valid, x_validar, principio=0, fin=150, nombre="Serie")
mostrar(tiempo_valid, pronostico, principio=1, fin=151, nombre="Pronostico")
plt.show()

error = pronostico - x_validar
abs_errores = np.abs(error)
mae = abs_errores.mean()
print(mae)