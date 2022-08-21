import numpy as np
import matplotlib.pyplot as plt

def mostrar (tiempo, serie, formato="-",principio=0,fin=None, nombre=None):
    plt.plot(tiempo[principio:fin],serie[principio:fin],formato,label=nombre)
    plt.xlabel = ("Tiempo")
    plt.ylabel = ("Valor")
    if nombre:
        plt.legend(fontsize=14)
    plt.grid(True)

# inclinacion de la grafica

def tendencia(tiempo,pendiente):
    '''
    La pendiente es el factor de crecimiento en horizontal respecto al vertical
    cuanto mas bajo sea mas horizontal sera la pendiente
    '''
    return pendiente * tiempo

tiempo = np.arange(4 * 365 + 1)
base = 20 # un incremento a toda la grafica
serie = base + tendencia(tiempo,.1)

plt.figure(figsize=(10,6))
mostrar(tiempo,serie)
plt.xlabel = ("Tiempo")
plt.ylabel = ("Valor")
plt.show()

# patrones cada cierto tiempo

def patron(epacio_entre_patron):
    return np.where(epacio_entre_patron < 0.4, np.cos(epacio_entre_patron * 2 * np.pi*1.2),
     1 / np.exp ( 2 * epacio_entre_patron))

def seasonality(tiempo,periodo,amplitud=1,fase=0):
    epacio_entre_patron = ((tiempo + fase) % periodo) / periodo
    return amplitud * patron(epacio_entre_patron)

amplitud = 40
serie = seasonality(tiempo, periodo=365, amplitud=amplitud)

plt.figure(figsize=(10, 6))
mostrar(tiempo, serie)
plt.show()

# sumamos las dos anteriores

pendiente = 0.05
serie = base + tendencia(tiempo, pendiente) + seasonality(tiempo, periodo=365, amplitud=amplitud)
plt.figure(figsize=(10, 6))
mostrar(tiempo, serie)
plt.show()

# ruido

def ruido (tiempo, nivel_ruido=0.5, semilla = None):
    r = np.random.RandomState(semilla)
    return r.randn(len(tiempo)) * nivel_ruido

N_ruido = ruido(tiempo,semilla = 42)
plt.figure(figsize=(10, 6))
mostrar(tiempo, N_ruido)
plt.show()

# sumamos todo

serie = base + tendencia(tiempo, pendiente) + seasonality(tiempo, periodo=365, amplitud=amplitud) + ruido(tiempo,5,semilla = 42)
plt.figure(figsize=(10, 6))
mostrar(tiempo, serie)
plt.show()