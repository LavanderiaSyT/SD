import matplotlib.pyplot as plt

# Leemos los datos desde el archivo
def leerDatos(archivo):
    with open(archivo, 'r') as file:
        lines = file.readlines()
        data = [[float(value) for value in line.split()] for line in lines]
    return data

# Generamos el gráfico
def generarGrafico(datos):
    horas = list(range(24))
    for i, fila in enumerate(datos):
        plt.plot(horas, fila, label=f'Día {i+1}')

    plt.xlabel('Horas del día')
    plt.ylabel('Valores')
    plt.title('Datos para cada hora del día')
    plt.legend()
    plt.show()

# Archivo de texto con los datos
archivoDatos = 'Predicciones.txt'

# Leemos los datos desde el archivo
datos = leerDatos(archivoDatos)

generarGrafico(datos)
