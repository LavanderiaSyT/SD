#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>

// Estructura para almacenar distancia y el índice correspondiente
typedef struct
{
    double distancia;
    int indice;
} DistanciaIndice;

// Función de comparación para qsort que ordena DistanciaIndice por distancia
int compararDistancias(const void *a, const void *b)
{
    double distanciaA = ((DistanciaIndice *)a)->distancia;
    double distanciaB = ((DistanciaIndice *)b)->distancia;

    if (distanciaA < distanciaB)
        return -1;
    else if (distanciaA > distanciaB)
        return 1;
    else
        return 0;
}

// Calculamos la distancia euclidiana entre dos series temporales
double calcularDistancia(double *serieTemporal1, double *serieTemporal2, int h)
{
    double distancia = 0.0;
    for (int i = 0; i < h; ++i)
    {
        distancia += (serieTemporal1[i] - serieTemporal2[i]) * (serieTemporal1[i] - serieTemporal2[i]);
    }
    return distancia;
}

// Predice el siguiente valor de una serie temporal utilizando k
double predecir(double **datos, int filas, int columnas, int k, int h, int filaActual)
{
    DistanciaIndice *distancias = (DistanciaIndice *)calloc(filas, sizeof(DistanciaIndice));
    if (distancias == NULL)
    {
        perror("Error al asignar memoria para distancias");
        exit(1);
    }

    // Calculamos las distancias entre la fila actual y todas las demás filas
    for (int i = 0; i < filas; ++i)
    {
        distancias[i].distancia = calcularDistancia(datos[i], datos[filaActual], columnas);
        distancias[i].indice = i;
    }

    // Ordenamos las distancias de menor a mayor
    qsort(distancias, filas - h, sizeof(DistanciaIndice), compararDistancias);

    double prediccion = 0.0;

    // Sumamos los valores de los k vecinos más cercanos para predecir el siguiente valor
    for (int i = 0; i < k; ++i)
    {
        int indiceVecino = distancias[i].indice;
        for (int j = 0; j < h; ++j)
        {
            prediccion += datos[indiceVecino][j];
        }
    }

    prediccion /= k * h;

    // Liberamos la memoria asignada para distancias
    free(distancias);

    return prediccion;
}

// Calculamos el MAPE (Mean Absolute Percentage Error) entre los datos reales y las predicciones
double calcularMAPE(double **datosReales, double **predicciones, int filas, int columnas)
{
    double errorTotal = 0.0;
    int contador = 0;

    // Calculamos el error para cada valor en los datos
    for (int i = 0; i < filas; ++i)
    {
        for (int j = 0; j < columnas; ++j)
        {
            double diferencia = datosReales[i][j] - predicciones[i][j];
            double denominador = fabs(datosReales[i][j]) + 1e-10;

            // Evitamos la división por cero
            if (denominador != 0.0)
            {
                errorTotal += fabs(diferencia / denominador);
                contador++;
            }
        }
    }
    return (contador > 0) ? (errorTotal / contador) * 100.0 : 0.0;
}

// Imprimimos las predicciones en un archivo de texto
void imprimirPredicciones(double **predicciones, int filas, int columnas)
{
    FILE *archivoPredicciones = fopen("Predicciones.txt", "w");
    if (!archivoPredicciones)
    {
        perror("Error al abrir el archivo de predicciones");
        exit(1);
    }

    // Escribimos las predicciones en el archivo
    for (int i = 0; i < filas; ++i)
    {
        for (int j = 0; j < columnas; ++j)
        {
            fprintf(archivoPredicciones, "%.2f ", predicciones[i][j]);
        }
        fprintf(archivoPredicciones, "\n");
    }

    fclose(archivoPredicciones);
}

// Imprimimos los MAPE en un archivo de texto
void imprimirMAPE(double *mapes, int filas)
{
    FILE *archivoMAPE = fopen("MAPE.txt", "w");
    if (!archivoMAPE)
    {
        perror("Error al abrir el archivo de MAPE");
        exit(1);
    }

    // Escribimos los MAPE en el archivo
    for (int i = 0; i < filas; ++i)
    {
        fprintf(archivoMAPE, "%.2f\n", mapes[i]);
    }

    fclose(archivoMAPE);
}

// Realizamos el análisis de escalabilidad e imprimimos los resultados en un archivo
void analisisEscalabilidad(int numProcesos, int numHilos, double tiempo, double mapeTotal, char *nombreArchivo)
{
    FILE *archivoTiempo = fopen("Tiempo.txt", "w");
    if (!archivoTiempo)
    {
        perror("Error al abrir el archivo de tiempo");
        exit(1);
    }

    // Escribimos los resultados del análisis de escalabilidad en el archivo
    fprintf(archivoTiempo, "Tiempo de ejecución: %.2f segundos\n", tiempo);
    fprintf(archivoTiempo, "Archivo procesado: %s\n", nombreArchivo);
    fprintf(archivoTiempo, "MAPE total: %.2f\n", mapeTotal);
    fprintf(archivoTiempo, "Número de procesos utilizados: %d\n", numProcesos);
    fprintf(archivoTiempo, "Número de hilos por proceso: %d\n", numHilos);

    fclose(archivoTiempo);
}

// Predecir todas las filas para un conjunto de datos
void predecirTodasLasFilas(double **datos, int filas, int columnas, int k, double **predicciones)
{
    // Paralelizamos el bucle anidado con OpenMP
#pragma omp parallel for collapse(2)
    for (int i = 0; i < filas; ++i)
    {
        for (int j = 0; j < columnas; ++j)
        {
            predicciones[i][j] = predecir(datos, filas, columnas, k, j + 1, i);
        }
    }
}

int main(int argc, char **argv)
{
    // Inicializamos MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Verificamos los argumentos de línea de comandos
    if (argc != 5)
    {
        if (rank == 0)
        {
            printf("Uso: mpirun -np <num_procs> ./prediccion_series_temporales <K> <ruta_datos> <num_procesos> <num_hilos>\n");
        }
        MPI_Finalize();
        exit(1);
    }

    // Obtenemos los parámetros desde la línea de comandos
    int k = atoi(argv[1]);
    char *rutaDatos = argv[2];
    int numProcesos = atoi(argv[3]);
    int numHilos = atoi(argv[4]);

    int filas, columnas;
    double **datos = NULL;
    int filasPorProceso;
    int filasRestantes;

    // MASTERPID
    if (rank == 0)
    {
        FILE *archivo = fopen(rutaDatos, "r");
        if (!archivo)
        {
            perror("Error al abrir el archivo");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Leemos las dimensiones de la matriz
        fscanf(archivo, "%d %d", &filas, &columnas);

        // Cálculamos las filas a procesar por cada proceso
        filasPorProceso = filas / numProcesos;
        filasRestantes = filas % numProcesos;

        // Distribución de filasPorProceso y filasRestantes a todos los procesos
        MPI_Bcast(&filasPorProceso, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&filasRestantes, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Asignamos memoria para la matriz de datos
        datos = (double **)calloc(filas, sizeof(double *));
        if (!datos)
        {
            perror("Error al asignar memoria para datos");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Leemos los datos desde el archivo
        for (int i = 0; i < filas; i++)
        {
            datos[i] = (double *)calloc(columnas, sizeof(double));
            if (!datos[i])
            {
                perror("Error al asignar memoria para datos[i]");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            char line[1000];
            fgets(line, sizeof(line), archivo);

            char *matriz = strtok(line, ",");
            int j = 0;
            while (matriz != NULL)
            {
                datos[i][j] = atof(matriz);
                matriz = strtok(NULL, ",");
                j++;
            }
        }

        fclose(archivo);
    }

    // Distribución de filasPorProceso y filasRestantes a todos los procesos
    MPI_Bcast(&filasPorProceso, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&filasRestantes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Rango y filasPorProceso para manejar las filas restantes
    int inicio = rank * filasPorProceso;
    int fin = inicio + filasPorProceso;

    // Número de filas por proceso para manejar las filas restantes
    if (rank < filasRestantes)
    {
        filasPorProceso++;
        inicio += rank;
        fin = inicio + 1;
    }
    else
    {
        inicio += filasRestantes;
        fin += filasRestantes;
    }

    // Asignamos memoria para los datos locales de cada proceso
    double **datosLocal = (double **)calloc(filasPorProceso, sizeof(double *));
    if (!datosLocal)
    {
        perror("Error al asignar memoria para datosLocal");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < filasPorProceso; ++i)
    {
        datosLocal[i] = (double *)calloc(columnas, sizeof(double));
        if (!datosLocal[i])
        {
            perror("Error al asignar memoria para datosLocal[i]");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int j = 0; j < columnas; ++j)
        {
            datosLocal[i][j] = datos[inicio + i][j];
        }
    }

    // Asignamos memoria para las predicciones locales de cada proceso
    double **prediccionesLocal = (double **)calloc(filasPorProceso, sizeof(double *));
    if (!prediccionesLocal)
    {
        perror("Error al asignar memoria para prediccionesLocal");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < filasPorProceso; ++i)
    {
        prediccionesLocal[i] = (double *)calloc(columnas, sizeof(double));
        if (!prediccionesLocal[i])
        {
            perror("Error al asignar memoria para prediccionesLocal[i]");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Inicio del tiempo de ejecución
    double tiempoInicio = MPI_Wtime();

    // Predicción de todas las filas asignadas a cada proceso
    predecirTodasLasFilas(datosLocal, filasPorProceso, columnas, k, prediccionesLocal);

    // Barrera o sincronización entre procesos
    MPI_Barrier(MPI_COMM_WORLD);

    // Fin del tiempo de ejecución
    double tiempoFin = MPI_Wtime();
    double tiempoTotal = tiempoFin - tiempoInicio;

    // Calculamos el MAPE local 
    double mapeLocal = calcularMAPE(datosLocal, prediccionesLocal, filasPorProceso, columnas);
    // Asignamos memoria para los MAPE de todos los procesos
    double *mapes = (double *)calloc(numProcesos, sizeof(double));
    if (!mapes)
    {
        perror("Error al asignar memoria para mapes");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Recopilamos el MAPE local de cada proceso
    MPI_Gather(&mapeLocal, 1, MPI_DOUBLE, mapes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Cálculamos el MAPE total
    double mapeTotal = 0.0;
    for (int i = 0; i < numProcesos; ++i)
    {
        mapeTotal += mapes[i];
    }
    mapeTotal /= numProcesos;

    // Barrera o sincronización entre procesos
    MPI_Barrier(MPI_COMM_WORLD);

    // MASTERPID
    if (rank == 0)
    {
        analisisEscalabilidad(numProcesos, numHilos, tiempoTotal, mapeTotal, rutaDatos);
        imprimirPredicciones(prediccionesLocal, filasPorProceso, columnas);
        imprimirMAPE(mapes, numProcesos);
        printf("Tiempo de ejecución: %.2f segundos\n", tiempoTotal);
        printf("MAPE total: %.2f\n", mapeTotal);
        free(mapes);
    }

    // Liberamos memoria
    for (int i = 0; i < filasPorProceso; ++i)
    {
        free(datosLocal[i]);
        free(prediccionesLocal[i]);
    }

    free(datosLocal);
    free(prediccionesLocal);

    if (rank == 0)
    {
        for (int i = 0; i < filas; ++i)
        {
            free(datos[i]);
        }
        free(datos);

        // Llamamos al analizador de predicciones en forma de gráfico
        system("python3 analisisPredicciones.py");
    }


    // Finalizamos MPI
    MPI_Finalize();

    return 0;
}
