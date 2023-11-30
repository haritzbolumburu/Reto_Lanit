# GENERALES
import pandas as pd
import numpy as np
import random
from operator import itemgetter
from collections import Counter
import logging

# VISUALIZACIÓN
import matplotlib.pyplot as plt

# MODELADO
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor

# MongoDB
import pymongo



# REGRESIÓN

# Regression to predict Potencia_Medida with Catboost, cv=5 and scoring='neg_mean_squared_error'

def train_and_validate(compressor:str, compresores:pd.DataFrame) -> dict:
    """Regresión con Catboost para predecir la potencia medida de un compresor, con validación cruzada de 5 folds y siendo el scoring el neg_mean_squared_error.
    Primero, se seleccionan las variables que se van a emplear en el modelo, y se separan los datos en train y test.
    Después, se entrena el modelo con los datos de train y se valida con los datos de test.
    Se almacenan ambas métricas, así como las importancias de las variables. Se devuelve un diccionario con los indicadores más importantes.

    Args:
        compressor (str): Nombre del compresor
        compresores (pd.DataFrame): DataFrame con los datos de los compresores

    Returns:
        model_info (dict): Diccionario con los indicadores más importantes del modelo
    """
    feature_columns = ['Presion', 'Temperatura', 'Frecuencia']
    target_column = 'Potencia_Medida'

    data = compresores[compresores['compresor'] == compressor]
    
    X = data[feature_columns]
    y = data[target_column]

    # Separación de los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Entrenamiento del modelo
    model = CatBoostRegressor(verbose=1000)
    model.fit(X_train, y_train)

    # Validación del modelo con los datos de test
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print('R2 en test:', r2.round(4))
    error_traintest = mean_squared_error(y_test, y_pred)
    print('MSE en test:', error_traintest.round(4))

    # Comparar con el modelo baseline: Potencia_Estimada
    y_pred_baseline = data['Potencia_Estimada']
    r2_baseline = r2_score(y, y_pred_baseline)
    print('R2 baseline:', r2_baseline.round(4))
    error_baseline = mean_squared_error(y, y_pred_baseline)
    print('MSE baseline:', error_baseline.round(4))

    # Guardado del modelo y las métricas de validación
    pickle.dump(model, open(f'Modelos/compresor_{compressor}.pkl', 'wb'))
    print(f'Modelo del compresor {compressor} guardado en la carpeta Modelos.')
    feature_importances_dict = dict(zip(feature_columns, model.feature_importances_))
    model_info = {
        'model_params': model.get_params(),
        'feature_importances': feature_importances_dict,
        'metrics': {
            'r2': r2,
            'r2_baseline': r2_baseline,
            'error_traintest': error_traintest,
            'error_baseline': error_baseline,
        }
    }
    return model_info


# Se entrenan y validan los modelos sobre los datos de cada compresor y se guardan modelos y métricas en un diccionario
def entrenar_modelos(compresores:pd.DataFrame, colec_regr:pymongo.collection.Collection) -> dict:
    """Se entrenan y validan los modelos sobre los datos de cada compresor y se guardan modelos y métricas en un diccionario.
    Se insertan los documentos en la colección de MongoDB 'colec_regr'.

    Args:
        compresores (pd.DataFrame): DataFrame con los datos de los compresores

    Returns:
        modelos (dict): Diccionario con los modelos y las métricas de validación
    """
    modelos = {}
    for compressor in compresores['compresor'].unique():
        model_info = train_and_validate(compressor, compresores)
        model_info['compressor'] = compressor
        try:
            colec_regr.insert_one(model_info)
            print(f'Documento del compresor {compressor} insertado correctamente en MongoDB.')
        except pymongo.errors.DuplicateKeyError as error:
            print('Ya existe ese documento.')

    return None


# OPTIMIZACIÓN

def simulate_presion(compresores:pd.DataFrame) -> float:
    """Función que simula la presión de los compresores, siguiendo la misma distribución que la real.

    Args:
        compresores (pd.DataFrame): DataFrame con los datos de los compresores
    
    Returns:
        float: Presión simulada
    """
    mean = compresores['Presion'].mean()
    std = compresores['Presion'].std()
    return np.random.normal(mean, std)

def simulate_temperatura(compresores:pd.DataFrame) -> float:
    """Función que simula la temperatura de los compresores, siguiendo la misma distribución que la real.

    Args:
        compresores (pd.DataFrame): DataFrame con los datos de los compresores
    
    Returns:
        float: Temperatura simulada
    """
    mean = compresores['Temperatura'].mean()
    std = compresores['Temperatura'].std()
    return np.random.normal(mean, std)


def cargar_modelos():
    """Función que carga los modelos de los compresores con Pickle para poder emplearlos para predecir la potencia medida.

    Returns:
        modelos (dict): Diccionario con los modelos de los compresores ya cargados con Pickle.
    """
    modeloA = pickle.load(open(f'Modelos/compresor_CompA.pkl', 'rb'))
    modeloB = pickle.load(open(f'Modelos/compresor_CompB.pkl', 'rb'))
    modeloC = pickle.load(open(f'Modelos/compresor_CompC.pkl', 'rb'))
    modeloD = pickle.load(open(f'Modelos/compresor_CompD.pkl', 'rb'))

    modelos = {
        'CompA': modeloA,
        'CompB': modeloB,
        'CompC': modeloC,
        'CompD': modeloD
    }
    return modelos


def predecir_potencia(compresores:pd.DataFrame, modelos:dict, individual:list, compressor:str) -> float:
    """Función que predice la potencia medida de un compresor a partir de un modelo y un list con los valores de Presión, Temperatura y Frecuencia.

    Args:
        compresores (pd.DataFrame): DataFrame con los datos de los compresores
        compressor (str): Nombre del compresor que será cargado con Pickle
        individual (list): Lista con los valores de Presión, Temperatura y Frecuencia
        modelos (dict): Diccionario con los modelos de regresión de cada compresor

    Returns:
        pred (float): Valor de la potencia medida predicha (el consumo)
    """
    # Cargar modelos con pickle
    modelo = modelos[compressor]
    pres = simulate_presion(compresores)
    temp = simulate_temperatura(compresores)
    dict_relaciones = {'CompA': 0, 'CompB': 1, 'CompC': 2, 'CompD': 3}
    frec = individual[dict_relaciones[compressor]]

    # Predecir la potencia medida
    df_pred = pd.DataFrame({'Presion': pres, 'Temperatura': temp, 'Frecuencia': frec}, index=[0])
    pred = modelo.predict(df_pred)[0]
    return pred


# Población inicial

def poblacion_inicial(N_INDIVIDUOS:int, PRODUCCION:list, MIN_PRODUCCION:int, N_COMPRESORES:int=4) -> list:
    """Genera una población inicial de individuos con valores aleatorios entre 30 y 85 (evitando penalizaciones).
    En total deben tener una producción mínima de min_produccion, teniendo que cuenta que:
    produccion total = sumatorio de frecuencia*produccion de cada compresor

    Args:
        N_INDIVIDUOS (int): número de individuos de la población
        PRODUCCION (list): producción de cada compresor con una frecuencia del 100%
        MIN_PRODUCCION (int): producción mínima requerida
        N_COMPRESORES (int, optional): número de compresores. Por defecto son 4.

    Returns:
        poblacion (list): lista de individuos
    """
    poblacion = []
    while len(poblacion) < N_INDIVIDUOS:
        individuo = np.random.randint(30, 85, N_COMPRESORES)
        individuo = [float(i) for i in individuo]
        caudal = PRODUCCION * individuo/100
        if sum(caudal) >= MIN_PRODUCCION:
            poblacion.append(individuo)
    return poblacion


# Fitness: devuelve el consumo de la solución tras aplicar varias penalizaciones

def fitness(individual:list, min_produccion:int, PRODUCCION:list, compresores:pd.DataFrame, modelos:dict) -> float:
    """Función que calcula el fitness de un individuo, es decir, el consumo de la solución tras aplicar varias penalizaciones:
    1. Si frecuencia > 90 --> función de penalización exponencial estricta: consumo*max(1,e^((x-90)/20))
    2. Si frecuencia es 0 o se acerca --> función de penalización exponencial suave: consumo*max(1,e^((10-x)/40))
    3. Si caudal < min_produccion --> consumo *= 1 + ((min_produccion - caudal) / 50)
    Además, se comprueba si el individuo es válido, es decir, si la producción total es mayor o igual que la mínima requerida
    y si sus valores de frecuencia están entre 10 y 90.
    
    Args:
        individual (list): lista con los valores de frecuencia de cada compresor
        min_produccion (int): producción mínima requerida
        PRODUCCION (list): producción de cada compresor con una frecuencia del 100%
        compresores (pd.DataFrame): DataFrame con los datos de los compresores
        modelos (dict): Diccionario con los modelos de regresión de cada compresor

    Returns:
        consumo (float): consumo de la solución (fitness)
        feasible (bool): True si el individuo es válido, False si no lo es
        caudal (float): caudal total de la solución
    """
    compresores_tipos = ['CompA', 'CompB', 'CompC', 'CompD']
    consumo = np.array([predecir_potencia(compresores, modelos, individual=individual, compressor=c) for c in compresores_tipos])
    consumo = np.sum(consumo) # consumo total de la solución
    caudal = np.sum(individual * PRODUCCION / 100) # caudal total de la solución
    
    # Penalizaciones:
    # - Si frecuencia > 90 --> función de penalización exponencial estricta: consumo*max(1,e^((x-90)/20))
    for compr in individual:
        consumo *= max(1, np.exp((compr-90)/20))
    # - Si frecuencia es 0 o se acerca --> función de penalización exponencial suave: consumo*max(1,e^((10-x)/40))
    for compr in individual:
        consumo *= max(1, np.exp((10-compr)/40))

    # Si el caudal total es menor que el mínimo requerido, también se penaliza
    if caudal < min_produccion:
        consumo *= 1 + ((min_produccion - caudal) / 50) # con consumo=320, min_produccion=250 y produccion=245 --> consumo=320*1.05=352

    # Se crea una columna de si es feasible o no
    individual = np.array(individual)
    individual_feasible = np.logical_and(individual >= 10, individual <= 90).all()
    produccion_feasible = caudal >= min_produccion
    feasible = individual_feasible and produccion_feasible

    return consumo, feasible, caudal


# Seleccion de padres: se usará rank selection para seleccionar los padres ya que los valores de fitness son cercanos entre sí

def rank_population(population:list) -> list:
    """Ordena la población de menor a mayor fitness

    Args:
        population (list): población de individuos

    Returns:
        population (list): población de individuos una vez ordenada de menor a mayor fitness
    """
    return sorted(population, key=itemgetter(0))

def rank_selection(ranked_population:list) -> list:
    """Selecciona individuos para la reproducción basándose en su rango en la lista de la población ordenada por fitness.

    Args:
        ranked_population (list): población de individuos una vez ordenada de menor a mayor fitness

    Returns:
        list: individuo seleccionado (una lista de frecuencias)
    """
    # Asigna una ponderación de rango a cada individuo (cuanto menor sea el índice, mayor será la ponderación)
    for i in range(len(ranked_population)):
        ranked_population[i][0] = (len(ranked_population) - i) ** 2

    # Calcula las probabilidades teniendo en cuenta la ponderación del rango
    total = sum([ranked_population[i][0] for i in range(len(ranked_population))])
    probabilities = [ranked_population[i][0] / total for i in range(len(ranked_population))]

    # Calcula las probabilidades acumulativas
    cumulative_probabilities = np.cumsum(probabilities)

    # Genera un número aleatorio y selecciona el individuo correspondiente
    alpha = random.random()
    selected_index = np.where(alpha <= cumulative_probabilities)[0][0]

    return ranked_population[selected_index]


# Crossover: Arithmetic crossover

def crossover(parent1:list, parent2:list, CROSSOVER_RATE:float) -> list:
    """Realiza el crossover entre dos padres para obtener dos hijos.
    Este método de cruce genera hijos que son una combinación lineal de sus padres.
    Alpha es un número aleatorio entre 0 y 1 que decide cuánto de cada padre va a cada hijo.
    Si alpha es cercano a 0, el primer hijo será muy similar al segundo padre (y viceversa para el segundo hijo).
    Si alpha es cercano a 1, sucede lo contrario. Si alpha es 0.5, los hijos serán aproximadamente la media de los padres.
    
    Args:
        parent1 (list): lista de frecuencias del primer padre
        parent2 (list): lista de frecuencias del segundo padre
        CROSSOVER_RATE (float): probabilidad de que se produzca el crossover

    Returns:
        child1 (list): lista de frecuencias del primer hijo, diferente al padre o no (depende de CROSSOVER_RATE)
        child2 (list): lista de frecuencias del segundo hijo, diferente al padre o no (depende de CROSSOVER_RATE)
    """
    if random.random() < CROSSOVER_RATE:
        alpha = random.random()
        child1 = [alpha*a + (1-alpha)*b for a, b in zip(parent1, parent2)]
        child2 = [alpha*b + (1-alpha)*a for a, b in zip(parent1, parent2)]
        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


# Mutación: añadir una variación aleatoria a un gen

def mutation(individual:list, mutation_scale:float, MUTATION_RATE:float) -> list:
    """Mutación: añadir una variación aleatoria a un gen (una frecuencia de un compresor)

    Args:
        individual (list): individuo a mutar
        mutation_scale (float): es un parámetro que controla la magnitud de las mutaciones,
                el porcentaje de variación, y su rango se mueve entre 0 y 100.
                Por ejemplo, si los valores varían entre 0 y 100, se podría empezar con un
                mutation_scale de 1.5.
        MUTATION_RATE (float): probabilidad de que se produzca la mutación

    Returns:
        individual (list): individuo mutado
    """
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] += random.uniform(-mutation_scale, mutation_scale) # añade una variación aleatoria a un gen
    return individual


# Evolucionar la población

def evolve(poblacion:list, MIN_PRODUCCION:float, PRODUCCION:list, CROSSOVER_RATE:float, MUTATION_RATE:float, compresores:pd.DataFrame, modelos:dict) -> tuple:
    """Evoluciona la población, es decir, selecciona los padres, los cruza y los muta.
    Se ejecutará este proceso tantas veces como se defina en el parámetro N_GENERATIONS.

    Args:
        poblacion (list): población de individuos (lista de listas)
        MIN_PRODUCCION (float): caudal mínimo exigido al mejor individuo. Si no llega, se penaliza su fitness (consumo)
        PRODUCCION (list): producción de cada compresor con una frecuencia del 100%
        CROSSOVER_RATE (float): probabilidad de que se produzca el crossover
        MUTATION_RATE (float): probabilidad de que se produzca la mutación
        compresores (pd.DataFrame): tabla con los datos de los compresores
        modelos (dict): diccionario con los modelos de los compresores

    Returns:
        mejor_resultado (list): mejor individuo de la población, formato [fitness, [individuo]]
        new_population (list): población de individuos evolucionada (lista de listas)
        perc_feasibles (float): porcentaje de individuos factibles (que no sufren ninguna penalización)
        produccion_media (float): caudal medio generado por cada individuo de la población
        produccion_mejor_individuo (float): caudal generado por el mejor individuo de la población
        fitness_medio (float): fitness medio de la población
    """

    # El mutation_rate varía en función del porcentaje de copias
    porcentaje_copias = ((sum(count - 1 for count in Counter([tuple(array) for array in poblacion]).values() if count > 1)) / len(poblacion))
    MUTATION_RATE = min(1, MUTATION_RATE*(1 + porcentaje_copias)) # si 20% de copias y mut_rate=0.4, nuevo_mut_rate=0.48
    
    # Se calcula el fitness de cada individuo, se ordena la población, y se obtiene el mejor individuo y su fitness
    results = [fitness(individual, MIN_PRODUCCION, PRODUCCION, compresores, modelos) for individual in poblacion]
    population = [[results[i][0], poblacion[i]] for i in range(len(poblacion))]
    ranked_population = rank_population(population)
    best_result = ranked_population[0] # mejor individuo y mejor fitness
    mejor_resultado = best_result.copy()

    # Se guardan algunas métricas: porcentaje de individuos feasibles y producción media de los indivs de la población...
    perc_feasibles = [[results[i][1], poblacion[i]] for i in range(len(poblacion))]
    recuento = Counter(elem[0] for elem in perc_feasibles)
    perc_feasibles = (recuento[True]/(recuento[True]+recuento[False]))*100
    produccion_media = [[results[i][2], poblacion[i]] for i in range(len(poblacion))]
    produccion_media = np.mean([elem[0] for elem in produccion_media])
    produccion_mejor_individuo = np.sum(mejor_resultado[1] * PRODUCCION / 100)
    fitness_medio = np.mean([elem[0] for elem in ranked_population])

    # Se crea una nueva población: parent selection + crossover + mutation
    new_population = []
    elitism_rate = 0.2
    elitism_size = int(elitism_rate * len(population))

    # Conservar los individuos de élite de la generación anterior
    elite_individuals = [individual[1] for individual in ranked_population[:elitism_size]]
    new_population.extend(elite_individuals)

    # Generar el resto de la nueva población
    while len(new_population) < len(population):
        parent1, parent2 = [rank_selection(ranked_population)[1] for _ in range(2)]
        child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)

        child1 = mutation(child1, mutation_scale=1.5, MUTATION_RATE=MUTATION_RATE)
        child2 = mutation(child2, mutation_scale=1.5, MUTATION_RATE=MUTATION_RATE)

        new_population.append(child1)
        new_population.append(child2)

    return mejor_resultado, new_population, perc_feasibles, produccion_media, produccion_mejor_individuo, fitness_medio


# Differential Evolution: rand/2/bin
def rand2bin(population, F=0.4, CR=0.15): # 0<F<2, 0.2<CR<1
    """Aplica el algoritmo de evolución diferencial rand/2/bin a una población:
    Rand significa que los vectores de mutación se generan de forma aleatoria.
    El 2 se debe a que se usan 2 vectores de mutación para generar el vector de prueba (4 individuos en total).
    Bin significa que se usa un cruce binomial discreto para generar el vector de prueba, es decir,
    se elige cada gen del vector de prueba de forma aleatoria entre el vector de mutación y el vector objetivo.

    Args:
        population (list): List of lists, each inner list is an individual.
        F (float, optional): Mutation factor. Por defecto es 0.5, y debe tomar valores entre 0 y 2.
            Representa la magnitud de la mutación. Cuanto mayor sea, mayor será la diferencia entre "y" y "x".
        CR (float, optional): Crossover rate. Por defecto es 0.2, y debe tomar valores entre 0 y 1.
            Representa el número de veces en las que se elige v (la mutación) en vez de x (el original).
            Se suele poner bajo para que x salga más veces.

    Returns:
        y (array): Individuo mutado, con componentes de los vectores x y v.

    """
    # Se seleccionan 6 individuos random de la población
    sample = random.sample(population, 6)

    # Se ordenan los individuos de menor a mayor fitness (ya que es minimización)
    ranked_sample = rank_population(sample)

    # Se saca el vector v y después se hace el bin entre v y x sobre y
    indivs = np.array([indiv[1] for indiv in ranked_sample])
    x, r1, r2, r3, r4, r5 = indivs
    v = r1 + F*(r2-r3) + F*(r4-r5)

    U = [random.random() for i in range(4)]
    y = x.copy()
    for i in range(len(v)):
        if U[i] < CR:
            y[i] = v[i] # en y (individuo mutado) se sustituye el valor inicial (x) por el valor mutado (v)

    return y


# Evolucionar la población, esta vez con Differential Evolution

def evolve_diff_evol(poblacion:list, MIN_PRODUCCION:float, PRODUCCION:list, compresores:pd.DataFrame, modelos:dict) -> tuple:
    """Evoluciona la población, es decir, selecciona los padres, los cruza y los muta.
    Se ejecutará este proceso tantas veces como se defina en el parámetro N_GENERATIONS.

    Args:
        poblacion (list): población de individuos (lista de listas)
        MIN_PRODUCCION (float): caudal mínimo exigido al mejor individuo. Si no llega, se penaliza su fitness (consumo)
        PRODUCCION (list): producción de cada compresor con una frecuencia del 100%
        compresores (pd.DataFrame): tabla con los datos de los compresores
        modelos (dict): diccionario con los modelos de los compresores

    Returns:
        mejor_resultado (list): mejor individuo de la población, formato [fitness, [individuo]]
        new_population (list): población de individuos evolucionada (lista de listas)
        perc_feasibles (float): porcentaje de individuos factibles (que no sufren ninguna penalización)
        produccion_media (float): caudal medio generado por cada individuo de la población
        produccion_mejor_individuo (float): caudal generado por el mejor individuo de la población
        fitness_medio (float): fitness medio de la población
    """
    
    # Se calcula el fitness de cada individuo, se ordena la población, y se obtiene el mejor individuo y su fitness
    results = [fitness(individual, MIN_PRODUCCION, PRODUCCION, compresores, modelos) for individual in poblacion]
    population = [[results[i][0], poblacion[i]] for i in range(len(poblacion))]
    ranked_population = rank_population(population)
    best_result = ranked_population[0] # mejor individuo y mejor fitness
    mejor_resultado = best_result.copy()

    # Se guardan algunas métricas: porcentaje de individuos feasibles y producción media de los indivs de la población...
    perc_feasibles = [[results[i][1], poblacion[i]] for i in range(len(poblacion))]
    recuento = Counter(elem[0] for elem in perc_feasibles)
    perc_feasibles = (recuento[True]/(recuento[True]+recuento[False]))*100
    produccion_media = [[results[i][2], poblacion[i]] for i in range(len(poblacion))]
    produccion_media = np.mean([elem[0] for elem in produccion_media])
    produccion_mejor_individuo = np.sum(mejor_resultado[1] * PRODUCCION / 100)
    fitness_medio = np.mean([elem[0] for elem in ranked_population])

    # Se crea una nueva población: differential evolution
    new_population = []
    while len(new_population) < len(population):
        hijo = rand2bin(population)
        new_population.append(hijo)

    return mejor_resultado, new_population, perc_feasibles, produccion_media, produccion_mejor_individuo, fitness_medio


# Algoritmo memético de búsqueda local

def vecinos(individual:list, step:float=0.5) -> list:
    """Genera los vecinos de un individuo, es decir, los individuos que se obtienen al variar
    ligeramente (tanto como se fije en el parámetro step) cada uno de los genes del individuo original.

    Args:
        individual (list): individuo del que se obtendrán los vecinos.
        step (float, optional): cuánto se modifica cada gen del individuo original,
                                tanto sumando como restando al valor de entrada del individuo.
                                Por defecto es 0.5.

    Returns:
        vecinos (list): lista de individuos vecinos del individuo original.
    """
    vecinos = []
    for i in range(len(individual)):
        new_individual1 = individual.copy()
        new_individual1[i] += step
        vecinos.append(new_individual1)

        new_individual2 = individual.copy()
        new_individual2[i] -= step
        vecinos.append(new_individual2)
    vecinos.append(individual)
    return vecinos

def busqueda_local(best_fitness:float, best_individual:list, MIN_PRODUCCION:float, PRODUCCION:list, compresores:pd.DataFrame, modelos:dict, step:float=0.5) -> list:
    """Realiza una búsqueda local sobre el mejor individuo de la población, es decir,
    sobre el individuo con el mejor fitness.
    Se obtienen los vecinos de este individuo y se comprueba si alguno de ellos tiene un fitness mejor.
    Si es así, se actualiza el mejor individuo al vecino con el fitness más bajo.
    Si ninguno de los vecinos es mejor, se finaliza la búsqueda local y se devuelve el mejor individuo.

    Args:
        best_fitness (float): el mejor fitness de la población obtenido hasta el momento.
        best_individual (list): el mejor individuo de la población obtenido hasta el momento.
        MIN_PRODUCCION (float): caudal mínimo exigido al mejor individuo. Si no llega, se penaliza su fitness (consumo)
        PRODUCCION (list): caudal producido por cada compresor al 100% de frecuencia.
        compresores (pd.DataFrame): tabla con los datos de los compresores.
        modelos (dict): modelos de regresión para cada compresor ya cargados con Pickle previamente.
        step (float, optional): cuánto se modifica cada gen del individuo original,
                                tanto sumando como restando al valor de entrada del individuo.
                                Por defecto es 0.5.

    Returns:
        best_fitness (float): el mejor fitness de la población obtenido hasta el momento, tras la búsqueda local.
        best_individual (list): el mejor individuo de la población obtenido hasta el momento, tras la búsqueda local.
    """
    best_individual = [float(i) for i in best_individual]
    while True:
        vecindario = vecinos(best_individual, step)
        vecindario_fitness = [fitness(individuo, MIN_PRODUCCION, PRODUCCION, compresores, modelos)[0] for individuo in vecindario]

        # Comprueba si algún vecino tiene un fitness mejor
        if min(vecindario_fitness) < best_fitness:
            # Actualiza el mejor individuo al vecino con el fitness más bajo
            best_fitness = min(vecindario_fitness)
            best_individual = vecindario[np.argmin(vecindario_fitness)]
        else:
            # Si ninguno de los vecinos es mejor, termina la búsqueda local
            break

    return best_fitness, best_individual


# Visualización de resultados

def graficar_evolucion(generacion_optima:int, busquedas_locales:list, evol_fitness:list, evol_caudal:list,
                       evol_fitness_medio:list, evol_caudal_medio:list, evol_perc_feasibles:list, MIN_PRODUCCION:float,
                       categoria:str, evol_compresores_cambiados:list=None):
    """Muestra una gráfica con la evolución del fitness del mejor individuo, del porcentaje de individuos feasibles,
    del caudal medio de la población, del caudal del mejor individuo y del caudal medio de la población.
    Guarda el gráfico como un archivo jpg en la carpeta "Graficos".

    Args:
        generacion_optima (int): generación en la que se obtuvo el mejor individuo.
        busquedas_locales (list): lista con el fitness del mejor individuo en cada generación tras aplicarle una búsqueda local.
        evol_fitness (list): lista con el fitness del mejor individuo en cada generación.
        evol_caudal (list): lista con el caudal del mejor individuo en cada generación.
        evol_fitness_medio (list): lista con el fitness medio de la población en cada generación.
        evol_caudal_medio (list): lista con el caudal medio de la población en cada generación.
        evol_perc_feasibles (list): lista con el porcentaje de individuos feasibles en cada generación.
        MIN_PRODUCCION (float): caudal mínimo exigido al mejor individuo.
        categoria (str): nombre del algoritmo de optimización. Se utiliza para nombrar el archivo jpg que guarda el gráfico.
        evol_compresores_cambiados (None, optional): lista con el número de compresores cambiados en cada generación.
    """
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle('Optimización del consumo de aire: evolución de métricas')

    # Crear subplots en las posiciones (0, 0), (0, 1) y (1, 0)
    ax1 = plt.subplot(2, 2, 1)  # Fitness, búsquedas locales y generación óptima
    ax2 = plt.subplot(2, 2, 2)  # Caudal
    ax3 = plt.subplot(2, 2, 3)  # Porcentaje de individuos feasibles

    # Crear subplot en la posición (1, 1) sólo si evol_compresores_cambiados no es None
    if evol_compresores_cambiados is not None:
        ax4 = plt.subplot(2, 2, 4)  # Número de compresores cambiados

    # Fitness, búsquedas locales y generación óptima
    ax1.plot(evol_fitness, label='Fitness del mejor individuo (consumo)')
    ax1.plot(evol_fitness_medio, label='Fitness medio de la población', linestyle='--')
    ax1.set_title('Evolución del Fitness')
    ax1.set_xlabel('nº de generación')
    ax1.set_ylabel('Fitness')
    ax1.axvline(x=generacion_optima, color='black', linestyle='-', label=f'Generación óptima: {generacion_optima}')
    ax1.legend(fontsize='small')
    for busqueda in busquedas_locales:
        color = 'g' if busqueda[1] else 'r'
        ax1.axvline(x=busqueda[0], color=color, linestyle='--', label=f'Generación {busqueda[0]}: {"éxito" if busqueda[1] else "fracaso"}')

    # Caudal
    ax2.plot(evol_caudal, label='Caudal del mejor individuo')
    ax2.plot(evol_caudal_medio, label='Caudal medio de la población', linestyle='--')
    ax2.axhline(y=MIN_PRODUCCION, color='black', linestyle='-', label=f'Caudal mínimo exigido: {MIN_PRODUCCION}')
    ax2.set_title('Evolución del Caudal')
    ax2.set_xlabel('nº de generación')
    ax2.set_ylabel('Caudal')
    ax2.legend(fontsize='small')

    # Porcentaje de individuos feasibles
    ax3.plot(evol_perc_feasibles, label='Porcentaje de individuos feasibles')
    ax3.set_title('Evolución del porcentaje de individuos feasibles')
    ax3.set_xlabel('nº de generación')
    ax3.set_ylabel('Porcentaje de individuos feasibles')
    ax3.legend(fontsize='small')

    # Compresores cambiados
    if evol_compresores_cambiados is not None:
        ax4.plot(evol_compresores_cambiados, label='nº de compresores cambiados', linestyle='dotted')
        ax4.axvline(x=generacion_optima, color='black', linestyle='-', label=f'Generación óptima: {generacion_optima}')
        ax4.set_title('Evolución del nº de compresores cambiados')
        ax4.set_xlabel('nº de generación')
        ax4.set_ylabel('nº de compresores cambiados')
        ax4.legend(fontsize='small')

    plt.savefig(f'Graficos/OPTIMIZACION_{categoria}.jpg')

    return None


# Almacenamiento de parámetros, métricas y resultados en MongoDB

def indexar_resultados_mongo(categoria: str, generacion_optima: int, N_INDIVIDUOS: int, N_GENERATIONS: int, MUTATION_RATE: float,
                             CROSSOVER_RATE: float, MIN_PRODUCCION: float, best_individual: list, best_fitness: float,
                             caudal: float, fitness_medio: float, caudal_medio: float, perc_feasible: float,
                             colec_opt: pymongo.collection.Collection):
    """
    Guarda los resultados de la optimización en una colección de MongoDB.
    Primero crea un diccionario con los resultados y luego lo inserta en la colección.
    Si hubiera algún error al insertar el documento debido a duplicidad de la clave, se muestra un mensaje de error
    sin parar la ejecución del programa.

    Args:
        categoria (str): La categoría del documento.
        generacion_optima (int): La generación en la cual se ha obtenido el mejor individuo.
        N_INDIVIDUOS (int): Número de individuos en la población.
        N_GENERATIONS (int): Número de generaciones a realizar.
        MUTATION_RATE (float): Tasa de mutación.
        CROSSOVER_RATE (float): Tasa de cruce.
        MIN_PRODUCCION (float): Caudal mínimo exigido.
        best_individual (list): El mejor individuo de la generación.
        best_fitness (float): El mejor fitness de la generación.
        caudal (float): Caudal del mejor individuo.
        fitness_medio (float): Fitness medio de la población.
        caudal_medio (float): Caudal medio de la población.
        perc_feasible (float): Porcentaje de individuos factibles.
        colec_opt (MongoDB Collection): Colección MongoDB donde se almacenarán los documentos.

    Returns:
        None
    """
    dict_opt = {
        'categoria': categoria,
        'params': {
            'pop_size': N_INDIVIDUOS,
            'n_generations': N_GENERATIONS,
            'mutation_rate': MUTATION_RATE,
            'crossover_rate': CROSSOVER_RATE,
            'min_produccion': MIN_PRODUCCION
        },
        'results': {
            'mejor_resultado': {
                'individuo': best_individual,
                'fitness': best_fitness,
                'caudal': caudal,
                'generacion': generacion_optima
            },
            'fitness_medio': fitness_medio,
            'caudal_medio': caudal_medio,
            'perc_feasible': perc_feasible
        }
    }

    # Insertar en la colección de optimización de MongoDB
    try:
        colec_opt.insert_one(dict_opt)
    except pymongo.errors.DuplicateKeyError as error:
        print('Ya existe ese documento.')

    return None


# Función principal con selection-crossover-mutation

def main_selection_xover_mutation(N_GENERATIONS:int, N_INDIVIDUOS:int, PRODUCCION:np.ndarray, MIN_PRODUCCION:float,
                                  CROSSOVER_RATE:float, MUTATION_RATE:float, compresores:pd.DataFrame, modelos:list,
                                  colec_opt:pymongo.collection.Collection):
    """
    Realiza la selección, cruce y mutación de una población en varias generaciones y visualiza los resultados.

    Args:
        N_GENERATIONS (int): Número de generaciones a realizar.
        N_INDIVIDUOS (int): Número de individuos en la población.
        PRODUCCION (numpy.ndarray): Vector con la producción de cada compresor.
        MIN_PRODUCCION (float): Caudal mínimo exigido.
        CROSSOVER_RATE (float): Tasa de cruce, en un rango [0, 1].
        MUTATION_RATE (float): Tasa de mutación, en un rango [0, 1].
        compresores (pd.DataFrame): DataFrame con los datos de los compresores.
        modelos (list): Lista con los modelos de regresión ya cargados con Pickle.
        colec_opt (MongoDB Collection): Colección MongoDB donde se almacenarán los documentos.

    Returns:
        best_individual (list): El mejor individuo logrado.
        best_fitness (float): El fitness del mejor individuo logrado.
        poblacion (list): Lista con los individuos de la última generación.
    """

    # Inicializar población inicial semi-aleatoria
    poblacion = poblacion_inicial(N_INDIVIDUOS, PRODUCCION, MIN_PRODUCCION)

    # Inicializar resultados finales
    best_fitness = np.inf
    evol_fitness = []
    evol_fitness_medio = []
    evol_caudal = []
    evol_caudal_medio = []
    evol_perc_feasibles = []
    busquedas_locales = []

    # Evolucionar población
    for gen in range(N_GENERATIONS):
        mejor_resultado, poblacion, perc_feasibles, caudal_medio, caudal, fitness_medio = evolve(poblacion, MIN_PRODUCCION, PRODUCCION, CROSSOVER_RATE, MUTATION_RATE, compresores, modelos)
        evol_fitness.append(mejor_resultado[0])
        evol_fitness_medio.append(fitness_medio)
        evol_caudal_medio.append(caudal_medio)
        evol_perc_feasibles.append(perc_feasibles)

        if mejor_resultado[0] < best_fitness:
            best_fitness = mejor_resultado[0]
            best_individual = mejor_resultado[1]

            # Algoritmo memético: búsqueda local
            best_fitness, best_individual = busqueda_local(best_fitness, best_individual, MIN_PRODUCCION, PRODUCCION, compresores, modelos, step=0.5)
            if best_fitness < mejor_resultado[0]:
                print(f'Ha mejorado con la búsqueda local de {mejor_resultado[0]} a {best_fitness}')
                evol_fitness[-1] = best_fitness
                busquedas_locales.append((gen, True))
            else:
                print('No ha mejorado con la búsqueda local')
                busquedas_locales.append((gen, False))
        caudal = best_individual @ PRODUCCION / 100
        evol_caudal.append(caudal)

        if gen % 1 == 0:
            print(f'Generación {gen+1}: Mejor individuo: {best_individual}, Mejor fitness: {best_fitness}, Caudal del mejor: {caudal}, Fitness medio: {fitness_medio}, Caudal medio: {caudal_medio}, Feasibles (%): {perc_feasibles}')
            logging.info(f'Generacion {gen+1}: Mejor individuo: {best_individual}, Mejor fitness: {best_fitness}, Caudal del mejor: {caudal}, Fitness medio: {fitness_medio}, Caudal medio: {caudal_medio}, Feasibles (%): {perc_feasibles}')

    print('\n', '-'*10, 'SOLUCIÓN FINAL', '-'*10)
    print(f'Individuo: {best_individual}, Fitness: {best_fitness}, Caudal: {caudal}')
    logging.info(f'Individuo: {best_individual}, Fitness: {best_fitness}, Caudal: {caudal}')

    generacion_optima = np.argmin(evol_fitness)

    graficar_evolucion(generacion_optima, busquedas_locales, evol_fitness, evol_caudal, evol_fitness_medio, evol_caudal_medio, evol_perc_feasibles, MIN_PRODUCCION, categoria='selection_xover_mutation')

    indexar_resultados_mongo('selection_xover_mutation', int(generacion_optima), N_INDIVIDUOS, N_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, MIN_PRODUCCION, best_individual, best_fitness, caudal, fitness_medio, caudal_medio, perc_feasibles, colec_opt)

    return best_individual, best_fitness, poblacion


# Función principal con Differential Evolution

def main_diff_evolution(N_GENERATIONS:int, N_INDIVIDUOS:int, PRODUCCION:np.ndarray, MIN_PRODUCCION:float,
                        compresores:pd.DataFrame, modelos:list, colec_opt:pymongo.collection.Collection):
    """
    En lugar de realizar la selección, cruce y mutación de una población en varias generaciones, se utiliza el algoritmo
    rand/2/bin de Differential Evolution para optimizar la producción de los compresores.

    Args:
        N_GENERATIONS (int): Número de generaciones a realizar.
        N_INDIVIDUOS (int): Número de individuos en la población.
        PRODUCCION (numpy.ndarray): Vector con la producción de cada compresor.
        MIN_PRODUCCION (float): Caudal mínimo exigido.
        compresores (pd.DataFrame): DataFrame con los datos de los compresores.
        modelos (list): Lista con los modelos de regresión ya cargados con Pickle.
        colec_opt (pymongo.collection.Collection): Colección de MongoDB donde se guardarán los resultados.

    Returns:
        best_individual (list): El mejor individuo logrado.
        best_fitness (float): El mejor fitness logrado.
        poblacion (list): Lista con los individuos de la última generación.
    """

    # Inicializar población inicial semi-aleatoria
    poblacion = poblacion_inicial(N_INDIVIDUOS, PRODUCCION, MIN_PRODUCCION)

    # Inicializar resultados finales
    best_fitness = np.inf
    evol_fitness = []
    evol_fitness_medio = []
    evol_caudal = []
    evol_caudal_medio = []
    evol_perc_feasibles = []
    busquedas_locales = []

    # Evolucionar población
    for gen in range(N_GENERATIONS):
        mejor_resultado, poblacion, perc_feasibles, caudal_medio, caudal, fitness_medio = evolve_diff_evol(poblacion, MIN_PRODUCCION, PRODUCCION, compresores, modelos)
        evol_fitness.append(mejor_resultado[0])
        evol_fitness_medio.append(fitness_medio)
        evol_caudal_medio.append(caudal_medio)
        evol_perc_feasibles.append(perc_feasibles)

        if mejor_resultado[0] < best_fitness:
            best_fitness = mejor_resultado[0]
            best_individual = mejor_resultado[1]

            # Algoritmo memético: búsqueda local
            best_fitness, best_individual = busqueda_local(best_fitness, best_individual, MIN_PRODUCCION, PRODUCCION, compresores, modelos, step=0.5)
            if best_fitness < mejor_resultado[0]:
                print(f'Ha mejorado con la búsqueda local de {mejor_resultado[0]} a {best_fitness}')
                evol_fitness[-1] = best_fitness
                busquedas_locales.append((gen, True))
            else:
                print('No ha mejorado con la búsqueda local')
                busquedas_locales.append((gen, False))
        caudal = best_individual @ PRODUCCION / 100
        evol_caudal.append(caudal)

        if gen % 1 == 0:
            print(f'Generación {gen+1}: Mejor individuo: {best_individual}, Mejor fitness: {best_fitness}, Caudal del mejor: {caudal}, Fitness medio: {fitness_medio}, Caudal medio: {caudal_medio}, Feasibles (%): {perc_feasibles}')
            logging.info(f'Generacion {gen+1}: Mejor individuo: {best_individual}, Mejor fitness: {best_fitness}, Caudal del mejor: {caudal}, Fitness medio: {fitness_medio}, Caudal medio: {caudal_medio}, Feasibles (%): {perc_feasibles}')

    print('\n', '-'*10, 'SOLUCIÓN FINAL', '-'*10)
    print(f'Individuo: {best_individual}, Fitness: {best_fitness}, Caudal: {caudal}')
    logging.info(f'Individuo: {best_individual}, Fitness: {best_fitness}, Caudal: {caudal}')

    generacion_optima = np.argmin(evol_fitness)

    graficar_evolucion(generacion_optima, busquedas_locales, evol_fitness, evol_caudal, evol_fitness_medio, evol_caudal_medio, evol_perc_feasibles, MIN_PRODUCCION, categoria='diff_evolution')

    MUTATION_RATE, CROSSOVER_RATE = None, None
    indexar_resultados_mongo('diff_evolution', int(generacion_optima), N_INDIVIDUOS, N_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, MIN_PRODUCCION, best_individual, best_fitness, caudal, fitness_medio, caudal_medio, perc_feasibles, colec_opt)

    return best_individual, best_fitness, poblacion


# Modificación en el caudal mínimo exigido: cambios necesarios en el individu para la adaptación

def modificaciones(individuo_anterior:list, mejor_resultado:tuple, caudal:float, NUEVO_MIN_CAUDAL:float):
    """Función que calcula los cambios necesarios en el individuo anterior para que se adapte al nuevo caudal mínimo

    Args:
        individuo_anterior (list): El mejor individuo logrado en el algoritmo anterior.
        mejor_resultado (tuple): Tupla con el fitness y el individuo de la mejor solución, en ese orden.
        caudal (float): Caudal mínimo exigido a los 4 compresores.
        NUEVO_MIN_CAUDAL (float): Nuevo caudal mínimo exigido a los 4 compresores.

    Returns:
        consumo (float): El consumo una vez se han añadido las penalizaciones por los cambios realizados.
        n_compresores_cambiados (int): Número de compresores que han cambiado de estado, entre 0 y 4.
    """
    # Penalizaciones por consumo
    consumo, individuo = mejor_resultado
    n_compresores_cambiados = len([i for i in range(len(individuo_anterior)) if round(individuo_anterior[i],2) != round(individuo[i],2)])
    consumo += 3*n_compresores_cambiados

    # Penalización extra por no cumplir el caudal mínimo
    deficit_caudal = NUEVO_MIN_CAUDAL - caudal
    if deficit_caudal > 0:
        if deficit_caudal <= 5: # Si el déficit de caudal es de 5 o menos
            consumo *= 1.05 # Aumenta el consumo en un 5%
        elif deficit_caudal <= 10: # Si el déficit de caudal es entre 5 y 10
            consumo *= 1.1 # Aumenta el consumo en un 10%
        else: # Si el déficit de caudal es mayor a 10
            consumo *= 1 + (deficit_caudal / 10) 

    return consumo, n_compresores_cambiados


# Cambio del caudal mínimo exigido y ejecución del algoritmo de nuevo para observar cuántos cambios son necesarios

def main_modificaciones(N_GENERATIONS:int, PRODUCCION:np.ndarray, NUEVO_MIN_CAUDAL:float,
                        poblacion:np.ndarray, best_fitness:float, compresores:pd.DataFrame, modelos:list,
                        individuo_anterior:list, colec_opt:pymongo.collection.Collection):
    """
    Vuelve a ejecutar el algoritmo de Differential Evolution, pero esta vez con un nuevo caudal mínimo exigido y
    partiendo de la población final y el mejor individuo logrados en el algoritmo anterior.
    Incluye un añadido a la función de fitness para penalizar los cambios realizados en el individuo (función "modificaciones").

    La población inicial será la población final de la ejecución anterior: "poblacion".
    Y el fitness a superar será el mejor fitness de la ejecución anterior: "best_fitness".

    Args:
        N_GENERATIONS (int): Número de generaciones a realizar.
        PRODUCCION (numpy.ndarray): Vector con la producción de cada compresor.
        NUEVO_MIN_CAUDAL (float): Nuevo caudal mínimo exigido.
        poblacion (numpy.ndarray): Población inicial.
        best_fitness (float): Mejor fitness alcanzado hasta el momento.
        compresores (pd.DataFrame): DataFrame con los datos de los compresores.
        modelos (list): Lista con los modelos de regresión ya cargados con Pickle.
        individuo_anterior (list): Mejor individuo logrado en el algoritmo anterior, id est, el de menor fitness.

    Returns:
        None
    """
    caudal_anterior = individuo_anterior @ PRODUCCION / 100
    if caudal_anterior > NUEVO_MIN_CAUDAL:
        return 'Se puede aumentar la producción sin modificar ningún compresor.'

    # Inicializar resultados finales
    evol_fitness = []
    evol_fitness_medio = []
    evol_caudal = []
    evol_caudal_medio = []
    evol_perc_feasibles = []
    busquedas_locales = []
    best_individual = individuo_anterior.copy()

    # Penalizar individuo anterior porque ya no llega al caudal mínimo
    best_fitness, n_cambios = modificaciones(individuo_anterior, (best_fitness, individuo_anterior), caudal_anterior, NUEVO_MIN_CAUDAL)
    evol_compresores_cambiados = [n_cambios]

    # Evolucionar población
    for gen in range(N_GENERATIONS):
        mejor_resultado, poblacion, perc_feasibles, caudal_medio, caudal, fitness_medio = evolve_diff_evol(poblacion, NUEVO_MIN_CAUDAL, PRODUCCION, compresores, modelos)
        evol_fitness.append(mejor_resultado[0])
        evol_fitness_medio.append(fitness_medio)
        evol_caudal_medio.append(caudal_medio)
        evol_perc_feasibles.append(perc_feasibles)

        mejor_resultado[0], n_cambios = modificaciones(individuo_anterior, mejor_resultado, caudal, NUEVO_MIN_CAUDAL)
        evol_compresores_cambiados.append(n_cambios)

        if mejor_resultado[0] < best_fitness:
            best_fitness = mejor_resultado[0]
            best_individual = mejor_resultado[1]

            # Algoritmo memético: búsqueda local
            best_fitness, best_individual = busqueda_local(best_fitness, best_individual, NUEVO_MIN_CAUDAL, PRODUCCION, compresores, modelos, step=0.5)
            if best_fitness < mejor_resultado[0]:
                print(f'Ha mejorado con la búsqueda local de {mejor_resultado[0]} a {best_fitness}')
                evol_fitness[-1] = best_fitness
                busquedas_locales.append((gen, True))
            else:
                print('No ha mejorado con la búsqueda local')
                busquedas_locales.append((gen, False))
        caudal = best_individual @ PRODUCCION / 100
        evol_caudal.append(caudal)

        if gen % 1 == 0:
            print(f'Generación {gen+1}: Mejor individuo: {best_individual}, Mejor fitness: {best_fitness}, Caudal del mejor: {caudal}, Fitness medio: {fitness_medio}, Caudal medio: {caudal_medio}, Feasibles (%): {perc_feasibles}')
            logging.info(f'Generacion {gen+1}: Mejor individuo: {best_individual}, Mejor fitness: {best_fitness}, Caudal del mejor: {caudal}, Fitness medio: {fitness_medio}, Caudal medio: {caudal_medio}, Feasibles (%): {perc_feasibles}')

    print('\n', '-'*10, 'SOLUCIÓN FINAL', '-'*10)
    print(f'Individuo: {best_individual}, Fitness: {best_fitness}, Caudal: {caudal}')
    logging.info(f'Individuo: {best_individual}, Fitness: {best_fitness}, Caudal: {caudal}')

    generacion_optima = np.argmin(evol_fitness)

    graficar_evolucion(generacion_optima, busquedas_locales, evol_fitness, evol_caudal, evol_fitness_medio, evol_caudal_medio, evol_perc_feasibles, NUEVO_MIN_CAUDAL, categoria='post_modificaciones', evol_compresores_cambiados=evol_compresores_cambiados)

    N_INDIVIDUOS, MUTATION_RATE, CROSSOVER_RATE = 100, None, None
    indexar_resultados_mongo('post_modificaciones', int(generacion_optima), N_INDIVIDUOS, N_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE, NUEVO_MIN_CAUDAL, best_individual, best_fitness, caudal, fitness_medio, caudal_medio, perc_feasibles, colec_opt)

    return None


# Queries sobre los resultados mediante MongoDB

def queries_resultados(colec_opt:pymongo.collection.Collection):
    """
    Realiza queries sobre los resultados de la optimización almacenados en MongoDB.
    Primero, obtiene el mejor fitness y el mejor individuo de la categoría "post_modificaciones".
    Después, compara resultados de diferentes algoritmos e incluso compara las soluciones óptimas logradas.

    Args:
        colec_opt (pymongo.collection.Collection): Colección de MongoDB donde se han almacenado los resultados de la optimización.
    
    Returns:
        None
    """
    # Query 1: Obtener el mejor fitness del algoritmo de optimización de la categoría "post_modificaciones"
    query1 = colec_opt.find_one({'categoria': 'post_modificaciones'}, {'_id': 0, 'results.mejor_resultado.fitness': 1})
    print(f'El MEJOR fitness para la categoría "post_modificaciones" ha sido: {query1["results"]["mejor_resultado"]["fitness"]}\n')

    # Query 2: Obtener el mejor individuo del algoritmo de optimización de la categoría "post_modificaciones"
    query2 = colec_opt.find_one({'categoria': 'post_modificaciones'}, {'_id': 0, 'results.mejor_resultado.individuo': 1})
    print(f'El MEJOR individuo para la categoría "post_modificaciones" tiene las siguientes frecuencias: {query2["results"]["mejor_resultado"]["individuo"]}\n')

    # Query 3: Comparar los resultados de fitness medios de las dos opciones de algoritmos
    query3 = list(colec_opt.aggregate([
        {"$match": {"$or": [{"categoria": "selection_xover_mutation"}, {"categoria": "diff_evolution"}]}},
        {"$project": {"categoria": 1, "fitness_medio": "$results.fitness_medio"}}
    ]))
    print(f'Los resultados de fitness MEDIOS de las dos opciones de algoritmos son: \n{pd.DataFrame(query3)[["categoria", "fitness_medio"]]}\n')

    # Query 4: Obtener el algoritmo con más porcentaje de individuos factibles
    query4 = colec_opt.find({}, {"_id": 0, "categoria": 1, "results.perc_feasible": 1}).sort("results.perc_feasible", -1).limit(1)
    query4 = list(query4)[0]
    print(f'El algoritmo con más porcentaje de individuos factibles es el {query4["categoria"]}, con un {query4["results"]["perc_feasible"]}% de individuos factibles\n')

    # Query 5: Comparar cuántos compresores cambian entre las mejores soluciones de las dos opciones de algoritmos
    option_2 = colec_opt.find_one({"categoria": "diff_evolution"}, {"_id": 0, "results.mejor_resultado.individuo": 1})
    option_2_individuo = option_2.get("results", {}).get("mejor_resultado", {}).get("individuo", [])

    option_3 = colec_opt.find_one({"categoria": "post_modificaciones"}, {"_id": 0, "results.mejor_resultado.individuo": 1})
    option_3_individuo = option_3.get("results", {}).get("mejor_resultado", {}).get("individuo", [])

    n_cambios = sum(i != j for i, j in zip(option_2_individuo, option_3_individuo))
    print(f'{n_cambios} compresores cambian al aumentar el caudal mínimo exigido.')

    return None