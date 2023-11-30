# DESDE MÓDULOS
from preprocesamiento import bases_de_datos as bbdd
from preprocesamiento import elastic as elastic
from modelado import funcs_modelado as opt

# GENERALES
import os
import numpy as np
import logging



# CONTROL EJECUCIÓN
BBDD = True
PREPROCESAR = False
MODELADO = True

# PATHS
DATOS_ORIGINALES = os.path.join("Datos", "Originales")
DATOS_FW = os.path.join(DATOS_ORIGINALES, "FW_logs", "log2.csv")
DATOS_WIFI = os.path.join(DATOS_ORIGINALES, "wifitraps_anonimo.log")
DATOS_NF = os.path.join(DATOS_ORIGINALES, "Network_flows", "Dataset-Unicauca-Version2-87Atts.csv")

DATOS_TRANSFORMADOS = os.path.join("Datos", "Transformados")
MODELOS = os.path.join('Modelos')
GRAFICOS = os.path.join('Graficos')
bbdd.crear_directorios()

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO, # Nivel de los mensajes que se registrarán
    filename='reto8_amarillo.log', # Nombre del archivo de log
    filemode='w', # Modo de apertura del archivo de log: read, write, append (write borra el contenido previo del archivo)
    format='%(asctime)s %(levelname)s %(message)s', # Formato de los mensajes de log
    datefmt='%d-%m-%Y %H:%M:%S' # Formato de fecha y hora
)



####### BASES DE DATOS #######

if BBDD:
    print("BASES DE DATOS")

    # MySQL
    compresores = bbdd.carga_compresores()
    credentials = bbdd.sql_connect()
    bbdd.sql_create_db(credentials)
    bbdd.sql_insert(compresores)
    logging.info("Datos insertados en SQL.")

    # MongoDB
    cliente = bbdd.mongo_connect()
    db, colec_regr, colec_opt = bbdd.mongo_create_db(cliente)
    logging.info("Base de datos MongoDB creada.")



####### PREPROCESAMIENTO #######

if PREPROCESAR:
    logging.info("PREPROCESAMIENTO")

    # Fichero Firewall
    print("Preprocesamiento del fichero del firewall:")
    df_fw = elastic.preprocesamiento_firewall(DATOS_FW)
    elastic.test_normalidad(df_fw)
    df_fw = elastic.creacion_variables(df_fw)
    df_fw.reset_index(inplace=True)
    df_fw.to_csv(os.path.join(DATOS_TRANSFORMADOS, 'df_firewall.csv'), index=False)

    # Fichero Wifi
    print("Preprocesamiento del fichero del wifi:")
    df_wifi = elastic.preprocesamiento_wifi(DATOS_WIFI)
    df_wifi = elastic.renombrar_columnas_wifi(df_wifi)
    df_wifi = elastic.limpieza_wifi(df_wifi)
    elastic.guardar_wifi(df_wifi)

    # Fichero Network Flows
    print("Preprocesamiento del fichero de network flows:")
    df_nf = elastic.preprocesamiento_nf(DATOS_NF)
    df_nf.to_csv(os.path.join(DATOS_TRANSFORMADOS, 'df_networkflow.csv'), index = False)

    print("FIN DEL PREPROCESAMIENTO, DATOS TRANSFORMADOS GUARDADOS")
    print('-'*80, '\n'*3, '-'*80)



####### MODELOS #######

if MODELADO:
    logging.info("MODELADO: Regresión y Optimización")


    # REGRESIÓN
    compresores = bbdd.carga_compresores()
    cliente = bbdd.mongo_connect()
    db, colec_regr, colec_opt = bbdd.mongo_create_db(cliente)
    opt.entrenar_modelos(compresores, colec_regr)
    logging.info("Modelos entrenados y guardados.")


    # OPTIMIZACIÓN

    N_INDIVIDUOS = 100
    N_GENERATIONS = 70
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.5

    PRODUCCION = np.array([100, 90, 95, 110])
    MIN_PRODUCCION = 250 # caudal mínimo requerido entre los 4 compresores para el algoritmo inicial

    # Carga de modelos ya entrenados y guardados con Pickle
    modelos = opt.cargar_modelos()

    # Main con selection-crossover-mutation
    best_individual, best_fitness, poblacion = opt.main_selection_xover_mutation(N_GENERATIONS, N_INDIVIDUOS, PRODUCCION, MIN_PRODUCCION, CROSSOVER_RATE, MUTATION_RATE, compresores, modelos, colec_opt)
    logging.info("Algoritmo genético 1/3 finalizado: selection-crossover-mutation")

    # Main con Differential Evolution
    best_individual, best_fitness, poblacion = opt.main_diff_evolution(N_GENERATIONS, N_INDIVIDUOS, PRODUCCION, MIN_PRODUCCION, compresores, modelos, colec_opt)
    logging.info("Algoritmo genético 2/3 finalizado: Differential Evolution")

    # Main con un nuevo caudal mínimo: nuevo algoritmo para ver cuántas modificaciones hay que realizar
    individuo_anterior = best_individual.copy()
    NUEVO_MIN_CAUDAL = 255
    opt.main_modificaciones(N_GENERATIONS, PRODUCCION, NUEVO_MIN_CAUDAL, poblacion, best_fitness, compresores, modelos, individuo_anterior, colec_opt)
    logging.info("Algoritmo genético 3/3 finalizado: nuevo caudal mínimo")

    # Queries sobre los resultados mediante MongoDB
    print('RESULTADOS: Queries sobre la colección de optimización de MongoDB\n')
    opt.queries_resultados(colec_opt)

    print("FIN DEL MODELADO")
    print('-'*80, '\n')

print("FIN DE LA EJECUCIÓN")