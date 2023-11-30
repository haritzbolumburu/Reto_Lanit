# GENERALES
import pandas as pd
import os

# MySQL
import json
import mysql.connector
from sqlalchemy import create_engine

# MONGODB
import pymongo


RANDOM_STATE = 42


def crear_directorios():
    """Función que crea los directorios donde se guardarán los datos transformados, los modelos y los gráficos.
    Si estos directorios ya existen, no se crea nada.

    Returns:
        None
    """
    if not os.path.exists('Datos/Transformados'):
        os.makedirs('Datos/Transformados')
    if not os.path.exists('Modelos'):
        os.makedirs('Modelos')
    if not os.path.exists('Graficos'):
        os.makedirs('Graficos')

    return None


# Carga de los ficheros con un bucle for en una funcion
def carga_compresores(ruta='Datos/Originales/Compresores'):
    """Carga de los ficheros con un bucle for en una funcion

    Args:
        ruta (str, optional): Ruta donde se encuentran los ficheros. Defaults to 'Datos/Originales/Compresores'.

    Returns:
        compresores: dataframe con todos los datos de los compresores, incluyendo una columna que indica el compresor
    """
    compresores = pd.DataFrame()
    for file in os.listdir(ruta):
        if file.endswith(".csv"):
            compresor = pd.read_csv(os.path.join(ruta, file), sep=',', decimal='.')
            compresor['compresor'] = file.split('.')[0]
            compresores = pd.concat([compresores, compresor], axis=0)
    return compresores


def sql_connect():
    """Función que devuelve las credenciales para acceder a la base de datos SQL.

    Args:
        None

    Returns:
        credentials (dict): Diccionario con las credenciales de la base de datos SQL: usuario, contraseña y ruta.
    """
    keyfile = '.pwd.json'
    with open(keyfile, 'r') as f:
        creds = json.load(f)

    credentials = {"usuario": creds.get("usuario"), "contrasena": creds.get("contrasena_sql"), "ruta": creds.get("ruta")}

    return credentials


def sql_create_db(credentials:dict):
    """Función que crea la base de datos en SQL.

    Args:
        credentials (dict): Diccionario con las credenciales de la base de datos SQL: usuario, contraseña y ruta.
    
    Returns:
        None
    """
    conn = mysql.connector.connect(user=credentials.get("usuario"), password=credentials.get("contrasena"), host=credentials.get("ruta"))
    cur = conn.cursor()
    try:
        cur.execute("DROP DATABASE IF EXISTS reto8")
        cur.execute('''CREATE DATABASE IF NOT EXISTS reto8''')
        cur.execute('''USE reto8''')
        cur.execute('''CREATE TABLE IF NOT EXISTS compresores(
                            Presion FLOAT NOT NULL,
                            Temperatura FLOAT NOT NULL,
                            Frecuencia FLOAT NOT NULL,
                            Potencia_Medida FLOAT NOT NULL,
                            Potencia_Estimada FLOAT NOT NULL,
                            compresor VARCHAR(5)
                        )''')
        conn.commit()
    except mysql.connector.Error as err:
        print(err)
        conn.rollback()
    finally:
        conn.close()
    return None


def sql_insert(df:pd.DataFrame):
    """Función que inserta los datos en la base de datos SQL.

    Args:
        df (pd.DataFrame): El dataframe con los datos a insertar

    Returns:
        None
    """
    engine = create_engine("mysql+mysqlconnector://root:1234@localhost/reto8")
    conn = engine.connect()
    df.to_sql(con=conn, name='compresores', if_exists='replace', chunksize=10000, index=False)
    conn.close()
    return None


def sql_consulta(consulta:str, credentials:dict):
    """Función que devuelve el resultado de una consulta SQL

    Args:
        consulta (str): La consulta SQL, en sintaxis de SQL y entre comillas en un string
        credentials (dict): Diccionario con las credenciales de la base de datos SQL: usuario, contraseña y ruta.

    Returns:
        resultado (pd.DataFrame): El resultado de la consulta
    """
    conn = mysql.connector.connect(user=credentials.get("usuario"), password=credentials.get("contrasena"), host=credentials.get("ruta"), database = 'reto8')
    resultado = pd.read_sql(consulta, conn)
    conn.close()
    return resultado


def sql_duplicados(credentials:dict):
    """Función que devuelve los duplicados de la base de datos SQL

    Args:
        credentials (dict): Diccionario con las credenciales de la base de datos SQL: usuario, contraseña y ruta.

    Returns:
        None
    """
    conn = mysql.connector.connect(user=credentials.get("usuario"), password=credentials.get("contrasena"), host=credentials.get("ruta"), database = 'reto8')
    cur = conn.cursor()

    try:
        query = f'''
        SELECT Presion, Temperatura, Frecuencia, Potencia_Medida, Potencia_Estimada, compresor, COUNT(*) AS num_ocurrencias
        FROM compresores
        GROUP BY Presion, Temperatura, Frecuencia, Potencia_Medida, Potencia_Estimada, compresor
        HAVING COUNT(*) > 1
        '''
        cur.execute(query)
        resultados = cur.fetchall()

        if len(resultados) > 0:
            print("Se encontraron duplicados:")
            for fila in resultados:
                print(fila)  
            print(f'Hay {len(resultados)} duplicados.')
        else:
            print("No se encontraron duplicados")

    except mysql.connector.Error as err:
        print(err)
    finally:
        conn.close()
    return None


# Conexión con el cliente
def mongo_connect():
    """Función que devuelve el cliente de MongoDB.

    Args:
        None
    
    Returns:
        cliente: cliente de MongoDB
    """
    try:
        cliente = pymongo.MongoClient("mongodb://localhost:27017/")
    except pymongo.errors.ConnectionFailure as error:
        print("Fallo al conectar a MongoDB:", error)
    return cliente


# Crear base de datos y colecciones
def mongo_create_db(cliente):
    """Función que crea la base de datos y las colecciones en MongoDB.

    Args:
        cliente (pymongo.MongoClient): cliente de MongoDB

    Returns:
        db (pymongo.database.Database): base de datos
        colec_regr (pymongo.collection.Collection): colección de regresión
        colec_opt (pymongo.collection.Collection): colección de optimización
    """
    if "reto8" in cliente.list_database_names():
        cliente.drop_database('reto8')

    db = cliente["reto8"]
    colec_regr = db["regresion"]
    colec_opt = db["optimizacion"]

    return db, colec_regr, colec_opt