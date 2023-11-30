import pandas as pd
import numpy as np
import scipy.stats as stats


### FICHERO FIREWALL

# PREPROCESAMIENTO

def preprocesamiento_firewall(RUTA:str) -> pd.DataFrame:
    """Lee el fichero del firewall desde la ruta y corrige los tipos de datos

    Args:
        RUTA (str): ruta al fichero de firewall

    Returns:
        pd.DataFrame: dataframe firewall con tipos de datos corregidos
    """
    df=pd.read_csv(RUTA)
    puertos=["Source Port", "Destination Port", "NAT Source Port","NAT Destination Port"]
    df[puertos]=df[puertos].astype('object')
    return df
    

# TEST DE NORMALIDAD

def normality_test_shapiro(data:np.array, alpha=0.05) ->str:
    """Aplica el test de normalidad shapiro para ver si la distribucion de los datos es normal

    Args:
        data (np.array): variable sobre la que se quiere aplicar el test
        alpha (float, optional): El umbral del p_value. Por defecto 0.05.

    Returns:
        str: devuelve si la distribucion es normal o no
    """

    stat, p = stats.shapiro(data)
    if p > alpha:
        return "distribucion normal"
    else:
        return "distribucion no normal"
    
def normality_test_ks(data:np.array, alpha=0.05) -> str:
    """Aplica el test de normalidad KS  para ver si la distribucion de los datos es normal

    Args:
        data (np.array): ariable sobre la que se quiere aplicar el test
        alpha (float, optional): El umbral del p_value. Por defecto 0.05.

    Returns:
        str: devuelve si la distribucion es normal o no
    """
    # Asumimos que la muestra sigue una distribución normal
    mu, sigma = np.mean(data), np.std(data)
    normal_data = (data - mu) / sigma

    # Realizamos la prueba de KS
    stat, p_value = stats.kstest(normal_data, 'norm')

    # Evaluamos el resultado de la prueba
    if p_value > alpha:
        return "distribucion normal"
    else:
        return "distribucion no normal"

def test_normalidad(df:pd.DataFrame):
    """Test de normalidad SHAPIRO Y KS sobre las variables numericas de un dataframe

    Args:
        df (pd.DataFrame): dataframe al que aplicar los tests

    Returns:
        _type_: None
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64'])
    print("Test de normalidad SHAPIRO:")
    for i in numeric_cols:
        print(i+ " " + str(normality_test_shapiro(df[i])))
    print("Test de normalidad KS:")    
    for i in numeric_cols:
        print(i+ " " + str(normality_test_ks(df[i])))
    return None


# DETECCION DE ANOMALIAS
def anomalias(df:pd.DataFrame)->pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): dataframe sobre el que se quieren calcular las anomalias

    Returns:
        pd.DataFrame: data frame con las nuevas variables
    """
    # Calcula el IQR para cada columna numérica
    Q1_dif = df['diferencia_trafico'].quantile(0.1)
    Q3_dif = df['diferencia_trafico'].quantile(0.9)
    Q1_pkt = df['pkts_sent'].quantile(0.1)
    Q3_pkt = df['pkts_sent'].quantile(0.9)
    IQR_dif = Q3_dif - Q1_dif
    IQR_pkt = Q3_pkt - Q1_pkt

    # Define el umbral para identificar outliers
    umbral_dif = 1.5
    umbral_pkt = 1.5

    # Crea una nueva columna en el dataframe para indicar si algún valor es outlier
    outliers = ((df['diferencia_trafico'] < (Q1_dif - umbral_dif * IQR_dif)) | (df['diferencia_trafico'] > (Q3_dif + umbral_dif * IQR_dif)) |  (df['pkts_sent'] > (Q3_pkt + umbral_pkt * IQR_pkt)))
    df['anomalia'] = outliers

    df['anomalia'] = np.where(df['anomalia'], "anomalia", "normal")
    return df


def creacion_variables(df:pd.DataFrame)->pd.DataFrame:
    """Crea variables a partir de los datos (trafico, diferencia de trafico y anomalias)

    Args:
        df (pd.DataFrame): data frame sobre el que crear las nuevas variables

    Returns:
        pd.DataFrame: data frame con las nuevas variables
    """
    df['trafico']=df['pkts_received'] > df['pkts_sent']
    df['trafico'] = np.where(df['trafico'], 'superior', 'inferior')

    df['diferencia_trafico']=df['pkts_received']-df['pkts_sent']
    df=anomalias(df)
    return df


#### FICHERO WIFI

def preprocesamiento_wifi(RUTA:str)->pd.DataFrame:
    """Carga el fichero .log de la ruta introducida

    Args:
        RUTA (str): ruta al fichero de wifi

    Returns:
        pd.DataFrame: el fichero de wifi en formato data frame
    """

    # Leer el archivo de texto y almacenar las líneas en una lista
    with open(RUTA, 'r') as archivo:
        lineas = archivo.readlines()

    # Crear una lista de diccionarios para almacenar los campos
    campos = []

    # Procesar cada línea del archivo
    for linea in lineas:
        # Dividir la línea en pares de clave y valor
        pares = linea.strip().split(',')

        # Crear un diccionario para cada campo y agregarlo a la lista de campos
        campo = {}
        for par in pares:
            if '=' not in par:
                continue  # Omitir pares sin el símbolo '='

            clave_valor = par.strip().split('=', 1)
            if len(clave_valor) != 2:
                continue  # Omitir pares sin una clave o valor válido

            clave, valor = clave_valor
            campo[clave.strip()] = valor.strip()
        campos.append(campo)

    # Crear el DataFrame a partir de la lista de campos
    df = pd.DataFrame(campos)
    df = df.drop(df.columns[-1], axis=1)
    return df


def renombrar_columnas_wifi(df:pd.DataFrame)-> pd.DataFrame:
    """renombra las columnas del dataframe de los datos del wifi

    Args:
        df (pd.DataFrame): dataframe del wifi al que se le quieren cambiar las columnas

    Returns:
        pd.DataFrame: dataframe con las columnas renombradas
    """
    columnas_existentes = df.columns.tolist()
    nuevos_nombres=['tiempo_ejecucion', 'identificador_OID', 'valor1', 'nombre1_dispositivo', 'valor2', 'valor3', 'direccion_ip', 'correo', 'nombre2']
    diccionario_nombres={}
    if len(columnas_existentes) == len(nuevos_nombres):

        for i in range(0,len(columnas_existentes)):
            diccionario_nombres[columnas_existentes[i]]=nuevos_nombres[i]

        df=df.rename(columns=diccionario_nombres)
        return df
    else:
        print('No hay el mismo numero de columnas')
        return None
    

def limpieza_wifi(df:pd.DataFrame)->pd.DataFrame:
    """limpieza del dataframe de los datos del wifi (corregir los missings y patrones incorrectos)

    Args:
        df (pd.DataFrame): data frame de datos de wifi

    Returns:
        pd.DataFrame: data frame de datos del wifi limpio
    """

    # Crea un nuevo dataset excluyendo las filas con más de 2 valores faltantes
    filas_con_missings = df.isnull().sum(axis=1) > 2
    df = df[~filas_con_missings]

    # QUITAMOS LAS QUE NO SIGUEN EL PATRON

    patron = r'^([A-F0-9]{2}\s|\w{2}\s){5}[A-F0-9]{2}$'

    filas_no_cumplen_formato = df[~df['valor2'].str.match(patron, na=False)]
    df = df.drop(filas_no_cumplen_formato.index) # quitamos 481

    filas_no_cumplen_formato = df[~df['valor1'].str.match(patron, na=False)]
    df = df.drop(filas_no_cumplen_formato.index)# 128 mal

    df['correo'] = df['correo'].fillna('desconocido')
    
    return df

def guardar_wifi(df:pd.DataFrame):
    """
    Esta función toma un DataFrame de Pandas como entrada y crea una cadena de texto en el formato requerido. 
    Luego, escribe esta cadena en un archivo log.

    Args:
        df (pd.DataFrame): El DataFrame que contiene la información a guardar.

    Returns:
        None
    """
    log_string = ''
    for _, row in df.iterrows():
        log_string += f"DISMAN-EVENT-MIB::sysUpTimeInstance = {row['tiempo_ejecucion']}, "
        log_string += f"SNMPv2-MIB::snmpTrapOID.0 = {row['identificador_OID']}, "
        log_string += f"SNMPv2-SMI::enterprises.9.9.599.1.3.1.1.1.0 = {row['valor1']}, "
        log_string += f"SNMPv2-SMI::enterprises.9.9.513.1.1.1.1.5.0 = {row['nombre1_dispositivo']}, "
        log_string += f"SNMPv2-SMI::enterprises.9.9.599.1.3.1.1.8.0 = {row['valor2']}, "
        log_string += f"SNMPv2-SMI::enterprises.9.9.513.1.2.1.1.1.0 = {row['valor3']}, "
        log_string += f"SNMPv2-SMI::enterprises.9.9.599.1.3.1.1.10.0 = {row['direccion_ip']}, "
        log_string += f"SNMPv2-SMI::enterprises.9.9.599.1.3.1.1.27.0 = {row['correo']}, "
        log_string += f"SNMPv2-SMI::enterprises.9.9.599.1.3.1.1.28.0 = {row['nombre2']}\n"

    file_path = 'Datos/Transformados/df_wifi.log'  # Ruta y nombre de archivo
    with open(file_path, 'w') as file:
        file.write(log_string)
    
    return None


#### FICHERO NETWORK FLOWS

def preprocesamiento_nf(RUTA:str) -> pd.DataFrame:
    """
    Esta función realiza un preprocesamiento en un archivo CSV especificado por la ruta de entrada.
    Carga el archivo en un DataFrame de Pandas, restablece su índice y devuelve el DataFrame.

    Args:
        RUTA (str): La ruta del archivo a preprocesar.

    Returns:
        df (pd.DataFrame): El DataFrame preprocesado.
    """
    df = pd.read_csv(RUTA)
    df.reset_index(inplace=True)
    return df