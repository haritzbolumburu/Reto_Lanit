import telebot
import datetime
from elasticsearch import Elasticsearch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import os
import signal
import time
from elasticsearch.helpers import scan
import pandas as pd

TOKEN = '6286922560:AAH1RBKZ10F74n5Kn_94IT0VWaZ2HP1kYPA'
bot = telebot.TeleBot(TOKEN)

elasticsearch_process = None
logstash_process = None
client = None

# Definimos funciones necesarias

def verificar_crear_ruta(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

def iniciar_elasticsearch():
    global elasticsearch_process
    elasticsearch_process = subprocess.Popen(['/home/reto08/elasticsearch-8.7.0/bin/elasticsearch'])
    time.sleep(10)
    global client
    client = Elasticsearch(['http://localhost:9200'])

def iniciar_logstash():
    global logstash_process
    logstash_process = subprocess.Popen(['/home/reto08/logstash-8.7.0/bin/logstash', '-f', '/home/reto08/logstash-8.7.0/config/conf_reto8.conf'])

def detener_elasticsearch():
    global elasticsearch_process
    if elasticsearch_process:
        elasticsearch_process.terminate()
        elasticsearch_process.wait()
        elasticsearch_process = None

def detener_logstash():
    global logstash_process
    if logstash_process:
        os.kill(logstash_process.pid, signal.SIGTERM)
        logstash_process.wait()
        logstash_process = None

def obtener_todos(indice):
    index = indice
    query = {'query': {'match_all': {}}}

    resp_ac = scan(client, query=query, index=index, size=10000)

    result=[]
    # Iterar sobre los resultados
    for hit in resp_ac:
        # Hacer lo que necesites con cada registro
        result.append(hit['_source'])
    df = pd.DataFrame(result)
    return df


# Crear ruta para guardar los graficos

ruta = "/home/reto08/Escritorio/Reto8/Graficos_bot"
verificar_crear_ruta(ruta)


# Comandos del bot

@bot.message_handler(commands=['start'])
def send_a(message):

    usuario = message.from_user.first_name
    bot.reply_to(message, "Kaixo " + usuario + '\n' + "Este es el bot creado por 6 estudiantes del grado Business Data Analitycs de  Mondragon Unibertsitatea" + '\n\n' + 
                 "Somos el equipo amarillo y con este bot se podran indexar los datos del reto en los indices de Elasticsearch pertinentes y realizar consultas para obtener graficos puediendo monitorizar ciertas variables."+'\n'+
                 "Estos son los indices que se han creado:"+'\n'+
                 "1- firewall_reto_8_amarillo: "+'\n'+
                 "2- wifi_reto_8_amarillo: "+'\n'+
                 "3- networkflow_reto_8_amarillo: "+'\n'+
                 "Con el comando /help podrás ver los distintos comandos y servicios que ofrece este bot."+'\n'+
                 "IMPORTANTE: Antes de realizar las consultas deberáss tener informacion almacenada en ellas e introducir tu propia contraseña de Elasticsearch. "+'\n'+
                 "En el caso de la indexación no será necesario."+'\n'+
                 "No es necesario arrancar los servicios de Elasticsearch ni de Logstash manualmente; el bot los activará automaticamente.")


@bot.message_handler(commands=['help'])    
def comandos(message):
    bot.reply_to(message, 
                 "Estos son los comandos implementados en el bot."+'\n'+
                 "/password --> Antes de nada deberas de introducir tu contraseña de ElasticSearch."+'\n' +
                 "/count indice --> Podrás ver la cantidad de registros indexados en el indice indicado."+'\n' +
                 "/grafico_barras_acciones --> Gráfico de barras de la variable acciones de los datos de firewall."+'\n' +
                 "/grafico_sectores_acciones --> Gráfico de sectores de la variable acciones de los datos de firewall."+'\n' +
                 "/grafico_barras_anomalias --> Gráfico de barras de la variable anomalías de los datos de firewall."+'\n' +
                 "/grafico_sectores_anomalias --> Grafico de sectores de la variable anomalías de los datos de firewall."+'\n' +
                 "/intervalo_indexacion --> Gráfico de sectores que muestra por cada 5 segundos los datos que se han indexado del fichero de firewall y cuantos de ellos son anomalías."+'\n' +
                 "/grafico_barras_wlan--> Gráfico de barras que muestra la cantidad de registros por cada tipo de WLAN de los datos de wifi."+'\n' +
                 "/indexar_conf--> Comando que ejecuta el fichero de configuración e indexa los datos en Elastic."+'\n' +
                 "/parar_conf--> Comando que detiene el servicio de Elasticsearch y Logstash y detiene la indexación de los datos.")
    


@bot.message_handler(commands=['password'])
def contraseña(message):

    try:
        iniciar_elasticsearch()
        bot.reply_to(message, 'Iniciando Elasticsearch...')
        time.sleep(20)
        password = message.text.split(" ")[1]
        global client
        client = Elasticsearch(
            'https://localhost:9200',
            ca_certs='/home/reto08/elasticsearch-8.7.0/config/certs/http_ca.crt',
            basic_auth=('elastic', password))
        client.info()
        
        resp = "Contraseña recibida!"

    except:
        resp = "No has introducido ninguna contraseña, la introducida es erronea o el servicio de Elasticsearch no puede arrancarse"

    bot.reply_to(message, resp)


@bot.message_handler(commands=['count'])
def contar(message):

    try:
        indice = message.text.split(" ")[1]
        df=obtener_todos(indice)

        bot.reply_to(message, f"El total de registros indexados en el indice {indice} es {df.shape[0]}")

    except:
        bot.reply_to(message, "Antes de nada introduzca su contraseña de ElasticSearch (/password su_contraseña)")



@bot.message_handler(commands=['grafico_barras_acciones'])
def grafico_barras_acciones(message):

    try:

        bot.reply_to(message, "Realizando grafico...")
        df=obtener_todos("firewall_reto_8_amarillo")

        action_counts = df['Action'].value_counts()

        plt.figure(figsize=(8, 6))
        action_counts.plot(kind='bar')
        plt.xlabel('Acciones')
        plt.title('Gráfico de barras: Acciones')
        plt.grid(True)

        plt.savefig('Graficos_bot/acciones_barras.png')

        id=message.chat.id
        bot.send_photo( chat_id=id,photo=open('Graficos_bot/acciones_barras.png', 'rb'))
    except:
        bot.reply_to(message, "Algo ha ido mal. Antes de nada introduzca su contraseña de ElasticSearch (/password su_contraseña)")
    finally:
        plt.clf()


@bot.message_handler(commands=['grafico_sectores_acciones'])
def grafico_sectores_acciones(message):

    try:
        bot.reply_to(message, "Realizando grafico...")
        df=obtener_todos("firewall_reto_8_amarillo")

        action_counts = df['Action'].value_counts()

        plt.figure(figsize=(8, 6))
        plt.pie(action_counts, labels=action_counts.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
 
        plt.savefig('Graficos_bot/acciones_sector.png')

        id=message.chat.id
        bot.send_photo( chat_id=id,photo=open('Graficos_bot/acciones_sector.png', 'rb'))
    except:
        bot.reply_to(message, "Algo ha ido mal. Antes de nada introduzca su contraseña de ElasticSearch (/password su_contraseña)")
    finally:
        plt.clf()

@bot.message_handler(commands=['grafico_barras_anomalias'])
def grafico_barras_anomalias(message):

    try:
        bot.reply_to(message, "Realizando grafico...")
        df=obtener_todos("firewall_reto_8_amarillo")

        anomalia_counts = df['anomalia'].value_counts()

        plt.figure(figsize=(8, 6))
        anomalia_counts.plot(kind='bar')
        plt.xlabel('Anomalias')
        plt.title('Gráfico de barras: Anomalias')
        plt.grid(True)

        plt.savefig('Graficos_bot/anomalias_barras.png')

        id=message.chat.id
        bot.send_photo( chat_id=id,photo=open('Graficos_bot/anomalias_barras.png', 'rb'))
    except:
        bot.reply_to(message, "Algo ha ido mal. Antes de nada introduzca su contraseña de ElasticSearch (/password su_contraseña)")
    finally:
        plt.clf()



@bot.message_handler(commands=['grafico_sectores_anomalias'])
def grafico_sectores_anomalias(message):

    try:
        
        bot.reply_to(message, "Realizando grafico...")
        df=obtener_todos("firewall_reto_8_amarillo")

        anomalia_counts = df['anomalia'].value_counts()

        plt.figure(figsize=(8, 6))
        plt.pie(anomalia_counts, labels=anomalia_counts.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  

        plt.savefig('Graficos_bot/anomalias_sector.png')

        id=message.chat.id
        bot.send_photo( chat_id=id,photo=open('Graficos_bot/anomalias_sector.png', 'rb'))
    except:
        bot.reply_to(message, "Algo ha ido mal. Antes de nada introduzca su contraseña de ElasticSearch (/password su_contraseña)")
    finally:
        plt.clf()
        detener_elasticsearch()


@bot.message_handler(commands=['intervalo_indexacion'])
def intervalo_indexacion(message):

    try:
        
        bot.reply_to(message, "Realizando grafico...")
        df=obtener_todos("firewall_reto_8_amarillo")
    
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])

        interval = '5S'

        counts_per_interval = df.groupby([pd.Grouper(key='@timestamp', freq=interval), 'anomalia']).size().unstack()

        plt.figure(figsize=(12, 6))
        counts_per_interval.plot(kind='bar', stacked=True, color=['red', 'blue'])
        plt.xlabel('Intervalo de 5 segundos')
        plt.ylabel('Cantidad de registros')
        plt.title('Cantidad de registros indexados por intervalo de 5 segundos')

        plt.savefig('Graficos_bot/intervalo_indexacion.png')

        id=message.chat.id
        bot.send_photo( chat_id=id,photo=open('Graficos_bot/intervalo_indexacion.png', 'rb'))
    except:
        bot.reply_to(message, "Algo ha ido mal. Antes de nada introduzca su contraseña de ElasticSearch (/password su_contraseña)")
    finally:
        plt.clf()
        detener_elasticsearch()




@bot.message_handler(commands=['grafico_barras_wlan'])
def grafico_barras_wlan(message):


    try:
        bot.reply_to(message, "Realizando grafico...")
        df=obtener_todos("wifi_reto_8_amarillo")

        wlan_counts = df['nombre2_dispositivo'].value_counts()

        plt.figure(figsize=(8, 6))
        wlan_counts.plot(kind='bar')
        plt.xlabel('Tipo WLAN')
        plt.title('Gráfico de barras: WLAN')
        plt.grid(True)

        plt.savefig('Graficos_bot/barras_wlan.png')

        id=message.chat.id
        bot.send_photo( chat_id=id,photo=open('Graficos_bot/barras_wlan.png', 'rb'))
    except:
        bot.reply_to(message, "Algo ha ido mal. Antes de nada introduzca su contraseña de ElasticSearch (/password su_contraseña)")
    finally:
        plt.clf()



@bot.message_handler(commands=['indexar_conf'])
def indexa_conf(message):
    if __name__ == '__main__':
        bot.reply_to(message, "Iniciando...")
        iniciar_elasticsearch()
        bot.reply_to(message, "Elasticsearch iniciado")
        iniciar_logstash()
        bot.reply_to(message, "Logstasg iniciado")
        

@bot.message_handler(commands=['parar_conf'])
def parar_conf(message):
    if __name__ == '__main__':
        bot.reply_to(message, "Parando...")
        detener_logstash()
        bot.reply_to(message, "Logstash parado")
        detener_elasticsearch()
        bot.reply_to(message, "Elasticsearch parado")



bot.polling()