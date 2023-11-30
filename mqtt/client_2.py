import paho.mqtt.client as paho
import logging
from functions import *
import logging
from time import sleep
import pandas as pd


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(name)s > %(levelname)s: %(message)s')

broker = "localhost"

cliente2 = paho.Client("Cliente 2", clean_session=True)

logging.debug(f'Connecting to broker {broker}')


cliente2.connect(broker)

topic2 = "topic_queries"

while True:
    print('\nEstas son las opciones que tienes disponibles: \n \
    1- "/indexar": Indexar todos los datos \n \
    2- "/ultimo": Muestra el ultimo registro anómalo indexado \n \
    3- "/mostrar_datos": Muestra todos los datos')

    query = input('\n¿Qué opción prefieres? ')

    if query=="/indexar":
        query=input("Introduce la contraseña de Elasticsearch => ")

    cliente2.publish(topic2, query, qos=1)
    logging.info(f"\nEnviando la query  {query} a {topic2}")