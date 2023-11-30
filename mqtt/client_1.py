from datetime import datetime
from time import sleep
import json
import paho.mqtt.client as paho
import logging
import csv
from elasticsearch import Elasticsearch
from functions import *
from dateutil.parser import parse


handler = Handler()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(name)s > %(levelname)s: %(message)s')

broker = "localhost"

cliente1 = paho.Client("Cliente 1", clean_session=True)

logging.debug(f'Connecting to broker {broker}')

cliente1.on_message = handler.on_message

cliente1.connect(broker)

topic1 = "topic_data"

with open('../Datos/Transformados/df_firewall.csv', 'r') as archivo_csv:
    lector_csv = csv.DictReader(archivo_csv)

    while True:
        for row in lector_csv:
            row['timestamp']=str(datetime.now())
            cliente1.publish(topic1, json.dumps(row))
            logging.info(f"Enviando a {topic1}: {row}")
            time.sleep(2)
        