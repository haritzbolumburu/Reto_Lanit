import paho.mqtt.client as paho
import time
import csv
from csv import DictReader
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
import json
import logging


class Handler:

    def __init__(self):
        self.datos = []

    def on_message(self, client, userdata, message):
        datos = self.datos

        if message.topic=="topic_data":
            record=json.loads(message.payload.decode("utf-8"))
            datos.append(record)
        
        if message.topic=="topic_queries":
            informacion=pd.DataFrame(datos)
            consultas(message.payload.decode("utf-8"), informacion)



def on_connect(client, userdata, flags, rc):
    logging.debug(f'Connected with result code {rc}')
    logging.debug("Subscribing")




def consultas(query,df):

    if query in ["/ultimo","/mostrar_datos"]:
        if query =="/ultimo":
            try:
                df_filt=df[df['anomalia']=='anomalia'].copy()
                ultimo=df_filt.iloc[-1]
                msg="\nLa ultima anomalia registrada tiene los siguentes campos: \n Accion => "+ultimo['Action']+"\n Trafico => "+ultimo['trafico']+"\n Bytes enviados => "+ultimo['Bytes Sent']
                
            except:
                msg="\nNo hay datos registrados que cumplan la solicitud"
            print(msg)

        if query =="/mostrar_datos":
            try:
                columnas=["Source Port","Destination Port","Bytes Sent","Bytes Received","Elapsed Time (sec)"]
                print(df[columnas])
            except:
                print("\nLos datos no estan disponibles")
                
    else: 
        try:
            index_todo(df, query)
        except:
            print("\nLa contraseña es incorrecta o el servicio de Elasticsearch no esta en marcha")


def index_todo(df, query): 

    ELASTIC_PASSWORD = query
    logging.debug("Iniciando Elasticsearch...")
    client = Elasticsearch(
        "https://localhost:9200",
        ca_certs="/home/reto08/elasticsearch-8.7.0/config/certs/http_ca.crt",
        http_auth=("elastic", ELASTIC_PASSWORD))


    indice = 1
    numericas=['Bytes', 'Bytes Sent','Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received', 'diferencia_trafico']
    strings=['Source Port', 'Destination Port', 'NAT Source Port','NAT Destination Port', 'Action','trafico', 'anomalia','timestamp']
    logging.debug("Indexando datos...")
    for index, row in df.iterrows():
        row=row.to_dict()
        for num in numericas:
                 row[num] = int(row[num])
        for let in strings:
                 row[let]=str(row[let])
            
        resp = client.index(index = 'firewall_reto_8_amarillo_mqtt', id = indice, document=row)
        indice = indice + 1
    logging.debug("\nDatos indexados con éxito")
        