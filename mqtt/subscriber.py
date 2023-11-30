import paho.mqtt.client as paho
import logging
from functions import *


topic_1 = "topic_data"
topic_2 = "topic_queries"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s > %(name)s > %(levelname)s: %(message)s')

client = paho.Client("S", clean_session=True)

broker = "localhost"

logging.debug(f'Connecting to broker {broker}')
client.connect(broker)

clase = Handler()
client.on_connect = on_connect
client.on_message = clase.on_message

client.subscribe(topic_1)
client.subscribe(topic_2)

logging.debug("Start loop")
client.loop_forever()
