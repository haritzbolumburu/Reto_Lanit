Reto8: Equipo Amarillo
==========
Reto desarrollado con Lanit.

Proceso seguido
--------

El proyecto comienza importando los datos de los compresores y realizando un análisis exploratorio inicial para entender las características y la distribución de los mismos. Luego, se realiza el preprocesamiento de los datos que se indexarán en Elastic antes de gestionarlos con Logstash.

A continuación, se emplean modelos de regresión Catboost para predecir la potencia medida de cada compresor por separado, además de algoritmos de optimización para minimizar el consumo total de los compresores dado un caudal mínimo de aire que debe ser suministrado.

Los datos más relevantes de los análisis descriptivos se desplegarán en una aplicación multifuncional desarrollada en Dash, que proveerá al usuario de información sobre los puertos, compresores y demás datos de interés.

La mayoría de los gráficos se han realizado con la librería Plotly, que permite una interacción con los mismos, y ya se despliegan sobre la aplicación de Dash. También se han realizado otros gráficos originales con la librería Matplotlib para la interpretación de los resultados de la optimización.

En la carpeta se incluirán tres Jupyter Notebooks por si el usuario desea consultar alguna duda. Sus nombres empiezan por "notebook_", y se incluyen el notebook del preprocesamiento, análisis exploratorio y gráficos de los compresores ("notebook_exploratorio_compresores.ipynb"), así como "notebook_elasticsearch.ipynb", donde se realizó el análisis exploratorio de los datos a indexar en Elastic, y "notebook_optimizacion.ipynb", donde se puede ejecutar el código de la optimización de los compresores de manera más ágil reduciendo el número de generaciones y visualizando los resultados de manera más cómoda.

Requerimientos de ejecución
------------

Para ejecutar este proyecto, es necesario importar el fichero "amarillo08.yml" para crear un environment que ya contiene todos los paquetes necesarios para la ejecución del proyecto.

Además, para la indexacción de los datos en Elastic, el usuario insertará sus credenciales en el un fichero ".pwd.json", situado en la carpeta principal y cuya estructura es la siguiente:
{
    "ruta" : "127.0.0.1",
    "usuario" : "root",
    "contrasena_sql" : "1234",
}

Se presupone que dentro del proyecto, existe la carpeta "Datos", con la carpeta "Originales" en ella, la cual incluye los ficheros con los datos originales. La carpeta "Transformados" se creará con la ejecución del proyecto, y recibirá sus ficheros de manera automática, así como las carpetas "Graficos" y "Modelos".

La estructura inicial debe ser la siguiente:
- Datos
  - Originales
    - Compresores
      - CompA.csv
      - CompB.csv
      - CompC.csv
      - CompD.csv
    - FW_logs
      - log2.csv
    - Network_flows
      - Dataset-Unicauca-Version2-87Atts.csv
    - wifitraps_anonimo.log

Una vez que se tengan todos los paquetes instalados y todos los datos originales introducidos en la carpeta, se debe **ejecutar en primer lugar el archivo principal del proyecto, llamado "main.py"**. Este archivo se encargará de ejecutar todos los demás archivos necesarios para la ejecución del proyecto, a excepción del bot que está en **elastic_y_bot.py, que deberá ejecutarse después de main.py** para que el bot pueda funcionar correctamente e indexe los datos en Elastic.

El código está bien documentado y se ha organizado en diferentes módulos y funciones para facilitar su comprensión, mantenimiento y despliegue.