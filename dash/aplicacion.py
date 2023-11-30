import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import os
from dash import dash_table
from dash.dependencies import Input, Output

# Carga de datos
def carga_compresores(ruta='../Datos/Originales/Compresores'):
    """Carga de los ficheros con un bucle for en una funcion

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

compresores = carga_compresores()

puertos=pd.read_csv(os.path.join('../Datos/Originales/FW_logs', "log2.csv"))
allow = puertos[puertos['Action'] == 'allow']

anomalias=pd.read_csv(os.path.join('../Datos/Transformados', "df_firewall.csv"))

# Hoja de estilo externa y asignación
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Definimos cada una de las FILAS por separado
row1 = html.Div(
    [html.Div(
        [
            html.H1('Análisis exploratorio proyecto Lanit-Mondragon Unibertsitatea'),
            html.Hr(),
            html.H6('Se ha desarrollado una aplicación Dash utilizando los datos del proyecto proporcionados por Lanit con el objetivo de visualizar y realizar un análisis exploratorio de los mismos. La aplicación permite explorar los datos de manera interactiva a través de varios filtros.'),
            html.Br()
        ],
        style={
            'display': 'inline-block',
            'margin-left': '10px',
            'background-color': 'black',
            'padding': '10px',
            'border-radius': '5px',
            'color' : 'white' 
        }
    )
    ],
    style={'background-color': 'black'}
)


tab_Lanit = html.Div(
    className="row",
    children=[
        # Columna de la imagen de Lanit
        html.Div(
            className="six columns",
            children=[
                html.Img(src=app.get_asset_url("lanit.png"), style={"width": "100%"}),
            ],
        ),
        # Columna del texto explicativo
        html.Div(
            className="six columns",
            children=[
                html.Br(),
                html.H1("LANIT",style={'color': '#E42218','font-family': 'Montserrat'}),
                html.P("Lanit es un grupo de empresas que cuenta con más de 25 años de experiencia en los ámbitos de consultoría tecnológica y de negocio. Sus principales especializaciones son la analítica de datos y el mantenimiento integral de infraestructuras de tecnologías de la información."),
                html.P("Cuentan con un crecimiento sólido y en constante evolución ya sea en el uso de aplicaciones como en métodos ágiles para responder rápidamente a los nuevos desafíos que sus clientes presentan cada día."),
                html.P("Asimismo, en la actualidad cuentan con un equipo de más de 40 personas, en su mayoría licenciados en ingeniería informática y matemáticas. Se trata de un equipo de expertos en todo el proceso del dato, trabajando remotamente o in situ en grandes proyectos o como parte de los propios equipos de cliente."),
            ],
        ),
    ],
)


row2 = html.Div([
    html.Div([
        html.Label(children='Seleccionar el rango de temperatura:'),
        dcc.RangeSlider(
        id = 'range_slider',
        min=-6,
        max=40,
        step=5,
        value=[-6,40],
        )  
    ], className = 'nine columns'),

 html.Div([
        html.Label('Selecciona el compresor:'),
        dcc.Dropdown(
        id = 'dropdown',
        options=[
            {'label': i, 'value': i} for i in compresores['compresor'].unique()
            ],
        value= ['CompA','CompC'],
        multi=True
    )    
    ], className = 'three columns'),

], className = 'row')


row3 = html.Div([
    html.Div([
        dcc.Graph(id = 'graf1')  
    ], className = 'six columns'),

    html.Div([
        dcc.Graph(id='graf2')
    ], className= 'six columns')

], className = 'row')


row4 = html.Div([
    html.Div([
        dcc.Graph(id = 'graf3')  
    ], className = 'six columns'),

    html.Div([
        dcc.Graph(id='graf4')
    ], className= 'six columns')

], className = 'row')

row5 = html.Div([
    html.Div([
        html.Label(children='Seleccionar el rango de bytes:'),
        dcc.RangeSlider(
            id='range_slider2',
            min= allow['Bytes'].min(),
            max=60000000,
            step=3500000,
            value=[60, 60000000],
        )  
    ], className='twelve columns')
])

row6 = html.Div([
    html.Div([
        dcc.Graph(id = 'graf5')  
    ], className = 'twelve columns'),

], className = 'row')

row7 = html.Div([

    html.Div([
        dcc.Graph(id='graf6')
    ], className= 'six columns'),

    html.Div([
    dash_table.DataTable(
    id = 'tabla',
    columns = [{'name': i, 'id':i} for i in puertos[['Action','Bytes','Bytes Sent','Bytes Received']].columns],
    data = puertos[['Action','Bytes','Bytes Sent','Bytes Received']].to_dict('records'),
    filter_action = 'native', 
    sort_action = 'native',
    sort_mode='multi',
    page_size=10
    )
] ,className = 'six columns')

], className = 'row')

row_bins = html.Div([
    html.Div([
        html.Label(children='Seleccionar el número de puertos que deseas visualizar en el ranking:'),
        dcc.Slider(
        id='bin-slider',
        min=1,
        max=10,
        step=1,
        value=5,
        marks={str(i): str(i) for i in range(1, 11)}
    )
    ], className = 'twelve columns')
])

row8 = html.Div([
    html.Div([
        dcc.Graph(id = 'graf7')  
    ], className = 'six columns'),

    html.Div([
        dcc.Graph(id='graf8')
    ], className= 'six columns')

], className = 'row')

row9 = html.Div([
    html.Div([
        dcc.Graph(id = 'graf9')  
    ], className = 'eight columns'),

    html.Div([
        dcc.Graph(id='graf10')
    ], className= 'four columns')

], className = 'row')


tab1 = html.Div([
    row2,
    html.Br(),
    row3,
    html.Br(),
    row4
])

tab2 = html.Div([
    row5,
    row6,
    html.P('*En el rango de datos se han decidido eliminar los valores atípicos.'),
    html.Hr(),
    html.H6('A continuación, se muestra un recuento de las diferentes acciones de los puertos. Asimismo, una tabla que muestra las diferentes acciones y los bytes.'),
    row7

])

tab3 = html.Div([
    row_bins,
    html.Br(),
    row8
])

tab4 = html.Div([
        html.H6('A continuación, se muestran unos gráficos relacionados con las anomalías encontradas en los datos:'),
    html.Br(),
    row9
])



# Definimos el Layout
app.layout = html.Div([
    row1,
    dcc.Tabs([
        dcc.Tab(label="Sobre el cliente", children=tab_Lanit, style={'fontWeight': 'bold'}),
        dcc.Tab(label='Análisis compresores', children=tab1, style={'fontWeight': 'bold'}),
        dcc.Tab(label='Análisis general puertos', children=tab2, style={'fontWeight': 'bold'}),
        dcc.Tab(label='Puertos que más envían y reciben', children=tab3, style={'fontWeight': 'bold'}),
        dcc.Tab(label='Análisis anomalías puertos', children=tab4, style={'fontWeight': 'bold'})
    ])
])


# Función de callback para actualizar el gráfico
@app.callback(
    [dash.Output(component_id='graf1', component_property='figure'),
     dash.Output(component_id='graf2', component_property='figure'),
     dash.Output(component_id='graf3', component_property='figure'),
     dash.Output(component_id='graf4', component_property='figure'),
     dash.Output(component_id='graf5', component_property='figure'),
     dash.Output(component_id='graf6', component_property='figure'),
     dash.Output(component_id='tabla', component_property='data'),
     dash.Output(component_id='graf7', component_property='figure'),
     dash.Output(component_id='graf8', component_property='figure'),
     dash.Output(component_id='graf9', component_property='figure'),
     dash.Output(component_id='graf10', component_property='figure')],
    [dash.Input(component_id='range_slider', component_property='value'),
     dash.Input(component_id='dropdown', component_property='value'),
     dash.Input(component_id='range_slider2', component_property='value'),
     dash.Input(component_id='bin-slider', component_property='value')]
)

def total(temperatura, comp, bytes,num_bins):
    #Análisis compresores
    datos_filtrados = compresores[(compresores['Temperatura'] >= temperatura[0]) & (compresores['Temperatura'] <= temperatura[1])]
    datos_filtrados = datos_filtrados[datos_filtrados['compresor'].isin(comp)]
    
    #Colores graficos
    colores = ['#70e8b7','#8b64b7','#e79eff','#b7fadf']

    fig1 = px.scatter(datos_filtrados, x='Temperatura', y='Presion', color='compresor', color_discrete_sequence=colores)
    fig1.update_layout(title='Temperatura y presión por compresor', xaxis_title='Temperatura', yaxis_title='Presión',
                    width=700, height=500)

    fig2 = px.scatter(datos_filtrados, x='Frecuencia', y='Potencia_Medida', color='compresor', color_discrete_sequence=colores)
    fig2.update_layout(title='Frecuencia y potencia medida por compresor', xaxis_title='Frecuencia', yaxis_title='Potencia medida',
                    width=700, height=500)

    frecuencias = datos_filtrados['compresor'].value_counts()
    fig3 = px.bar(frecuencias, x=frecuencias.index, y=frecuencias, color=frecuencias.index, color_discrete_sequence=colores)
    fig3.update_layout(title='Cantidad de filas por compresor', xaxis_title='Compresor', yaxis_title='Frecuencia',
                    width=700, height=500)

    datos_filtrados['duplicado'] = datos_filtrados.duplicated()
    datos_filtrados[datos_filtrados['duplicado'] == True].groupby('compresor').count()
    fig4 = px.scatter(datos_filtrados, x='Frecuencia', y='Potencia_Medida', color='duplicado', color_discrete_sequence=colores)
    fig4.update_layout(title='Frecuencia y potencia medida, marcando los duplicados',
                    xaxis_title='Frecuencia', yaxis_title='Potencia medida')
    

    #Analisis puertos
    datos_filtrados2 = allow[(allow['Bytes'] >= bytes[0]) & (allow['Bytes'] <= bytes[1])]
    datos_filtrados3 = puertos.copy()

    colores_morados = ['#d9c8e1', '#b89ed1', '#8b64b7', '#5c399d', '#351e82', '#0e0458']
    fig5 = px.scatter(datos_filtrados2, y='Bytes', color='Packets', color_continuous_scale=colores_morados)
    fig5.update_layout(coloraxis_colorbar=dict(title='Packets'))
    fig6 = px.pie(datos_filtrados3, names='Action', title='Recuento de acciones', hole=0.5, color_discrete_sequence=colores_morados)
    fig6.update_traces(textposition='inside', textinfo='percent+label')
    fig6.add_annotation(text='La acción más repetida es "allow"', x=1.2, y=0.5, showarrow=False)

    tabla1 = datos_filtrados3.to_dict('records')
    
   # Analisis de los puertos que mas bytes envian y reciben
    df2 = datos_filtrados3.groupby(['Source Port']).sum(numeric_only=True).sort_values(by=['Bytes Sent'], ascending=False).head(num_bins).reset_index()
    df3 = datos_filtrados3.groupby(['Source Port']).sum(numeric_only=True).sort_values(by=['Bytes Received'], ascending=False).head(num_bins).reset_index()
    df2['Source Port'] = df2['Source Port'].astype(str)
    df3['Source Port'] = df3['Source Port'].astype(str)

    fig7 = px.histogram(df2, x='Source Port', y='Bytes Sent',
                        title='Puertos que más Bytes envían',
                        color_discrete_sequence= ['#8b64b7'],
                        nbins=num_bins)

    fig7.update_xaxes(title_text='Puerto')
    fig7.update_yaxes(title_text='Bytes enviados')
    fig7.update_layout(legend_title='Puertos')

    fig8 = px.histogram(df3, x='Source Port', y='Bytes Received',
                        title='Puertos que más Bytes reciben',
                        color_discrete_sequence= ['#70e8b7'],
                        nbins=num_bins)

    fig8.update_xaxes(title_text='Puerto')
    fig8.update_yaxes(title_text='Bytes recibidos')
    fig8.update_layout(legend_title='Puertos')

    #Análisis anomalías
    datos_filtrados4 = anomalias.copy()
    datos_filtrados4['anomalia'] = datos_filtrados4['anomalia'].astype('category')
    fig9 = px.density_heatmap(datos_filtrados4, x='Action', y='anomalia', color_continuous_scale=colores_morados)
    fig9.update_layout(
        title='Mapa de calor de Detección de Anomalías por Acción',
        yaxis_title='Detección de Anomalías'
    )

    fig10 = px.pie(datos_filtrados4, names='trafico', title='Clasificación de la anomalía (tráfico)',
                hole=0.5, color_discrete_sequence= ['#b89ed1','#d9c8e1'])
    fig10.update_traces(textposition='inside', textinfo='percent+label')

    return fig1, fig2, fig3, fig4, fig5, fig6, tabla1, fig7, fig8, fig9, fig10

# Ejecucción de la app
if __name__ == '__main__':
    app.run_server(debug=False)