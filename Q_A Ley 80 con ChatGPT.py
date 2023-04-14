# Q&A Ley 80 con ChatGPT

__author__ = "Andres Ardila, Johan Triviño, Jenny Gamboa"
__maintainer__ = "Proyecto de Profundización"
__copyright__ = "Copyright 2023"
__version__ = "0.0.1"

import multiprocessing
import string
import nltk
from gensim.models import Word2Vec as w2v
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.stem import SnowballStemmer
import pandas as pd
import openai
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import os
from pathlib import Path as p

dir_root = p(__file__).parents[2]
print(dir_root)

ruta_source = str(p(dir_root) /'Sample_Data'/'For_Modeling'/'procesado_normas.json')
print(ruta_source)

server = app.server
# LLamado del archivo csv del normograma.
Norma=pd.read_json(ruta_source, encoding='utf-8-sig')
print(Norma.shape)

#Limpieza de columnas que no influyen en el análisis
Norma.drop(['index'], axis = 'columns', inplace=True)

#Definir el idioma de las stopwords en español
stopwords = set(stopwords.words('spanish'))

#Ciclos para la limpieza de los articulos, limpiando los caracteres especiales y las stopwords, y almacenarlas en una lista.
articulos = list(Norma['Contenido_Articulo'])
articulos_limpio = []
#Eliminamos líneas vacías 
for i in range(len(articulos)):
    art_limpio = []
    for linea in articulos[i].split("."):       
          linea = bytes(linea, 'utf-8').decode('utf-8', 'ignore')
          linea = "".join(c for c in linea if (c not in string.punctuation and c not in ['','¡','¿'])).lower()
          linea = linea.split(" ")
          #Eliminamos stopwords
          linea1=[]
          for palabra in list(linea):              
              if palabra not in stopwords and palabra not in string.punctuation:
                  linea1.append(palabra)
          art_limpio += linea1      
    articulos_limpio += [art_limpio]

#Función para la aplicación del steaming al corpus del texto. 
def steamer(Corpus):        
  Corpus=[i.lower() for i in Corpus]
  stem=SnowballStemmer('spanish')  
  Corpus=[' '.join([stem.stem(j) for j in i.split()]) for i in Corpus]
  return Corpus

#Ciclo para crear array de corpus luego de aplicar steam.
articulos_limpio_stm =[]
for i in range (len(articulos_limpio)):
  art_pos=[' '.join([j for j in articulos_limpio[i]])]
  art_limpio_stm =[]
  Corpus= steamer(art_pos)
  linea = Corpus[0].split(" ")
  articulos_limpio_stm.append(linea)


# Vectorización del array.
art2vec = w2v(
    articulos_limpio_stm,
    sg=1,
    seed=1,
    workers=multiprocessing.cpu_count(),
    vector_size=256,
    min_count=50,
    window=12
)

#Ciclo para crear array con las etiquetas.
labels = []
for word in art2vec.wv.key_to_index:
    labels.append(word)

#Funciones para la separación del texto, busqueda del artículo.

#funcion para separar texto
def septext(palabra):
  palabra = str(palabra)
  palabra = palabra.split(' ')
  palab=['art']
  count = 0
  Corpus= steamer(palabra)
  palabra = Corpus
  for pala in palabra:
    if pala in labels:
       count += 1
       if count >1:
        palab.append(pala)
       else:
         palab = [pala] 

  if len(palab) == 0:
    palabra= 'art'
  return palab

#Funcion para encontrar para encontrar Articulos
def art2find(palabra): 
  similaresword=[i[0] for i in art2vec.wv.most_similar(positive=palabra)]
  vword=art2vec.wv[palabra]
  cercanas =[]
  indices = []
  for i in similaresword:
    cercanas.append(i)   
  
  for j in cercanas:
    str_match = list(filter(lambda x: palabra and j in x, articulos_limpio_stm))    
    for k in str_match:
      ind = articulos_limpio_stm.index(k)
      if ind not in indices:
        indices.append(ind)
  return indices

#Retorno de Dataframe de contexto
def procesar_pregunta(pregunta):
    global Norma
    entrada = septext(pregunta)
    print(entrada)
    y = art2find(entrada)
    ind=np.argsort(y,axis=0)
    Nor = Norma[['Norma','No_Articulo', 'Contenido_Articulo']].loc[ind[0:20]]
    df = Nor.copy()
    return df

#Retorno de aticulos relacionados
def Articulos_relacionados(df):    
    df1 = df[['Norma','No_Articulo']]
    return df1


# Credenciales de OpenAI
openai.api_key = "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"

# Carga el contexto


# Estilo de la tabla
estilo_tabla = {
    'max-width': '100px',
    'margin': 'auto',
    'font-family': 'Arial',
    'border': '1px solid black',
    'border-collapse': 'collapse',
    'text-align': 'right',
}

estilo_encabezado = {
    'font-size': '18px',
    'background-color': 'lightgrey',
    'font-weight': 'bold',
    'padding': '5px',
    'textAlign': 'center'
}

estilo_celda = {
    'padding': '5px',
}

estilo_titulo = {
    'text-align': 'center',
}

# Inicializa la aplicación Dash
app = dash.Dash(__name__)

# Define la interfaz de usuario
app.layout = html.Div([
    html.H1("QA Contratación Pública Colombia", style=estilo_titulo),
    html.Div([
        dcc.Textarea(
            id="texto",
            placeholder="Escribe tu pregunta aquí...",
            style={"width": "100%", "height": 100}
        ),
        html.Button("Enviar", id="boton"),
    ]),
    html.Div(id="respuesta")    
])

# Define la función para hacer la pregunta a OpenAI
respuestas_dadas =[]
def hacer_pregunta(contexto, pregunta, respuestas_dadas):
    prompt = f"{contexto} en Colombia  Pregunta: {pregunta} Respuesta:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    for choice in response.choices:
        if "Respuesta:" in choice.text:
            nueva_respuesta = choice.text.replace("Respuesta:", "").strip()
            if nueva_respuesta not in respuestas_dadas:
                respuestas_dadas.append(nueva_respuesta)
                return nueva_respuesta, respuestas_dadas
            else:
                return None, respuestas_dadas
        break
    return None, respuestas_dadas

# Define la función que se ejecuta al hacer clic en el botón
@app.callback(
    Output('respuesta', 'children'),
    [Input('boton', 'n_clicks')],
    [State('texto', 'value')]
)
def actualizar_respuesta(n_clicks, pregunta):
    global respuestas_dadas
    if pregunta:
        df = procesar_pregunta(pregunta)
        df1 =Articulos_relacionados(df)
        for i, row in df.iterrows():
            contexto = row["Contenido_Articulo"]
            print(len(contexto))
            respuesta, respuestas_dadas_nueva = hacer_pregunta(contexto, pregunta, respuestas_dadas)
            if respuesta:
                # Genera un resumen con ChatGPT
                resumen = openai.Completion.create(
                    engine="text-davinci-002",
                    prompt=f"Resumen del texto: {pregunta}",
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )
                # Obtiene la respuesta del resumen generado por ChatGPT
                for choice in resumen.choices:
                    if "Resumen del texto:" not in choice.text:
                        resumen_generado = choice.text.strip()
                        break
                # Muestra el resumen y la respuesta
                respuesta_split = respuesta.split('Pregunta:')
                primeros_tres = " ".join(respuesta_split[:2])
                #print(respuesta_split)
                respuestas_dadas = respuestas_dadas_nueva # actualiza la variable global
                return html.Div([                
                    html.P(resumen_generado, style={'font-weight': 'bold'}),
                    html.P(primeros_tres, style={'color': 'Black'}),
                    html.Div([
                            html.H2("Artículos relacionados", style=estilo_titulo),
                            dash_table.DataTable(
                                id='tabla',
                                columns=[{"name": i, "id": i} for i in df1.columns],
                                data=df1.to_dict('records'),
                                style_cell=estilo_celda,
                                style_header=estilo_encabezado,
                                style_table=estilo_tabla,
                            )
                        ], style={'width': '100%', 'float': 'right'})
                ])                
            else:
                respuestas_dadas = respuestas_dadas_nueva # actualiza la variable global
                continue # pasa al siguiente row
        return html.P("No se encontró ninguna respuesta.", style={'color': 'red'})

# Ejecuta la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
