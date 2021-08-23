import spacy
import datetime
import os
import regex as re
import pandas as pd
import numpy as np

ROOT = os.path.abspath('/content/drive/MyDrive/Datos - Hackathon JusticIA/')

df_trans = pd.read_csv(os.path.join(ROOT, 'JusticIA_DatosTranscripciones.csv'))

##Se agregan más columnas porque hay entradas con más de un valor válido
df_orgs = pd.read_csv(os.path.join(ROOT, 'organizations.csv'), names=[f'N{i}' for i in range(3)])
df_lugrs = pd.read_csv(os.path.join(ROOT, 'places.csv'), names=[f'N{i}' for i in range(3)])

lugrs = df_lugrs.iloc[1:].sort_values('N0')
orgs = df_orgs.iloc[1:].sort_values('N0')

no_deseadas='expe,h 147,l 10'.split(',')


def obtener_texto_idx(idx:int, df=None):
    """
    Devuelve el texto disponible en df_trans accesible por
    su índice. 
    """
    if df is None:
        df = df_trans
    return df.iloc[idx].Texto

def limpiar_1(texto:str):
    """
    Se hace una primera limpieza de los datos. Eliminando saltos de línea,
    y espacios innecesarios.
    """
    texto = texto.replace('\n', '')
    return texto

def corregir_brs(texto):
    return re.sub(r'([A-Z]+)[-—\s]*\n[-—\s]*([A-Z]+)', 
                   r'\1\2',
                   texto)

def extraer_mayus(texto):
    return [f[0].strip().title() for f in re.findall(r'(([\p{Lu}\.]{2,}\s?)+)', texto)]


def extraer_fechas(texto, fmt=None):
    mes_patts = 'ener?o?,febr?e?r?o?,marz?o?,abri?l?,may?o?,juni?o?,'\
                'juli?o?,ago?s?t?o?,sept?i?e?m?b?r?e?,octu?b?r?e?,'\
                'novi?e?m?b?r?e?,dici?e?m?b?r?e?'.split(',')

    fechas_1 = [re.split(r'[\-.\/_— =]+', f) for f in
                   re.findall(r'(\d{1,2}[\-.\/_— =]+[A-Za-z]{3,}[ \-.\/_—=]+\d{2,4})',
                             texto)\
                if 'exp' not in f.lower()
                ]  
    fechas_2 = [re.split(r'[\-.\/_— ]+', f) for f in
                            re.findall(r'([A-Za-z]{3,}[ \-.\/_—=]+\d{1,2}[ \-.\/_—=]+\d{1,2})',
                                      texto)\
                if 'exp' not in f.lower()
                ]

    fechas_strs = [[f[2],f[1],f[0]] for f in fechas_1 if len(f)==3] +\
                  [[f[2],f[0],f[1]] for f in fechas_2 if len(f)==3]
    
    fechas_dt=[]
    for a,m,d in fechas_strs:
        for i, k in enumerate(mes_patts):
            patt = re.compile(k)
            match = re.match(patt, m.lower())
            if match:
                try:
                    fechas_dt.append(datetime.date(int(a)+1900,
                                                i+1,
                                                int(d)))
                except:
                    pass
                break

    if fmt is not None:
        return [f.strftime(fmt) for f in fechas_dt]
    else:
        return fechas_dt

def extraer_nombres(texto):
    posibles_nombres = [nlp(n.lower()) for n in extraer_mayus(texto)]
    nombres=[]
    for n in posibles_nombres:
        nombres_valid = [ent.text for ent in n.ents if ent.label_=='PER']
        nombres += nombres_valid
    return [n.title() for n in nombres if len(n)>5]

def buscar(p, df, devolver_idxs=False):
    cols = df.columns.tolist()
    p = re.sub(r'[,.-\/\\]+', ' ', p)
    idxs = np.array([df[s].str.contains(p, na=False, regex=True).tolist() for s in cols]).T
    return np.where(idxs)[0] if devolver_idxs else np.any(idxs)

    
def extraer_datos(df):
    assert 'Texto' in df.columns.tolist()

    nlp = spacy.load('es_core_news_sm')
    res_df = None
    for idx in range(df.shape[0]):
        filename = df.iloc[idx].NombreArchivo
        texto = obtener_texto_idx(idx, df)
        texto = corregir_brs(texto)
        fechas = extraer_fechas(texto, '%d-%m-%Y')
        nombres = extraer_nombres(texto)

        doc = nlp(texto.lower())
        lugares = []
        organizaciones = []
        for ent in doc.ents:
            p = re.sub(r'[,\.-\/\\]+', '', ent.text)
            if ent.label_=='LOC' and len(p)>4 and p not in no_deseadas and buscar(p, lugrs):
                lugares.append(p)
            elif ent.label_=='ORG' and buscar(p, orgs):
                organizaciones.append(p)
        
        n_tot = len(fechas) + len(nombres) + len(lugares) + len(organizaciones)
        new_df = pd.DataFrame({'filename':[filename]*n_tot,
                               'label': fechas+nombres+lugares+organizaciones,
                               'class': ['Fecha']*len(fechas) +\
                                        ['Persona']*len(nombres) +\
                                        ['Lugar']*len(lugares) +\
                                        ['Organización']*len(organizaciones)})
        
        if res_df is None:
            res_df = new_df
        else:
            res_df = res_df.append(new_df, ignore_index=True)

    return res_df


if __name__ == '__main__':
    df = df_trans[df_trans.MetodoTexto=='automatico'][:50]

    res = extraer_datos(df)
    res.head(100)