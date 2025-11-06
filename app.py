# app.py
from flask import Flask, render_template, jsonify, request
import pandas as pd
import geopandas as gpd
import numpy as np # Importamos numpy para manipulação numérica

# --------------------------------------------------------------------------
# 1. INICIALIZAÇÃO E CARREGAMENTO DE DADOS (sem alterações)
# --------------------------------------------------------------------------
app = Flask(__name__)

def carregar_e_preparar_dados():
    try:
        df_crimes = pd.read_csv('crimes.csv', sep=',')
        gdf_municipios = gpd.read_file('municipios_ce.geojson')
        df_populacao = pd.read_csv('populacao_ce.csv')
    except FileNotFoundError as e:
        print(f"ERRO CRÍTICO: Arquivo '{e.filename}' não encontrado.")
        return None, None, None

    df_crimes.columns = [
        'AIS', 'NATUREZA', 'MUNICIPIO', 'LOCAL', 'DATA', 'HORA', 'DIA_SEMANA',
        'MEIO_EMPREGADO', 'GENERO', 'ORIENTACAO_SEXUAL', 'IDADE_VITIMA',
        'ESCOLARIDADE_VITIMA', 'RACA_VITIMA'
    ]
    df_crimes['DATA'] = pd.to_datetime(df_crimes['DATA'], dayfirst=True, errors='coerce')
    df_crimes.dropna(subset=['DATA'], inplace=True)
    df_crimes['ANO'] = df_crimes['DATA'].dt.year
    
    def normalize_text(text_series):
        return text_series.str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    df_crimes['MUNICIPIO_NORM'] = normalize_text(df_crimes['MUNICIPIO'])
    gdf_municipios['NM_MUN_NORM'] = normalize_text(gdf_municipios['name'])
    df_populacao['MUNICIPIO_NORM'] = normalize_text(df_populacao['municipio'])
    
    print("Dados carregados e pré-processados com sucesso.")
    return df_crimes, gdf_municipios, df_populacao

df_crimes_global, gdf_municipios_global, df_populacao_global = carregar_e_preparar_dados()

# --------------------------------------------------------------------------
# 2. DEFINIÇÃO DAS ROTAS (ENDPOINTS) DA APLICAÇÃO
# --------------------------------------------------------------------------

@app.route('/')
def index():
    lista_crimes = sorted(df_crimes_global['NATUREZA'].dropna().unique().tolist())
    return render_template('index.html', crimes=lista_crimes)

@app.route('/api/dados_mapa')
def get_map_data():
    """
    Endpoint de API que processa os dados e atribui a cor correta para cada município.
    """
    crime_selecionado = request.args.get('crime', 'HOMICIDIO DOLOSO')

    # --- LÓGICA DE CÁLCULO E COLORAÇÃO (REFEITA) ---

    # 1. Filtra e agrega os crimes
    df_filtrado = df_crimes_global[df_crimes_global['NATUREZA'] == crime_selecionado]
    crimes_por_municipio = df_filtrado.groupby('MUNICIPIO_NORM').size().reset_index(name='QUANTIDADE')

    # 2. Prepara dados de população e junta com os de crimes
    mapa_data = gdf_municipios_global.merge(
        df_populacao_global[['MUNICIPIO_NORM', 'populacao']],
        left_on='NM_MUN_NORM',
        right_on='MUNICIPIO_NORM',
        how='left'
    ).merge(
        crimes_por_municipio,
        on='MUNICIPIO_NORM',
        how='left'
    )

    # 3. Limpeza e cálculo da taxa
    mapa_data['QUANTIDADE'] = mapa_data['QUANTIDADE'].fillna(0).astype(int)
    mapa_data.dropna(subset=['populacao'], inplace=True) # Remove municípios sem população
    mapa_data = mapa_data[mapa_data['populacao'] > 0] # Remove municípios com população zero
    mapa_data['TAXA_POR_100K'] = (mapa_data['QUANTIDADE'] / mapa_data['populacao']) * 100000

    # 4. Lógica de Classificação de Cores (6 Níveis)
    cores = ['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026']
    
    # Cria 6 "bins" (faixas) usando quantis. Isso distribui os dados de forma equilibrada.
    # pd.qcut divide os dados em N grupos de tamanhos iguais.
    # O `labels=False` retorna um número de 0 a 5 para cada município, indicando o grupo.
    # `duplicates='drop'` lida com casos onde os limites dos quantis são iguais.
    mapa_data['COR_INDEX'] = pd.qcut(
        mapa_data['TAXA_POR_100K'], 
        q=len(cores), 
        labels=False, 
        duplicates='drop'
    )
    
    # Atribui a cor exata com base no índice do grupo
    mapa_data['COR'] = mapa_data['COR_INDEX'].apply(lambda x: cores[x])
    
    # Para municípios com taxa 0, garantimos a cor mais clara
    mapa_data.loc[mapa_data['TAXA_POR_100K'] == 0, 'COR'] = '#f2f2f2'

    # 5. Retorna o GeoJSON para o frontend
    return jsonify(mapa_data.to_json())


@app.route('/api/municipios')
def get_municipios():
    lista_municipios = sorted(gdf_municipios_global['name'].unique().tolist())
    return jsonify(lista_municipios)

# --------------------------------------------------------------------------
# 5. EXECUÇÃO DO SERVIDOR
# --------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
