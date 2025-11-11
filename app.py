# app.py (Versão com Correção do Erro de Serialização)

import pandas as pd
import geopandas as gpd
from flask import Flask, jsonify, render_template, request
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# --- CARREGAMENTO E PROCESSAMENTO ---
print("Iniciando o carregamento e processamento dos dados...")
try:
    df_crimes = pd.read_csv('crimes.csv', sep=',')
    gdf_municipios_original = gpd.read_file('municipios_ce.geojson')
    df_populacao = pd.read_csv('populacao_ce.csv')
    df_crimes.columns = ['AIS', 'NATUREZA', 'MUNICIPIO', 'LOCAL', 'DATA', 'HORA', 'DIA_SEMANA','MEIO_EMPREGADO', 'GENERO', 'ORIENTACAO_SEXUAL', 'IDADE_VITIMA','ESCOLARIDADE_VITIMA', 'RACA_VITIMA']
except FileNotFoundError as e:
    print(f"ERRO CRÍTICO: Arquivo não encontrado - {e}.")
    exit()

def normalize_text(text_series):
    return text_series.str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

df_crimes['DATA'] = pd.to_datetime(df_crimes['DATA'], dayfirst=True, errors='coerce')
df_crimes['ANO'] = df_crimes['DATA'].dt.year
df_crimes['MES'] = df_crimes['DATA'].dt.month
mapeamento_genero = {'Masculino': 'Masculino', 'Homem Trans': 'Masculino', 'Feminino': 'Feminino', 'Mulher Trans': 'Feminino', 'Travesti': 'Feminino'}
df_crimes['GENERO_AGRUPADO'] = df_crimes['GENERO'].map(mapeamento_genero)
df_crimes['MUNICIPIO_NORM'] = normalize_text(df_crimes['MUNICIPIO'])
gdf_municipios = gdf_municipios_original.copy()
gdf_municipios['NM_MUN_NORM'] = normalize_text(gdf_municipios['name'])
df_populacao['MUNICIPIO_NORM'] = normalize_text(df_populacao['municipio'])
crimes_agrupados_mun = df_crimes.groupby(['MUNICIPIO_NORM', 'NATUREZA']).size().reset_index(name='QUANTIDADE')
crimes_com_pop_mun = pd.merge(crimes_agrupados_mun, df_populacao[['MUNICIPIO_NORM', 'populacao']], on='MUNICIPIO_NORM', how='left')
crimes_com_pop_mun.dropna(subset=['populacao'], inplace=True)
crimes_com_pop_mun['TAXA_POR_100K'] = (crimes_com_pop_mun['QUANTIDADE'] / crimes_com_pop_mun['populacao']) * 100000
municipios_ais_map = {'Fortaleza': 'AIS 1-10', 'Caucaia': 'AIS 11', 'Maracanaú': 'AIS 12', 'Aquiraz': 'AIS 13', 'Cascavel': 'AIS 13', 'Eusébio': 'AIS 13', 'Pindoretama': 'AIS 13', 'Alcântaras': 'AIS 14', 'Barroquinha': 'AIS 14', 'Camocim': 'AIS 14', 'Cariré': 'AIS 14', 'Carnaubal': 'AIS 14', 'Chaval': 'AIS 14', 'Coreaú': 'AIS 14', 'Croatá': 'AIS 14', 'Forquilha': 'AIS 14', 'Frecheirinha': 'AIS 14', 'Graça': 'AIS 14', 'Granja': 'AIS 14', 'Groaíras': 'AIS 14', 'Guaraciaba do Norte': 'AIS 14', 'Ibiapina': 'AIS 14', 'Martinópole': 'AIS 14', 'Massapê': 'AIS 14', 'Meruoca': 'AIS 14', 'Moraújo': 'AIS 14', 'Mucambo': 'AIS 14', 'Pacujá': 'AIS 14', 'Santana do Acaraú': 'AIS 14', 'São Benedito': 'AIS 14', 'Senador Sá': 'AIS 14', 'Sobral': 'AIS 14', 'Tianguá': 'AIS 14', 'Ubajara': 'AIS 14', 'Uruoca': 'AIS 14', 'Viçosa do Ceará': 'AIS 14', 'Acarape': 'AIS 15', 'Aracoiaba': 'AIS 15', 'Aratuba': 'AIS 15', 'Barreira': 'AIS 15', 'Baturité': 'AIS 15', 'Boa Viagem': 'AIS 15', 'Canindé': 'AIS 15', 'Capistrano': 'AIS 15', 'Caridade': 'AIS 15', 'Guaramiranga': 'AIS 15', 'Itapiúna': 'AIS 15', 'Itatira': 'AIS 15', 'Madalena': 'AIS 15', 'Mulungu': 'AIS 15', 'Ocara': 'AIS 15', 'Pacoti': 'AIS 15', 'Palmácia': 'AIS 15', 'Paramoti': 'AIS 15', 'Redenção': 'AIS 15', 'Ararendá': 'AIS 16', 'Catunda': 'AIS 16', 'Crateús': 'AIS 16', 'Hidrolândia': 'AIS 16', 'Independência': 'AIS 16', 'Ipaporanga': 'AIS 16', 'Ipu': 'AIS 16', 'Ipueiras': 'AIS 16', 'Monsenhor Tabosa': 'AIS 16', 'Nova Russas': 'AIS 16', 'Novo Oriente': 'AIS 16', 'Pires Ferreira': 'AIS 16', 'Poranga': 'AIS 16', 'Reriutaba': 'AIS 16', 'Santa Quitéria': 'AIS 16', 'Tamboril': 'AIS 16', 'Varjota': 'AIS 16', 'Acaraú': 'AIS 17', 'Amontada': 'AIS 17', 'Apuiarés': 'AIS 17', 'Bela Cruz': 'AIS 17', 'Cruz': 'AIS 17', 'General Sampaio': 'AIS 17', 'Irauçuba': 'AIS 17', 'Itapajé': 'AIS 17', 'Itapipoca': 'AIS 17', 'Itarema': 'AIS 17', 'Jijoca de Jericoacoara': 'AIS 17', 'Marco': 'AIS 17', 'Miraíma': 'AIS 17', 'Morrinhos': 'AIS 17', 'Pentecoste': 'AIS 17', 'Tejuçuoca': 'AIS 17', 'Tururu': 'AIS 17', 'Umirim': 'AIS 17', 'Uruburetama': 'AIS 17', 'Alto Santo': 'AIS 18', 'Aracati': 'AIS 18', 'Beberibe': 'AIS 18', 'Ererê': 'AIS 18', 'Fortim': 'AIS 18', 'Icapuí': 'AIS 18', 'Iracema': 'AIS 18', 'Itaiçaba': 'AIS 18', 'Jaguaribe': 'AIS 18', 'Jaguaruana': 'AIS 18', 'Limoeiro do Norte': 'AIS 18', 'Jaguaribara': 'AIS 18', 'Palhano': 'AIS 18', 'Pereiro': 'AIS 18', 'Potiretama': 'AIS 18', 'Quixeré': 'AIS 18', 'Russas': 'AIS 18', 'São João do Jaguaribe': 'AIS 18', 'Tabuleiro do Norte': 'AIS 18', 'Abaiara': 'AIS 19', 'Altaneira': 'AIS 19', 'Antonina do Norte': 'AIS 19', 'Araripe': 'AIS 19', 'Assaré': 'AIS 19', 'Aurora': 'AIS 19', 'Barbalha': 'AIS 19', 'Barro': 'AIS 19', 'Brejo Santo': 'AIS 19', 'Campos Sales': 'AIS 19', 'Caririaçu': 'AIS 19', 'Crato': 'AIS 19', 'Farias Brito': 'AIS 19', 'Jardim': 'AIS 19', 'Jati': 'AIS 19', 'Juazeiro do Norte': 'AIS 19', 'Mauriti': 'AIS 19', 'Milagres': 'AIS 19', 'Missão Velha': 'AIS 19', 'Nova Olinda': 'AIS 19', 'Penaforte': 'AIS 19', 'Porteiras': 'AIS 19', 'Potengi': 'AIS 19', 'Salitre': 'AIS 19', 'Santana do Cariri': 'AIS 19', 'Banabuiú': 'AIS 20', 'Choró': 'AIS 20', 'Deputado Irapuan Pinheiro': 'AIS 20', 'Ibaretama': 'AIS 20', 'Ibicuitinga': 'AIS 20', 'Jaguaretama': 'AIS 20', 'Milhã': 'AIS 20', 'Morada Nova': 'AIS 20', 'Pedra Branca': 'AIS 20', 'Quixadá': 'AIS 20', 'Quixeramobim': 'AIS 20', 'Senador Pompeu': 'AIS 20', 'Solonópole': 'AIS 20', 'Acopiara': 'AIS 21', 'Baixio': 'AIS 21', 'Cariús': 'AIS 21', 'Cedro': 'AIS 21', 'Granjeiro': 'AIS 21', 'Icó': 'AIS 21', 'Iguatu': 'AIS 21', 'Ipaumirim': 'AIS 21', 'Jucás': 'AIS 21', 'Lavras da Mangabeira': 'AIS 21', 'Orós': 'AIS 21', 'Quixelô': 'AIS 21', 'Saboeiro': 'AIS 21', 'Tarrafas': 'AIS 21', 'Umari': 'AIS 21', 'Várzea Alegre': 'AIS 21', 'Aiuaba': 'AIS 22', 'Arneiroz': 'AIS 22', 'Catarina': 'AIS 22', 'Mombaça': 'AIS 22', 'Parambu': 'AIS 22', 'Piquet Carneiro': 'AIS 22', 'Quiterianópolis': 'AIS 22', 'Tauá': 'AIS 22', 'Paracuru': 'AIS 23', 'Paraipaba': 'AIS 23', 'São Gonçalo do Amarante': 'AIS 23', 'São Luís do Curu': 'AIS 23', 'Trairi': 'AIS 23', 'Guaiúba': 'AIS 24', 'Maranguape': 'AIS 24', 'Pacatuba': 'AIS 24', 'Chorozinho': 'AIS 25', 'Horizonte': 'AIS 25', 'Itaitinga': 'AIS 25', 'Pacajus': 'AIS 25'}
gdf_municipios['AIS'] = gdf_municipios['name'].map(municipios_ais_map)
gdf_ais = gdf_municipios.dissolve(by='AIS').reset_index()
df_crimes['AIS_MAPEADA'] = df_crimes['MUNICIPIO'].map(municipios_ais_map)
df_populacao['AIS'] = df_populacao['municipio'].map(municipios_ais_map)
pop_por_ais = df_populacao.groupby('AIS')['populacao'].sum().reset_index()
crimes_agrupados_ais = df_crimes.groupby(['AIS_MAPEADA', 'NATUREZA']).size().reset_index(name='QUANTIDADE')
crimes_com_pop_ais = pd.merge(crimes_agrupados_ais, pop_por_ais, left_on='AIS_MAPEADA', right_on='AIS', how='left')
crimes_com_pop_ais.dropna(subset=['populacao'], inplace=True)
crimes_com_pop_ais['TAXA_POR_100K'] = (crimes_com_pop_ais['QUANTIDADE'] / crimes_com_pop_ais['populacao']) * 100000
LISTA_DE_CRIMES = sorted(df_crimes['NATUREZA'].dropna().unique().tolist())
print("Processamento de dados concluído. Aplicação pronta.")

def projetar_ano_incompleto(df_historico, ano_incompleto, ultimo_mes_registrado, colunas_grupo):
    df_ano_incompleto = df_historico[df_historico['ANO'] == ano_incompleto].copy()
    df_anos_anteriores = df_historico[df_historico['ANO'] < ano_incompleto].copy()
    if df_anos_anteriores.empty:
        return df_ano_incompleto.groupby(colunas_grupo)['TOTAL'].sum().reset_index()
    periodo_faltante = df_anos_anteriores[df_anos_anteriores['MES'] > ultimo_mes_registrado]
    soma_periodo_faltante_por_ano = periodo_faltante.groupby(['ANO'] + colunas_grupo)['TOTAL'].sum().reset_index()
    media_periodo_faltante = soma_periodo_faltante_por_ano.groupby(colunas_grupo)['TOTAL'].mean().reset_index()
    media_periodo_faltante.rename(columns={'TOTAL': 'ESTIMATIVA_FALTANTE'}, inplace=True)
    total_ano_incompleto = df_ano_incompleto.groupby(colunas_grupo)['TOTAL'].sum().reset_index()
    df_projetado = pd.merge(total_ano_incompleto, media_periodo_faltante, on=colunas_grupo, how='left')
    df_projetado['ESTIMATIVA_FALTANTE'] = df_projetado['ESTIMATIVA_FALTANTE'].fillna(0)
    df_projetado['TOTAL'] = df_projetado['TOTAL'] + df_projetado['ESTIMATIVA_FALTANTE']
    return df_projetado[colunas_grupo + ['TOTAL']]

def get_filtered_df():
    genero_filtro = request.args.get('genero')
    df_filtrado = df_crimes.copy()
    if genero_filtro == 'feminino':
        df_filtrado = df_filtrado[df_filtrado['GENERO_AGRUPADO'] == 'Feminino'].copy()
    return df_filtrado

@app.route('/')
def index():
    return render_template('index.html', crimes=LISTA_DE_CRIMES)

@app.route('/api/municipality_map_data/<crime_type>')
def get_municipality_map_data(crime_type):
    dados_do_crime = crimes_com_pop_mun[crimes_com_pop_mun['NATUREZA'] == crime_type]
    mapa_completo = gdf_municipios.merge(dados_do_crime[['MUNICIPIO_NORM', 'QUANTIDADE', 'TAXA_POR_100K']], left_on='NM_MUN_NORM', right_on='MUNICIPIO_NORM', how='left')
    mapa_completo[['QUANTIDADE', 'TAXA_POR_100K']] = mapa_completo[['QUANTIDADE', 'TAXA_POR_100K']].fillna(0)
    max_taxa = mapa_completo['TAXA_POR_100K'].max()
    return jsonify({'geojson': json.loads(mapa_completo.to_json()), 'max_taxa': max_taxa})

@app.route('/api/ais_map_data/<crime_type>')
def get_ais_map_data(crime_type):
    dados_do_crime = crimes_com_pop_ais[crimes_com_pop_ais['NATUREZA'] == crime_type]
    mapa_completo = gdf_ais.merge(dados_do_crime[['AIS_MAPEADA', 'QUANTIDADE', 'TAXA_POR_100K']], left_on='AIS', right_on='AIS_MAPEADA', how='left')
    mapa_completo[['QUANTIDADE', 'TAXA_POR_100K']] = mapa_completo[['QUANTIDADE', 'TAXA_POR_100K']].fillna(0)
    max_taxa = mapa_completo['TAXA_POR_100K'].max()
    return jsonify({'geojson': json.loads(mapa_completo.to_json()), 'max_taxa': max_taxa})

# CORREÇÃO AQUI: A rota de municípios agora usa uma cópia para não contaminar o gdf principal
@app.route('/api/municipalities')
def get_municipalities():
    """Retorna uma lista de municípios com seus nomes e coordenadas centrais."""
    # Usa uma cópia para evitar modificar o GeoDataFrame original
    gdf_temp = gdf_municipios.copy()
    
    if gdf_temp.crs is None:
        gdf_temp.set_crs("EPSG:4326", inplace=True)

    gdf_temp['centroid'] = gdf_temp['geometry'].centroid
    
    municipalities_list = []
    for index, row in gdf_temp.iterrows():
        municipalities_list.append({
            'name': row['name'],
            'lat': row['centroid'].y,
            'lon': row['centroid'].x
        })
        
    return jsonify(municipalities_list)

def get_clean_age_df(df_base):
    df_idade = df_base.copy()
    df_idade['IDADE_NUM'] = pd.to_numeric(df_idade['IDADE_VITIMA'], errors='coerce')
    df_idade.dropna(subset=['IDADE_NUM'], inplace=True)
    df_idade['IDADE_NUM'] = df_idade['IDADE_NUM'].astype(int)
    df_idade = df_idade[(df_idade['IDADE_NUM'] >= 0) & (df_idade['IDADE_NUM'] <= 110)]
    return df_idade

@app.route('/api/data/grafico_densidade_etaria')
def get_data_grafico_densidade_etaria():
    df_filtrado = get_filtered_df()
    df_idade = get_clean_age_df(df_filtrado)
    if df_idade.empty:
        return jsonify({'labels': [], 'datasets': []})
    contagem_por_idade = df_idade.groupby('IDADE_NUM').size().reindex(range(111), fill_value=0)
    return jsonify({'labels': contagem_por_idade.index.tolist(), 'datasets': [{'label': 'Número de Vítimas', 'data': contagem_por_idade.values.tolist(), 'borderColor': 'rgba(75, 192, 192, 1)', 'backgroundColor': 'rgba(75, 192, 192, 0.5)', 'fill': True, 'tension': 0.4}]})

@app.route('/api/data/grafico_comparativo_idade_genero')
def get_data_grafico_5b():
    df_filtrado = get_filtered_df()
    df_idade = get_clean_age_df(df_filtrado)
    if df_idade.empty or 'GENERO_AGRUPADO' not in df_idade.columns:
        return jsonify({'labels': [], 'datasets': []})
    df_idade.dropna(subset=['GENERO_AGRUPADO'], inplace=True)
    dados_grafico = df_idade.groupby(['IDADE_NUM', 'GENERO_AGRUPADO']).size().unstack(fill_value=0)
    if 'Masculino' not in dados_grafico: dados_grafico['Masculino'] = 0
    if 'Feminino' not in dados_grafico: dados_grafico['Feminino'] = 0
    dados_grafico = dados_grafico.reindex(range(111), fill_value=0)
    return jsonify({'labels': dados_grafico.index.tolist(), 'datasets': [{'label': 'Masculino', 'data': dados_grafico['Masculino'].tolist(), 'borderColor': 'rgba(54, 162, 235, 1)', 'backgroundColor': 'rgba(54, 162, 235, 0.5)', 'fill': True, 'tension': 0.4}, {'label': 'Feminino', 'data': dados_grafico['Feminino'].tolist(), 'borderColor': 'rgba(255, 99, 132, 1)', 'backgroundColor': 'rgba(255, 99, 132, 0.5)', 'fill': True, 'tension': 0.4}]})

@app.route('/api/data/grafico_evolucao_anual')
def get_data_grafico_evolucao_anual():
    df_filtrado = get_filtered_df()
    try:
        predict_years = int(request.args.get('predict', 0))
    except (ValueError, TypeError):
        predict_years = 0
    df_analise = df_filtrado.dropna(subset=['ANO', 'MES', 'GENERO_AGRUPADO'])
    df_analise['ANO'] = df_analise['ANO'].astype(int)
    dados_agrupados = df_analise.groupby(['ANO', 'MES', 'GENERO_AGRUPADO']).size().reset_index(name='TOTAL')
    ano_maximo = int(df_analise['ANO'].max())
    hoje = datetime.now()
    is_projected = False
    if ano_maximo == hoje.year and hoje.month < 12:
        is_projected = True
        ultimo_mes = int(df_analise[df_analise['ANO'] == ano_maximo]['MES'].max())
        df_projetado = projetar_ano_incompleto(dados_agrupados, ano_maximo, ultimo_mes, ['GENERO_AGRUPADO'])
        dados_finais = dados_agrupados[dados_agrupados['ANO'] < ano_maximo].groupby(['ANO', 'GENERO_AGRUPADO'])['TOTAL'].sum().reset_index()
        df_projetado['ANO'] = ano_maximo
        dados_finais = pd.concat([dados_finais, df_projetado], ignore_index=True)
    else:
        dados_finais = dados_agrupados.groupby(['ANO', 'GENERO_AGRUPADO'])['TOTAL'].sum().reset_index()
    dados_pivot = dados_finais.pivot(index='ANO', columns='GENERO_AGRUPADO', values='TOTAL').fillna(0)
    labels = [str(int(ano)) for ano in dados_pivot.index.tolist()]
    if is_projected:
        labels[-1] = f"{labels[-1]} (Projetado)"
    datasets = [
        {'label': 'Masculino', 'data': dados_pivot['Masculino'].tolist() if 'Masculino' in dados_pivot else [], 'borderColor': 'rgba(54, 162, 235, 1)'},
        {'label': 'Feminino', 'data': dados_pivot['Feminino'].tolist() if 'Feminino' in dados_pivot else [], 'borderColor': 'rgba(255, 99, 132, 1)'}
    ]
    if predict_years > 0 and not dados_pivot.empty:
        anos_historicos = dados_pivot.index.values.reshape(-1, 1)
        anos_futuros = np.arange(ano_maximo + 1, ano_maximo + 1 + predict_years).reshape(-1, 1)
        if 'Masculino' in dados_pivot:
            model_masc = LinearRegression().fit(anos_historicos, dados_pivot['Masculino'].values)
            previsao_masc = model_masc.predict(anos_futuros).round(0).astype(int)
            datasets[0]['data'].extend(previsao_masc.tolist())
        if 'Feminino' in dados_pivot:
            model_fem = LinearRegression().fit(anos_historicos, dados_pivot['Feminino'].values)
            previsao_fem = model_fem.predict(anos_futuros).round(0).astype(int)
            datasets[1]['data'].extend(previsao_fem.tolist())
        for ano in anos_futuros.flatten():
            labels.append(f"{ano} (Previsão)")
    return jsonify({'labels': labels, 'datasets': datasets})

@app.route('/api/data/grafico_distribuicao_raca')
def get_data_grafico_4():
    df_filtrado = get_filtered_df()
    df_raca = df_filtrado[df_filtrado['RACA_VITIMA'] != 'NÃO INFORMADA'].copy()
    if df_raca.empty: return jsonify({'labels': [], 'datasets': []})
    contagem_raca = df_raca['RACA_VITIMA'].value_counts()
    return jsonify({'labels': contagem_raca.index.tolist(), 'datasets': [{'label': 'Total de Vítimas', 'data': contagem_raca.values.tolist(), 'backgroundColor': ['#440154', '#482878', '#3e4a89', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6dcd59', '#b4de2c', '#fde725']}]})

@app.route('/api/data/grafico_crimes_mulher_dia_hora')
def get_data_grafico_crimes_mulher_dia_hora():
    df_feminino = df_crimes[df_crimes['GENERO_AGRUPADO'] == 'Feminino'].copy()
    if df_feminino.empty: return jsonify({'labels': [], 'datasets': []})
    df_feminino['HORA_NUM'] = pd.to_datetime(df_feminino['HORA'], format='%H:%M:%S', errors='coerce').dt.hour
    df_feminino.dropna(subset=['HORA_NUM'], inplace=True)
    df_feminino['HORA_NUM'] = df_feminino['HORA_NUM'].astype(int)
    ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    df_feminino['DIA_SEMANA'] = pd.Categorical(df_feminino['DIA_SEMANA'], categories=ordem_dias, ordered=True)
    crimes_por_dia_hora = df_feminino.groupby(['DIA_SEMANA', 'HORA_NUM']).size().unstack(fill_value=0)
    datasets = []
    cores = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF']
    for i, dia in enumerate(ordem_dias):
        if dia in crimes_por_dia_hora.index:
            datasets.append({'label': dia, 'data': crimes_por_dia_hora.loc[dia].reindex(range(24), fill_value=0).tolist(), 'borderColor': cores[i], 'fill': False, 'tension': 0.1})
    return jsonify({'labels': list(range(24)), 'datasets': datasets})

@app.route('/api/data/grafico_comparativo_crime_log')
def get_data_grafico_3a():
    df_analise = df_crimes.dropna(subset=['GENERO_AGRUPADO', 'NATUREZA'])
    crimes_por_natureza = df_analise.groupby(['NATUREZA', 'GENERO_AGRUPADO']).size().unstack(fill_value=0)
    crimes_por_natureza['TOTAL'] = crimes_por_natureza.sum(axis=1)
    crimes_por_natureza = crimes_por_natureza.sort_values('TOTAL', ascending=False)
    return jsonify({'labels': crimes_por_natureza.index.tolist(), 'datasets': [{'label': 'Masculino', 'data': crimes_por_natureza['Masculino'].tolist(), 'backgroundColor': 'rgba(54, 162, 235, 0.7)'}, {'label': 'Feminino', 'data': crimes_por_natureza['Feminino'].tolist(), 'backgroundColor': 'rgba(255, 99, 132, 0.7)'}]})

@app.route('/api/data/grafico_proporcao_genero_crime')
def get_data_grafico_3b():
    df_analise = df_crimes.dropna(subset=['GENERO_AGRUPADO', 'NATUREZA'])
    crimes_por_natureza = df_analise.groupby(['NATUREZA', 'GENERO_AGRUPADO']).size().reset_index(name='QUANTIDADE')
    total_por_natureza = crimes_por_natureza.groupby('NATUREZA')['QUANTIDADE'].transform('sum')
    crimes_por_natureza['PROPORCAO_%'] = (crimes_por_natureza['QUANTIDADE'] / total_por_natureza) * 100
    df_pivot = crimes_por_natureza.pivot(index='NATUREZA', columns='GENERO_AGRUPADO', values='PROPORCAO_%').fillna(0)
    df_pivot = df_pivot.sort_values('Feminino', ascending=True)
    return jsonify({'labels': df_pivot.index.tolist(), 'datasets': [{'label': 'Masculino', 'data': df_pivot['Masculino'].tolist(), 'backgroundColor': 'rgba(54, 162, 235, 0.7)'}, {'label': 'Feminino', 'data': df_pivot['Feminino'].tolist(), 'backgroundColor': 'rgba(255, 99, 132, 0.7)'}]})

def get_meio_empregado_df(filtro_genero='todos'):
    df_filtrado = df_crimes.copy()
    if filtro_genero == 'feminino':
        df_filtrado = df_crimes[df_crimes['GENERO_AGRUPADO'] == 'Feminino'].copy()
    df_meio = df_filtrado.dropna(subset=['MEIO_EMPREGADO'])
    df_meio = df_meio[df_meio['MEIO_EMPREGADO'] != 'NÃO INFORMADO'].copy()
    return df_meio

@app.route('/api/data/grafico_proporcao_meio_empregado')
def get_data_grafico_6a():
    """ Prepara dados para o Gráfico 6a: Proporção do Meio Empregado (Pizza). """
    filtro = request.args.get('filtro', 'todos') # Pega o parâmetro 'filtro' da URL
    df_meio = get_meio_empregado_df(filtro)
    
    if df_meio.empty: return jsonify({'labels': [], 'datasets': []})
        
    contagem_meio = df_meio['MEIO_EMPREGADO'].value_counts()
    
    output = {
        'labels': contagem_meio.index.tolist(),
        'datasets': [{
            'label': 'Meio Empregado',
            'data': contagem_meio.values.tolist(),
            'backgroundColor': ['#440154', '#482878', '#3e4a89', '#31688e', '#26828e', '#1f9e89', '#35b779', '#6dcd59', '#b4de2c', '#fde725']
        }]
    }
    return jsonify(output)

@app.route('/api/data/grafico_evolucao_meio_empregado')
def get_data_grafico_6b():
    """ Prepara dados para o Gráfico 6b: Evolução do Meio Empregado (Linhas). """
    filtro = request.args.get('filtro', 'todos')
    df_meio = get_meio_empregado_df(filtro)
    
    if df_meio.empty: return jsonify({'labels': [], 'datasets': []})
    
    df_meio = df_meio.dropna(subset=['ANO'])
    df_meio['ANO'] = df_meio['ANO'].astype(int)
    
    evolucao_meio = df_meio.groupby(['ANO', 'MEIO_EMPREGADO']).size().unstack(fill_value=0)
    
    datasets = []
    cores = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF', '#440154', '#21908d', '#fde725']
    
    for i, meio in enumerate(evolucao_meio.columns):
        datasets.append({
            'label': meio,
            'data': evolucao_meio[meio].tolist(),
            'borderColor': cores[i % len(cores)],
            'fill': False,
            'tension': 0.1
        })
        
    output = {
        'labels': [str(int(ano)) for ano in evolucao_meio.index.tolist()],
        'datasets': datasets
    }
    return jsonify(output)

@app.route('/api/data/grafico_evolucao_odio')
def get_data_grafico_7a():
    df_filtrado = get_filtered_df()
    naturezas_odio = ['CONDUTA HOMOFOBICA', 'CONDUTA TRANSFOBICA', 'PRECONCEITO RAÇA OU COR']
    df_odio = df_filtrado[df_filtrado['NATUREZA'].isin(naturezas_odio)].copy()
    if df_odio.empty or 'ANO' not in df_odio.columns:
        return jsonify({'labels': [], 'datasets': []})
    df_odio = df_odio.dropna(subset=['ANO'])
    df_odio['ANO'] = df_odio['ANO'].astype(int)
    evolucao_odio = df_odio.groupby(['ANO', 'NATUREZA']).size().unstack(fill_value=0)
    datasets = []
    cores = {'CONDUTA HOMOFOBICA': '#E41A1C', 'CONDUTA TRANSFOBICA': '#377EB8', 'PRECONCEITO RAÇA OU COR': '#4DAF4A'}
    for natureza in naturezas_odio:
        if natureza in evolucao_odio.columns:
            datasets.append({
                'label': natureza,
                'data': evolucao_odio[natureza].tolist(),
                'borderColor': cores.get(natureza, '#000000'),
                'fill': False,
                'tension': 0.1
            })
    return jsonify({'labels': [str(int(ano)) for ano in evolucao_odio.index.tolist()], 'datasets': datasets})

@app.route('/api/data/grafico_perfil_orientacao_sexual')
def get_data_grafico_7b():
    df_filtrado = get_filtered_df()
    naturezas_lgbtfobia = ['CONDUTA HOMOFOBICA', 'CONDUTA TRANSFOBICA']
    df_lgbtfobia = df_filtrado[df_filtrado['NATUREZA'].isin(naturezas_lgbtfobia)].copy()
    df_lgbtfobia = df_lgbtfobia[df_lgbtfobia['ORIENTACAO_SEXUAL'] != 'NÃO INFORMADO'].copy()
    if df_lgbtfobia.empty:
        return jsonify({'labels': [], 'datasets': []})
    contagem_orientacao = df_lgbtfobia['ORIENTACAO_SEXUAL'].value_counts()
    return jsonify({
        'labels': contagem_orientacao.index.tolist(),
        'datasets': [{
            'label': 'Total de Vítimas',
            'data': contagem_orientacao.values.tolist(),
            'backgroundColor': ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
        }]
    })

# BLOCO FINAL PARA INICIAR O SERVIDOR
if __name__ == '__main__':
    app.run(debug=True)