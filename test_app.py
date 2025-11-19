# test_app.py

import pytest
from app import app as flask_app  # Importa a nossa aplicação Flask do arquivo app.py

# --- CONFIGURAÇÃO DO AMBIENTE DE TESTE ---

@pytest.fixture
def app():
    """Cria uma instância da nossa aplicação Flask para ser usada nos testes."""
    yield flask_app

@pytest.fixture
def client(app):
    """Cria um 'cliente' de teste. É como um navegador simulado que pode fazer requisições à nossa app."""
    return app.test_client()

# --- TESTES DAS ROTAS PRINCIPAIS E DE MAPA ---

def test_pagina_inicial(client):
    """Testa se a página inicial (/) carrega corretamente."""
    response = client.get('/')
    assert response.status_code == 200  # Verifica se a resposta foi "OK"
    assert b"Dashboard de An\xc3\xa1lise Criminal" in response.data # Verifica se o título está na página

def test_api_municipios(client):
    """Testa se a API que lista os municípios está funcionando."""
    response = client.get('/api/municipalities')
    assert response.status_code == 200
    json_data = response.get_json()
    assert isinstance(json_data, list)  # A resposta deve ser uma lista
    assert len(json_data) > 180  # O Ceará tem 184 municípios
    assert 'name' in json_data[0] and 'lat' in json_data[0] # Verifica se a estrutura está correta

def test_api_mapa_municipios(client):
    """Testa a API do mapa de municípios para um crime específico."""
    response = client.get('/api/municipality_map_data/HOMICIDIO DOLOSO')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'geojson' in json_data and 'max_taxa' in json_data # Verifica a estrutura da resposta

def test_api_mapa_ais(client):
    """Testa a API do mapa de AIS para um crime específico."""
    response = client.get('/api/ais_map_data/HOMICIDIO DOLOSO')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'geojson' in json_data and 'max_taxa' in json_data

# --- TESTES DAS ROTAS DE API DOS GRÁFICOS ---
# Usamos 'parametrize' para testar várias rotas com o mesmo código, o que é muito eficiente.

@pytest.mark.parametrize("endpoint", [
    "/api/data/grafico_evolucao_anual",
    "/api/data/grafico_comparativo_idade_genero",
    "/api/data/grafico_crimes_mulher_dia_hora",
    "/api/data/grafico_distribuicao_raca",
    "/api/data/grafico_densidade_etaria",
    "/api/data/grafico_comparativo_crime_log",
    "/api/data/grafico_proporcao_genero_crime",
    "/api/data/grafico_proporcao_meio_empregado",
    "/api/data/grafico_evolucao_meio_empregado",
    "/api/data/grafico_evolucao_odio",
    "/api/data/grafico_perfil_orientacao_sexual",
])
def test_apis_de_graficos_gerais(client, endpoint):
    """Testa todas as APIs de gráficos sem parâmetros."""
    response = client.get(endpoint)
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'labels' in json_data and 'datasets' in json_data # Toda API de gráfico deve retornar essa estrutura

@pytest.mark.parametrize("endpoint", [
    "/api/data/grafico_distribuicao_raca",
    "/api/data/grafico_densidade_etaria",
])
def test_apis_de_graficos_com_filtro_genero(client, endpoint):
    """Testa as APIs que aceitam o filtro de gênero."""
    response = client.get(f"{endpoint}?genero=feminino")
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'labels' in json_data and 'datasets' in json_data

def test_api_grafico_meio_empregado_com_filtro(client):
    """Testa a API de meio empregado com seu filtro específico."""
    response = client.get("/api/data/grafico_proporcao_meio_empregado?filtro=feminino")
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'labels' in json_data and 'datasets' in json_data

def test_api_grafico_evolucao_com_previsao(client):
    """Testa a API de evolução anual com o parâmetro de previsão."""
    response = client.get("/api/data/grafico_evolucao_anual?predict=5")
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'labels' in json_data and 'datasets' in json_data
    # Verifica se os labels de previsão foram adicionados
    assert any("(Previsão)" in label for label in json_data['labels'])

