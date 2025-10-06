# -*- coding: utf-8 -*-

"""
Análise Criminal no Ceará (v7.3) - Lendo Arquivo GeoJSON Local

Esta versão implementa a análise geográfica lendo diretamente o arquivo
'municipios_ce.geojson' fornecido pelo usuário, localizado na mesma pasta.
"""

# 1. IMPORTAÇÃO DAS BIBLIOTECAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
import requests
import os
from datetime import datetime

# 2. CONFIGURAÇÕES GLOBAIS
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
def projetar_ano_corrente(df, coluna_grupo):
    """
    Calcula uma projeção linear para o ano mais recente no DataFrame,
    se ele não estiver completo.
    
    Args:
        df (pd.DataFrame): DataFrame com colunas 'ANO' e a coluna a ser agrupada.
        coluna_grupo (str): O nome da coluna para agrupar junto com o ano (ex: 'GENERO_AGRUPADO').

    Returns:
        pd.DataFrame: DataFrame com o valor do último ano projetado.
    """
    df_analise = df.groupby(['ANO', coluna_grupo]).size().reset_index(name='TOTAL')
    
    ano_maximo = df['ANO'].max()
    hoje = datetime.now()

    # Só projeta se o ano máximo for o ano corrente (ou futuro, no nosso caso de teste)
    if ano_maximo >= hoje.year:
        print(f"\nAno {ano_maximo} está incompleto. Calculando projeção...")
        
        # Encontra o último dia com registro no ano
        ultimo_dia_registrado = df[df['ANO'] == ano_maximo]['DATA'].max().dayofyear
        
        # Filtra os dados do ano corrente
        dados_ano_corrente = df_analise[df_analise['ANO'] == ano_maximo].copy()
        
        # Calcula o fator de projeção
        fator_projecao = 365 / ultimo_dia_registrado
        print(f"Último dia com registro em {ano_maximo}: {ultimo_dia_registrado}. Fator de projeção: {fator_projecao:.2f}")
        
        # Aplica a projeção
        dados_ano_corrente['TOTAL_PROJETADO'] = dados_ano_corrente['TOTAL'] * fator_projecao
        
        # Atualiza o DataFrame principal com os valores projetados
        for index, row in dados_ano_corrente.iterrows():
            # Localiza a linha no df_analise original e atualiza o valor 'TOTAL'
            df_analise.loc[
                (df_analise['ANO'] == ano_maximo) & (df_analise[coluna_grupo] == row[coluna_grupo]),
                'TOTAL'
            ] = row['TOTAL_PROJETADO']
            
    return df_analise
# 3. FUNÇÕES DE GERAÇÃO DE GRÁFICOS (sem alterações)
def gerar_grafico_1(df, tipo_analise):
    """Gera o Gráfico 1: Evolução anual do número de crimes por gênero, com projeção."""
    print(f"\nGerando Gráfico 1: Evolução do Número Absoluto de '{tipo_analise}'...")
    
    # Aplica a projeção para o ano corrente
    df_analise_anual = projetar_ano_corrente(df, 'GENERO_AGRUPADO')
    df_analise_anual.rename(columns={'GENERO_AGRUPADO': 'GENERO'}, inplace=True)

    ano_maximo = df['ANO'].max()

    plt.figure()
    # Plota todos os anos, exceto o último
    sns.lineplot(data=df_analise_anual[df_analise_anual['ANO'] < ano_maximo], x='ANO', y='TOTAL', hue='GENERO', marker='o')
    
    # Plota o último ano (projetado) com um marcador diferente
    df_projetado = df_analise_anual[df_analise_anual['ANO'] == ano_maximo]
    sns.lineplot(data=df_projetado, x='ANO', y='TOTAL', hue='GENERO', marker='X', markersize=10, legend=False)

    plt.title(f'Gráfico 1: Evolução do Número Absoluto de {tipo_analise} no Ceará')
    plt.xlabel('Ano')
    plt.ylabel(f'Número Total de Ocorrências ({tipo_analise})')
    plt.legend(title='Gênero')
    
    # Adiciona nota sobre a projeção
    plt.figtext(0.99, 0.01, f'*Dado de {ano_maximo} é uma projeção anualizada.', horizontalalignment='right', style='italic', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para a nota de rodapé
    plt.savefig(f'grafico_1_evolucao_{tipo_analise.lower().replace(" ", "_")}.png')
    plt.show()

def gerar_grafico_2(df, tipo_analise):
    """Gera o Gráfico 2: Análise de crimes contra mulheres por dia e hora (linhas)."""
    print(f"Preparando dados para análise de '{tipo_analise}' contra mulheres por dia e hora...")
    df_feminino = df[df['GENERO_AGRUPADO'] == 'Feminino'].copy()
    if df_feminino.empty:
        print("Não há dados de vítimas femininas para gerar o Gráfico 2.")
        return
    df_feminino['HORA_NUM'] = pd.to_datetime(df_feminino['HORA'], format='%H:%M:%S', errors='coerce').dt.hour
    df_feminino.dropna(subset=['HORA_NUM'], inplace=True)
    df_feminino['HORA_NUM'] = df_feminino['HORA_NUM'].astype(int)
    ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    df_feminino['DIA_SEMANA'] = pd.Categorical(df_feminino['DIA_SEMANA'], categories=ordem_dias, ordered=True)
    crimes_por_dia_hora = df_feminino.groupby(['DIA_SEMANA', 'HORA_NUM']).size().reset_index(name='QUANTIDADE')
    print(f"Gerando Gráfico 2: '{tipo_analise}' contra Mulheres por Hora (linhas)...")
    plt.figure()
    sns.lineplot(data=crimes_por_dia_hora, x='HORA_NUM', y='QUANTIDADE', hue='DIA_SEMANA', marker='o')
    plt.title(f'Gráfico 2: {tipo_analise} contra Mulheres por Hora do Dia')
    plt.xlabel('Hora do Dia'); plt.ylabel(f'Quantidade de {tipo_analise}'); plt.xticks(range(0, 24)); plt.legend(title='Dia da Semana'); plt.tight_layout()
    plt.savefig(f'grafico_2_linhas_dia_hora_feminino_{tipo_analise.lower().replace(" ", "_")}.png')
    plt.show()

def gerar_graficos_3a_e_3b(df_completo, mapeamento_genero):
    """Gera os Gráficos 3a e 3b: Comparativo por tipo de crime (escala log e proporção)."""
    print("\nPreparando dados para os gráficos comparativos por tipo de crime...")
    df_completo['GENERO_AGRUPADO'] = df_completo['GENERO'].map(mapeamento_genero)
    df_crimes_filtrado = df_completo.dropna(subset=['GENERO_AGRUPADO', 'NATUREZA'])
    crimes_por_natureza_genero = df_crimes_filtrado.groupby(['NATUREZA', 'GENERO_AGRUPADO']).size().reset_index(name='QUANTIDADE')
    # Gráfico 3a
    print("Gerando Gráfico 3a: Comparativo com Escala Logarítmica...")
    plt.figure()
    barplot = sns.barplot(data=crimes_por_natureza_genero, x='NATUREZA', y='QUANTIDADE', hue='GENERO_AGRUPADO')
    barplot.set_yscale("log")
    plt.title('Gráfico 3a: Quantidade de Vítimas por Tipo de Crime (Escala Logarítmica)')
    plt.xlabel('Tipo de Crime (Natureza)'); plt.ylabel('Quantidade Total de Vítimas (Escala Log)'); plt.legend(title='Gênero')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig('grafico_3a_comparativo_escala_log.png'); plt.show()
    # Gráfico 3b
    print("Gerando Gráfico 3b: Análise de Proporção de Gênero por Tipo de Crime...")
    total_por_natureza = crimes_por_natureza_genero.groupby('NATUREZA')['QUANTIDADE'].transform('sum')
    crimes_por_natureza_genero['PROPORCAO_%'] = (crimes_por_natureza_genero['QUANTIDADE'] / total_por_natureza) * 100
    df_pivot = crimes_por_natureza_genero.pivot(index='NATUREZA', columns='GENERO_AGRUPADO', values='PROPORCAO_%')
    df_pivot.plot(kind='barh', stacked=True, figsize=(16, 10), colormap='viridis')
    plt.title('Gráfico 3b: Proporção de Gênero em Cada Tipo de Crime')
    plt.xlabel('Porcentagem (%)'); plt.ylabel('Tipo de Crime (Natureza)'); plt.legend(title='Gênero', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 100)
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f%%', label_type='center', color='white', fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.savefig('grafico_3b_comparativo_proporcao.png'); plt.show()

def gerar_grafico_4(df, tipo_analise):
    """Gera um gráfico de barras da distribuição de vítimas por raça/cor."""
    print(f"\nGerando Gráfico 4: Análise de Vítimas por Raça/Cor para '{tipo_analise}'...")
    
    # Limpa os dados, removendo entradas não informadas
    df_raca = df[df['RACA_VITIMA'] != 'NÃO INFORMADA'].copy()
    
    if df_raca.empty:
        print("Não há dados válidos de raça/cor para gerar este gráfico.")
        return
        
    # Agrupa por raça e conta as ocorrências
    contagem_raca = df_raca['RACA_VITIMA'].value_counts().reset_index()
    contagem_raca.columns = ['RACA_VITIMA', 'TOTAL']
    
    # Calcula a proporção
    total_geral = contagem_raca['TOTAL'].sum()
    contagem_raca['PROPORCAO_%'] = (contagem_raca['TOTAL'] / total_geral) * 100
    
    # Gera o gráfico
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(data=contagem_raca, x='RACA_VITIMA', y='TOTAL', palette='viridis')
    
    # Adiciona os rótulos (número absoluto e porcentagem) em cima das barras
    for index, row in contagem_raca.iterrows():
        label = f"{row['TOTAL']}\n({row['PROPORCAO_%']:.1f}%)"
        barplot.text(index, row['TOTAL'] + (total_geral * 0.01), label, color='black', ha="center")
        
    plt.title(f'Gráfico 4: Distribuição de Vítimas por Raça/Cor ({tipo_analise})')
    plt.xlabel('Raça / Cor da Vítima')
    plt.ylabel('Número Absoluto de Vítimas')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'grafico_4_distribuicao_raca_{tipo_analise.lower().replace(" ", "_")}.png')
    plt.show()

def gerar_grafico_5(df, tipo_analise):
    """Gera o Gráfico 5: Análise do perfil etário das vítimas."""
    print(f"\nGerando Gráfico 5: Análise de Vítimas por Faixa Etária para '{tipo_analise}'...")
    
    # Limpa e converte a coluna de idade para numérico
    df_idade = df.copy()
    df_idade['IDADE_NUM'] = pd.to_numeric(df_idade['IDADE_VITIMA'], errors='coerce')
    df_idade.dropna(subset=['IDADE_NUM'], inplace=True)
    df_idade['IDADE_NUM'] = df_idade['IDADE_NUM'].astype(int)
    # Filtra idades improváveis (ex: maiores que 110)
    df_idade = df_idade[df_idade['IDADE_NUM'] <= 110]

    if df_idade.empty:
        print("Não há dados de idade válidos para gerar este gráfico.")
        return

    # Gráfico 5a: Histograma da Idade de Todas as Vítimas
    plt.figure(figsize=(16, 8))
    sns.histplot(data=df_idade, x='IDADE_NUM', bins=30, kde=True)
    idade_media = df_idade['IDADE_NUM'].mean()
    plt.axvline(idade_media, color='red', linestyle='--', linewidth=2)
    plt.text(idade_media + 2, plt.ylim()[1] * 0.9, f'Idade Média: {idade_media:.1f} anos', color='red')
    plt.title(f'Gráfico 5a: Distribuição da Idade das Vítimas ({tipo_analise})')
    plt.xlabel('Idade da Vítima'); plt.ylabel('Número de Ocorrências')
    plt.tight_layout()
    plt.savefig(f'grafico_5a_histograma_idade_{tipo_analise.lower().replace(" ", "_")}.png')
    plt.show()

    # Gráfico 5b: Comparação da Distribuição de Idade por Gênero
    plt.figure(figsize=(16, 8))
    sns.kdeplot(data=df_idade, x='IDADE_NUM', hue='GENERO_AGRUPADO', fill=True, common_norm=False)
    plt.title(f'Gráfico 5b: Comparativo da Densidade de Idade por Gênero ({tipo_analise})')
    plt.xlabel('Idade da Vítima'); plt.ylabel('Densidade')
    plt.tight_layout()
    plt.savefig(f'grafico_5b_kde_idade_genero_{tipo_analise.lower().replace(" ", "_")}.png')
    plt.show()

def gerar_grafico_6(df, tipo_analise):
    """
    Gera o Gráfico 6: Análise do Meio Empregado, com um submenu para
    filtrar por gênero (Todos vs. Feminino).
    """
    print(f"\nAnálise por Meio Empregado para '{tipo_analise}'...")
    print("-"*50)
    print("Escolha o filtro de gênero para esta análise:")
    print("1 - Todos os Registros")
    print("2 - Apenas Vítimas Femininas")
    
    escolha_genero = ""
    while escolha_genero not in ["1", "2"]:
        escolha_genero = input("Digite sua opção (1 ou 2) e pressione Enter: ")

    df_filtrado = df.copy()
    titulo_sufixo = f"({tipo_analise})"

    if escolha_genero == "2":
        df_filtrado = df[df['GENERO_AGRUPADO'] == 'Feminino'].copy()
        titulo_sufixo = f"({tipo_analise} - Vítimas Femininas)"
        print("\nFiltrando dados para vítimas femininas...")

    if df_filtrado.empty:
        print("Não há dados para a seleção de gênero escolhida."); return

    df_meio = df_filtrado.dropna(subset=['MEIO_EMPREGADO'])
    df_meio = df_meio[df_meio['MEIO_EMPREGADO'] != 'NÃO INFORMADO'].copy()

    if df_meio.empty:
        print("Não há dados válidos de 'Meio Empregado' para gerar este gráfico."); return

    # Gráfico 6a (Pizza) - não muda
    print("Gerando Gráfico 6a: Proporção Geral do Meio Empregado...")
    contagem_meio = df_meio['MEIO_EMPREGADO'].value_counts()
    plt.figure(figsize=(10, 10))
    plt.pie(contagem_meio, labels=contagem_meio.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(contagem_meio)))
    plt.title(f'Gráfico 6a: Proporção Geral do Meio Empregado {titulo_sufixo}')
    plt.ylabel(''); plt.tight_layout()
    plt.savefig(f'grafico_6a_pizza_meio_empregado_{titulo_sufixo.lower().replace(" ", "_")}.png'); plt.show()

    # Gráfico 6b (Linhas) - AGORA COM PROJEÇÃO
    print("Gerando Gráfico 6b: Evolução do Meio Empregado ao Longo dos Anos...")
    evolucao_meio = projetar_ano_corrente(df_meio, 'MEIO_EMPREGADO')
    ano_maximo = df_meio['ANO'].max()

    plt.figure(figsize=(16, 9))
    sns.lineplot(data=evolucao_meio[evolucao_meio['ANO'] < ano_maximo], x='ANO', y='TOTAL', hue='MEIO_EMPREGADO', marker='o', palette='viridis')
    df_projetado = evolucao_meio[evolucao_meio['ANO'] == ano_maximo]
    sns.lineplot(data=df_projetado, x='ANO', y='TOTAL', hue='MEIO_EMPREGADO', marker='X', markersize=10, legend=False)
    
    plt.title(f'Gráfico 6b: Evolução do Meio Empregado ao Longo dos Anos {titulo_sufixo}')
    plt.xlabel('Ano'); plt.ylabel('Número Absoluto de Ocorrências'); plt.legend(title='Meio Empregado')
    plt.figtext(0.99, 0.01, f'*Dado de {ano_maximo} é uma projeção anualizada.', horizontalalignment='right', style='italic', fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'grafico_6b_evolucao_meio_empregado_{titulo_sufixo.lower().replace(" ", "_")}.png')
    plt.show()

def gerar_grafico_7(df, tipo_analise):
    """Gera o Gráfico 7: Análise de Crimes de Ódio."""
    print(f"\nGerando Gráfico 7: Análise de Crimes de Ódio para '{tipo_analise}'...")

    # Define as naturezas dos crimes de ódio
    naturezas_odio = ['CONDUTA HOMOFOBICA', 'CONDUTA TRANSFOBICA', 'PRECONCEITO RAÇA OU COR']
    df_odio = df[df['NATUREZA'].isin(naturezas_odio)].copy()

    if df_odio.empty:
        print("Não foram encontrados registros de crimes de ódio para gerar esta análise.")
        return

    # --- Gráfico 7a: Evolução Temporal dos Crimes de Ódio ---
    print("Gerando Gráfico 7a: Evolução Anual dos Crimes de Ódio...")
    evolucao_odio = df_odio.groupby(['ANO', 'NATUREZA']).size().reset_index(name='TOTAL')
    
    plt.figure(figsize=(16, 9))
    sns.lineplot(data=evolucao_odio, x='ANO', y='TOTAL', hue='NATUREZA', marker='o', palette='Set1')
    plt.title(f'Gráfico 7a: Evolução dos Registros de Crimes de Ódio por Ano ({tipo_analise})')
    plt.xlabel('Ano')
    plt.ylabel('Número de Ocorrências')
    plt.legend(title='Tipo de Crime de Ódio')
    plt.tight_layout()
    plt.savefig(f'grafico_7a_evolucao_crimes_odio_{tipo_analise.lower().replace(" ", "_")}.png')
    plt.show()

    # --- Gráfico 7b: Perfil da Orientação Sexual em Crimes de LGBTfobia ---
    print("Gerando Gráfico 7b: Perfil da Orientação Sexual em Vítimas de LGBTfobia...")
    naturezas_lgbtfobia = ['CONDUTA HOMOFOBICA', 'CONDUTA TRANSFOBICA']
    df_lgbtfobia = df_odio[df_odio['NATUREZA'].isin(naturezas_lgbtfobia)].copy()
    
    # Limpa os dados de orientação sexual
    df_lgbtfobia = df_lgbtfobia[df_lgbtfobia['ORIENTACAO_SEXUAL'] != 'NÃO INFORMADO'].copy()

    if df_lgbtfobia.empty:
        print("Não há dados válidos de orientação sexual para gerar o Gráfico 7b.")
        return

    contagem_orientacao = df_lgbtfobia['ORIENTACAO_SEXUAL'].value_counts().reset_index()
    contagem_orientacao.columns = ['ORIENTACAO_SEXUAL', 'TOTAL']

    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(data=contagem_orientacao, x='ORIENTACAO_SEXUAL', y='TOTAL', palette='coolwarm')
    
    # Adiciona os rótulos com os números absolutos
    for index, row in contagem_orientacao.iterrows():
        barplot.text(index, row['TOTAL'] + (contagem_orientacao['TOTAL'].max() * 0.01), row['TOTAL'], color='black', ha="center")

    plt.title(f'Gráfico 7b: Perfil da Orientação Sexual em Vítimas de LGBTfobia ({tipo_analise})')
    plt.xlabel('Orientação Sexual da Vítima')
    plt.ylabel('Número Absoluto de Vítimas')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'grafico_7b_perfil_orientacao_sexual_{tipo_analise.lower().replace(" ", "_")}.png')
    plt.show()

def executar_analise_geografica(df_crimes):
    """
    Gera mapas interativos individuais para cada tipo de crime,
    separados por Municípios e AIS.
    """
    print("\nIniciando Análise Geográfica (Municípios e AIS)...")
    print("="*60)

    # Função auxiliar para baixar APENAS o arquivo de população
    def baixar_arquivo_populacao(url, nome_arquivo):
        if not os.path.exists(nome_arquivo):
            print(f"Arquivo de população '{nome_arquivo}' não encontrado. Baixando...")
            try:
                resposta = requests.get(url, timeout=60)
                resposta.raise_for_status()
                with open(nome_arquivo, 'w', encoding='utf-8') as f:
                    f.write(resposta.text)
                print(f"Download de '{nome_arquivo}' concluído.")
            except requests.exceptions.RequestException as e:
                print(f"ERRO CRÍTICO: Falha ao baixar o arquivo de população: {e}"); exit()
        else:
            print(f"Arquivo de população '{nome_arquivo}' já existe localmente.")

    # 1. Carregamento dos Dados
    url_populacao = "https://gist.githubusercontent.com/AI-Manus/bf201d4d41b544a444877443d2844b67/raw/280a3c23de5901a542335de66d742b6343956073/populacao_ce_municipios_2021.csv"
    arquivo_populacao = "populacao_ce.csv"
    baixar_arquivo_populacao(url_populacao, arquivo_populacao )

    try:
        gdf_municipios = gpd.read_file('municipios_ce.geojson')
        df_populacao = pd.read_csv(arquivo_populacao)
        print("Arquivos 'municipios_ce.geojson' e 'populacao_ce.csv' carregados com sucesso.")
    except Exception as e:
        print(f"ERRO CRÍTICO ao ler os arquivos locais. Verifique se 'municipios_ce.geojson' está na pasta.")
        print(f"Detalhe do erro: {e}")
        exit()

    # 2. Mapeamento e Preparação dos Dados
    print("Mapeando municípios para AIS e processando dados...")
    
    municipios_ais = {
        'Fortaleza': 'AIS 1-10', 'Caucaia': 'AIS 11', 'Maracanaú': 'AIS 12', 'Aquiraz': 'AIS 13', 'Cascavel': 'AIS 13', 'Eusébio': 'AIS 13', 'Pindoretama': 'AIS 13',
        'Alcântaras': 'AIS 14', 'Barroquinha': 'AIS 14', 'Camocim': 'AIS 14', 'Cariré': 'AIS 14', 'Carnaubal': 'AIS 14', 'Chaval': 'AIS 14', 'Coreaú': 'AIS 14', 'Croatá': 'AIS 14', 'Forquilha': 'AIS 14', 'Frecheirinha': 'AIS 14', 'Graça': 'AIS 14', 'Granja': 'AIS 14', 'Groaíras': 'AIS 14', 'Guaraciaba do Norte': 'AIS 14', 'Ibiapina': 'AIS 14', 'Martinópole': 'AIS 14', 'Massapê': 'AIS 14', 'Meruoca': 'AIS 14', 'Moraújo': 'AIS 14', 'Mucambo': 'AIS 14', 'Pacujá': 'AIS 14', 'Santana do Acaraú': 'AIS 14', 'São Benedito': 'AIS 14', 'Senador Sá': 'AIS 14', 'Sobral': 'AIS 14', 'Tianguá': 'AIS 14', 'Ubajara': 'AIS 14', 'Uruoca': 'AIS 14', 'Viçosa do Ceará': 'AIS 14',
        'Acarape': 'AIS 15', 'Aracoiaba': 'AIS 15', 'Aratuba': 'AIS 15', 'Barreira': 'AIS 15', 'Baturité': 'AIS 15', 'Boa Viagem': 'AIS 15', 'Canindé': 'AIS 15', 'Capistrano': 'AIS 15', 'Caridade': 'AIS 15', 'Guaramiranga': 'AIS 15', 'Itapiúna': 'AIS 15', 'Itatira': 'AIS 15', 'Madalena': 'AIS 15', 'Mulungu': 'AIS 15', 'Ocara': 'AIS 15', 'Pacoti': 'AIS 15', 'Palmácia': 'AIS 15', 'Paramoti': 'AIS 15', 'Redenção': 'AIS 15',
        'Ararendá': 'AIS 16', 'Catunda': 'AIS 16', 'Crateús': 'AIS 16', 'Hidrolândia': 'AIS 16', 'Independência': 'AIS 16', 'Ipaporanga': 'AIS 16', 'Ipu': 'AIS 16', 'Ipueiras': 'AIS 16', 'Monsenhor Tabosa': 'AIS 16', 'Nova Russas': 'AIS 16', 'Novo Oriente': 'AIS 16', 'Pires Ferreira': 'AIS 16', 'Poranga': 'AIS 16', 'Reriutaba': 'AIS 16', 'Santa Quitéria': 'AIS 16', 'Tamboril': 'AIS 16', 'Varjota': 'AIS 16',
        'Acaraú': 'AIS 17', 'Amontada': 'AIS 17', 'Apuiarés': 'AIS 17', 'Bela Cruz': 'AIS 17', 'Cruz': 'AIS 17', 'General Sampaio': 'AIS 17', 'Irauçuba': 'AIS 17', 'Itapajé': 'AIS 17', 'Itapipoca': 'AIS 17', 'Itarema': 'AIS 17', 'Jijoca de Jericoacoara': 'AIS 17', 'Marco': 'AIS 17', 'Miraíma': 'AIS 17', 'Morrinhos': 'AIS 17', 'Pentecoste': 'AIS 17', 'Tejuçuoca': 'AIS 17', 'Tururu': 'AIS 17', 'Umirim': 'AIS 17', 'Uruburetama': 'AIS 17',
        'Alto Santo': 'AIS 18', 'Aracati': 'AIS 18', 'Beberibe': 'AIS 18', 'Ererê': 'AIS 18', 'Fortim': 'AIS 18', 'Icapuí': 'AIS 18', 'Iracema': 'AIS 18', 'Itaiçaba': 'AIS 18', 'Jaguaribe': 'AIS 18', 'Jaguaruana': 'AIS 18', 'Limoeiro do Norte': 'AIS 18', 'Nova Jaguaribara': 'AIS 18', 'Palhano': 'AIS 18', 'Pereiro': 'AIS 18', 'Potiretama': 'AIS 18', 'Quixeré': 'AIS 18', 'Russas': 'AIS 18', 'São João do Jaguaribe': 'AIS 18', 'Tabuleiro do Norte': 'AIS 18','Jaguaribara': 'AIS 18',
        'Abaiara': 'AIS 19', 'Altaneira': 'AIS 19', 'Antonina do Norte': 'AIS 19', 'Araripe': 'AIS 19', 'Assaré': 'AIS 19', 'Aurora': 'AIS 19', 'Barbalha': 'AIS 19', 'Barro': 'AIS 19', 'Brejo Santo': 'AIS 19', 'Campos Sales': 'AIS 19', 'Caririaçu': 'AIS 19', 'Crato': 'AIS 19', 'Farias Brito': 'AIS 19', 'Jardim': 'AIS 19', 'Jati': 'AIS 19', 'Juazeiro do Norte': 'AIS 19', 'Mauriti': 'AIS 19', 'Milagres': 'AIS 19', 'Missão Velha': 'AIS 19', 'Nova Olinda': 'AIS 19', 'Penaforte': 'AIS 19', 'Porteiras': 'AIS 19', 'Potengi': 'AIS 19', 'Salitre': 'AIS 19', 'Santana do Cariri': 'AIS 19',
        'Banabuiú': 'AIS 20', 'Choró': 'AIS 20', 'Deputado Irapuan Pinheiro': 'AIS 20', 'Ibaretama': 'AIS 20', 'Ibicuitinga': 'AIS 20', 'Jaguaretama': 'AIS 20', 'Milhã': 'AIS 20', 'Morada Nova': 'AIS 20', 'Pedra Branca': 'AIS 20', 'Quixadá': 'AIS 20', 'Quixeramobim': 'AIS 20', 'Senador Pompeu': 'AIS 20', 'Solonópole': 'AIS 20',
        'Acopiara': 'AIS 21', 'Baixio': 'AIS 21', 'Cariús': 'AIS 21', 'Cedro': 'AIS 21', 'Granjeiro': 'AIS 21', 'Icó': 'AIS 21', 'Iguatu': 'AIS 21', 'Ipaumirim': 'AIS 21', 'Jucás': 'AIS 21', 'Lavras da Mangabeira': 'AIS 21', 'Orós': 'AIS 21', 'Quixelô': 'AIS 21', 'Saboeiro': 'AIS 21', 'Tarrafas': 'AIS 21', 'Umari': 'AIS 21', 'Várzea Alegre': 'AIS 21',
        'Aiuaba': 'AIS 22', 'Arneiroz': 'AIS 22', 'Catarina': 'AIS 22', 'Mombaça': 'AIS 22', 'Parambu': 'AIS 22', 'Piquet Carneiro': 'AIS 22', 'Quiterianópolis': 'AIS 22', 'Tauá': 'AIS 22',
        'Paracuru': 'AIS 23', 'Paraipaba': 'AIS 23', 'São Gonçalo do Amarante': 'AIS 23', 'São Luís do Curu': 'AIS 23', 'Trairi': 'AIS 23',
        'Guaiúba': 'AIS 24', 'Maranguape': 'AIS 24', 'Pacatuba': 'AIS 24',
        'Chorozinho': 'AIS 25', 'Horizonte': 'AIS 25', 'Itaitinga': 'AIS 25', 'Pacajus': 'AIS 25'
    }
    
    def normalize_text(text_series):
        return text_series.str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    df_crimes['MUNICIPIO_NORM'] = normalize_text(df_crimes['MUNICIPIO'])
    gdf_municipios['NM_MUN_NORM'] = normalize_text(gdf_municipios['name'])
    df_populacao['MUNICIPIO_NORM'] = normalize_text(df_populacao['municipio'])
    
    gdf_municipios['AIS'] = gdf_municipios['name'].map(municipios_ais)
    gdf_municipios.loc[gdf_municipios['name'] == 'Fortaleza', 'AIS'] = 'AIS 1-10'
    
    print("Gerando geometrias das AIS a partir dos municípios...")
    gdf_ais = gdf_municipios.dissolve(by='AIS').reset_index()
    
    # --- ANÁLISE POR MUNICÍPIO ---
    crimes_por_municipio = df_crimes.groupby(['MUNICIPIO_NORM', 'NATUREZA']).size().reset_index(name='QUANTIDADE')
    dados_municipio = pd.merge(crimes_por_municipio, df_populacao[['MUNICIPIO_NORM', 'populacao']], on='MUNICIPIO_NORM', how='left')
    dados_municipio.dropna(subset=['populacao'], inplace=True)
    dados_municipio['TAXA_POR_100K'] = (dados_municipio['QUANTIDADE'] / dados_municipio['populacao']) * 100000
    mapa_data_municipios = gdf_municipios.merge(dados_municipio, left_on='NM_MUN_NORM', right_on='MUNICIPIO_NORM', how='left')
    mapa_data_municipios[['QUANTIDADE', 'TAXA_POR_100K']] = mapa_data_municipios[['QUANTIDADE', 'TAXA_POR_100K']].fillna(0)

    # --- ANÁLISE POR AIS ---
    df_crimes['AIS_MAPEADA'] = df_crimes['MUNICIPIO'].map(municipios_ais)
    df_crimes.loc[df_crimes['MUNICIPIO'] == 'Fortaleza', 'AIS_MAPEADA'] = 'AIS 1-10'

    df_populacao['AIS'] = df_populacao['municipio'].map(municipios_ais)
    df_populacao.loc[df_populacao['municipio'] == 'Fortaleza', 'AIS'] = 'AIS 1-10'
    pop_por_ais = df_populacao.groupby('AIS')['populacao'].sum().reset_index()

    crimes_por_ais = df_crimes.groupby(['AIS_MAPEADA', 'NATUREZA']).size().reset_index(name='QUANTIDADE')
    dados_ais = pd.merge(crimes_por_ais, pop_por_ais, left_on='AIS_MAPEADA', right_on='AIS', how='left')
    dados_ais.dropna(subset=['populacao'], inplace=True)
    dados_ais['TAXA_POR_100K'] = (dados_ais['QUANTIDADE'] / dados_ais['populacao']) * 100000
    
    mapa_data_ais = gdf_ais.merge(dados_ais, on='AIS', how='left')
    mapa_data_ais[['QUANTIDADE', 'TAXA_POR_100K']] = mapa_data_ais[['QUANTIDADE', 'TAXA_POR_100K']].fillna(0)

    # 4. Criação dos Mapas Interativos (ESTRUTURA SIMPLIFICADA)
    print("Criando os mapas interativos...")
    lista_de_crimes = sorted(df_crimes['NATUREZA'].dropna().unique().tolist())

    # --- MAPA 1: MUNICÍPIOS ---
    mapa_municipios = folium.Map(location=[-5.0, -39.5], zoom_start=7, tiles='CartoDB positron')
    for crime in lista_de_crimes:
        camada_choropleth = folium.Choropleth(
            geo_data=mapa_data_municipios[mapa_data_municipios['NATUREZA'] == crime],
            name=crime,
            data=mapa_data_municipios[mapa_data_municipios['NATUREZA'] == crime],
            columns=['NM_MUN_NORM', 'TAXA_POR_100K'],
            key_on='feature.properties.NM_MUN_NORM',
            fill_color='YlOrRd',
            legend_name=f'Taxa de {crime} por 100k (Municípios)',
            highlight=True,
            show=(crime == lista_de_crimes[0]) # Mostra a primeira camada por padrão
        )
        folium.GeoJsonTooltip(['name', 'QUANTIDADE', 'TAXA_POR_100K'], aliases=['Município:', 'Nº Absoluto:', 'Taxa/100k:']).add_to(camada_choropleth.geojson)
        camada_choropleth.add_to(mapa_municipios)

    folium.LayerControl(collapsed=False).add_to(mapa_municipios)
    mapa_municipios.save("mapa_municipios.html")
    print("\nSucesso! O arquivo 'mapa_municipios.html' foi criado.")

    # --- MAPA 2: AIS ---
    mapa_ais = folium.Map(location=[-5.0, -39.5], zoom_start=7, tiles='CartoDB positron')
    for crime in lista_de_crimes:
        camada_choropleth = folium.Choropleth(
            geo_data=mapa_data_ais[mapa_data_ais['NATUREZA'] == crime],
            name=crime,
            data=mapa_data_ais[mapa_data_ais['NATUREZA'] == crime],
            columns=['AIS', 'TAXA_POR_100K'],
            key_on='feature.properties.AIS',
            fill_color='PuBuGn',
            legend_name=f'Taxa de {crime} por 100k (AIS)',
            highlight=True,
            show=(crime == lista_de_crimes[0])
        )
        folium.GeoJsonTooltip(['AIS', 'QUANTIDADE', 'TAXA_POR_100K'], aliases=['AIS:', 'Nº Absoluto:', 'Taxa/100k:']).add_to(camada_choropleth.geojson)
        camada_choropleth.add_to(mapa_ais)

    folium.LayerControl(collapsed=False).add_to(mapa_ais)
    mapa_ais.save("mapa_ais.html")
    print("Sucesso! O arquivo 'mapa_ais.html' foi criado.")

if __name__ == "__main__":
    try:
        df_crimes_original = pd.read_csv('crimes.csv', sep=',')
    except FileNotFoundError:
        print("Erro: Arquivo 'crimes.csv' não encontrado."); exit()
    
    df_crimes_original.columns = [
        'AIS', 'NATUREZA', 'MUNICIPIO', 'LOCAL', 'DATA', 'HORA', 'DIA_SEMANA',
        'MEIO_EMPREGADO', 'GENERO', 'ORIENTACAO_SEXUAL', 'IDADE_VITIMA',
        'ESCOLARIDADE_VITIMA', 'RACA_VITIMA'
    ]

    # --- MENU PRINCIPAL ---
    print("Bem-vindo à ferramenta de análise criminal do Ceará.")
    print("="*50)
    print("MENU PRINCIPAL: Escolha o tipo de análise que deseja realizar:")
    print("1 - Análise de Gráficos (Foco em Homicídios)")
    print("2 - Análise de Gráficos (Todos os Crimes)")
    print("3 - Análise Geográfica (Mapa de Taxa de Crimes)")
    
    escolha_principal = ""
    while escolha_principal not in ["1", "2", "3"]:
        escolha_principal = input("Digite sua opção (1, 2 ou 3) e pressione Enter: ")

    # --- LÓGICA BASEADA NA ESCOLHA PRINCIPAL ---
    if escolha_principal in ["1", "2"]:
        if escolha_principal == "1":
            tipo_analise = "Homicídios"
            naturezas_filtro = ['HOMICIDIO DOLOSO', 'FEMINICIDIO', 'LATROCINIO', 'LESAO CORPORAL SEGUIDA DE MORTE']
            df_para_analise = df_crimes_original[df_crimes_original['NATUREZA'].isin(naturezas_filtro)].copy()
        else:
            tipo_analise = "Todos os Crimes Registrados"
            df_para_analise = df_crimes_original.copy()

        mapeamento_genero = {'Masculino': 'Masculino', 'Homem Trans': 'Masculino', 'Feminino': 'Feminino', 'Mulher Trans': 'Feminino', 'Travesti': 'Feminino'}
        df_para_analise['GENERO_AGRUPADO'] = df_para_analise['GENERO'].map(mapeamento_genero)
        df_para_analise.dropna(subset=['GENERO_AGRUPADO'], inplace=True)
        df_para_analise['DATA'] = pd.to_datetime(df_para_analise['DATA'], dayfirst=True, errors='coerce')
        df_para_analise['ANO'] = df_para_analise['DATA'].dt.year
        df_para_analise.dropna(subset=['ANO'], inplace=True)
        df_para_analise['ANO'] = df_para_analise['ANO'].astype(int)

        # --- MENU DE GRÁFICOS ATUALIZADO ---
        print("\n" + "="*50)
        print("MENU DE GRÁFICOS: Qual gráfico você deseja gerar?")
        print("1 - Gráfico 1 (Evolução Anual por Gênero)")
        print("2 - Gráfico 2 (Análise de Crimes contra Mulheres por Dia/Hora)")
        print("3 - Gráficos 3a e 3b (Comparativo por Tipo de Crime)")
        print("4 - Gráfico 4 (Análise por Raça/Cor da Vítima)")
        print("5 - Gráfico 5 (Análise por Faixa Etária da Vítima)")
        print("6 - Gráfico 6 (Análise por Meio Empregado)")
        print("7 - Gráfico 7 (Análise de Crimes de Ódio)") # Nova opção
        print("8 - TODOS os gráficos") # Opção atualizada
        
        escolha_grafico = ""
        while escolha_grafico not in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            escolha_grafico = input("Digite sua opção (1 a 8) e pressione Enter: ")

        if escolha_grafico == "1":
            gerar_grafico_1(df_para_analise, tipo_analise)
        elif escolha_grafico == "2":
            gerar_grafico_2(df_para_analise, tipo_analise)
        elif escolha_grafico == "3":
            gerar_graficos_3a_e_3b(df_crimes_original, mapeamento_genero)
        elif escolha_grafico == "4":
            gerar_grafico_4(df_para_analise, tipo_analise)
        elif escolha_grafico == "5":
            gerar_grafico_5(df_para_analise, tipo_analise)
        elif escolha_grafico == "6":
            gerar_grafico_6(df_para_analise, tipo_analise)
        elif escolha_grafico == "7":
            gerar_grafico_7(df_para_analise, tipo_analise) # Chamada da nova função
        elif escolha_grafico == "8":
            print("\nGerando TODOS os gráficos...")
            gerar_grafico_1(df_para_analise, tipo_analise)
            gerar_grafico_2(df_para_analise, tipo_analise)
            gerar_graficos_3a_e_3b(df_crimes_original, mapeamento_genero)
            gerar_grafico_4(df_para_analise, tipo_analise)
            gerar_grafico_5(df_para_analise, tipo_analise)
            gerar_grafico_6(df_para_analise, tipo_analise)
            gerar_grafico_7(df_para_analise, tipo_analise)

    elif escolha_principal == "3":
        executar_analise_geografica(df_crimes_original)

    print("\nOperação concluída com sucesso!")

    try:
        df_crimes_original = pd.read_csv('crimes.csv', sep=',')
    except FileNotFoundError:
        print("Erro: Arquivo 'crimes.csv' não encontrado."); exit()
    
    df_crimes_original.columns = [
        'AIS', 'NATUREZA', 'MUNICIPIO', 'LOCAL', 'DATA', 'HORA', 'DIA_SEMANA',
        'MEIO_EMPREGADO', 'GENERO', 'ORIENTACAO_SEXUAL', 'IDADE_VITIMA',
        'ESCOLARIDADE_VITIMA', 'RACA_VITIMA'
    ]

    # --- MENU PRINCIPAL ---
    print("Bem-vindo à ferramenta de análise criminal do Ceará.")
    print("="*50)
    print("MENU PRINCIPAL: Escolha o tipo de análise que deseja realizar:")
    print("1 - Análise de Gráficos (Foco em Homicídios)")
    print("2 - Análise de Gráficos (Todos os Crimes)")
    print("3 - Análise Geográfica (Mapa de Taxa de Crimes)")
    
    escolha_principal = ""
    while escolha_principal not in ["1", "2", "3"]:
        escolha_principal = input("Digite sua opção (1, 2 ou 3) e pressione Enter: ")

    # --- LÓGICA BASEADA NA ESCOLHA PRINCIPAL ---
    if escolha_principal in ["1", "2"]:
        if escolha_principal == "1":
            tipo_analise = "Homicídios"
            naturezas_filtro = ['HOMICIDIO DOLOSO', 'FEMINICIDIO', 'LATROCINIO', 'LESAO CORPORAL SEGUIDA DE MORTE']
            df_para_analise = df_crimes_original[df_crimes_original['NATUREZA'].isin(naturezas_filtro)].copy()
        else:
            tipo_analise = "Todos os Crimes Registrados"
            df_para_analise = df_crimes_original.copy()

        mapeamento_genero = {'Masculino': 'Masculino', 'Homem Trans': 'Masculino', 'Feminino': 'Feminino', 'Mulher Trans': 'Feminino', 'Travesti': 'Feminino'}
        df_para_analise['GENERO_AGRUPADO'] = df_para_analise['GENERO'].map(mapeamento_genero)
        df_para_analise.dropna(subset=['GENERO_AGRUPADO'], inplace=True)
        df_para_analise['DATA'] = pd.to_datetime(df_para_analise['DATA'], dayfirst=True, errors='coerce')
        df_para_analise['ANO'] = df_para_analise['DATA'].dt.year
        df_para_analise.dropna(subset=['ANO'], inplace=True)
        df_para_analise['ANO'] = df_para_analise['ANO'].astype(int)

        # --- MENU DE GRÁFICOS ATUALIZADO ---
        print("\n" + "="*50)
        print("MENU DE GRÁFICOS: Qual gráfico você deseja gerar?")
        print("1 - Gráfico 1 (Evolução Anual por Gênero)")
        print("2 - Gráfico 2 (Análise de Crimes contra Mulheres por Dia/Hora)")
        print("3 - Gráficos 3a e 3b (Comparativo por Tipo de Crime)")
        print("4 - Gráfico 4 (Análise por Raça/Cor da Vítima)")
        print("5 - Gráfico 5 (Análise por Faixa Etária da Vítima)")
        print("6 - Gráfico 6 (Análise por Meio Empregado)")
        print("7 - TODOS os gráficos")
        
        escolha_grafico = ""
        while escolha_grafico not in ["1", "2", "3", "4", "5", "6", "7"]:
            escolha_grafico = input("Digite sua opção (1 a 7) e pressione Enter: ")

        if escolha_grafico == "1":
            gerar_grafico_1(df_para_analise, tipo_analise)
        elif escolha_grafico == "2":
            gerar_grafico_2(df_para_analise, tipo_analise)
        elif escolha_grafico == "3":
            gerar_graficos_3a_e_3b(df_crimes_original, mapeamento_genero)
        elif escolha_grafico == "4":
            gerar_grafico_4(df_para_analise, tipo_analise)
        elif escolha_grafico == "5":
            gerar_grafico_5(df_para_analise, tipo_analise)
        elif escolha_grafico == "6":
            gerar_grafico_6(df_para_analise, tipo_analise)
        elif escolha_grafico == "7":
            print("\nGerando TODOS os gráficos...")
            gerar_grafico_1(df_para_analise, tipo_analise)
            gerar_grafico_2(df_para_analise, tipo_analise)
            gerar_graficos_3a_e_3b(df_crimes_original, mapeamento_genero)
            gerar_grafico_4(df_para_analise, tipo_analise)
            gerar_grafico_5(df_para_analise, tipo_analise)
            print("\n--- Gerando Gráfico 6 para Todos os Registros ---")
            gerar_grafico_6(df_para_analise, tipo_analise)
            print("\n--- Gerando Gráfico 6 para Apenas Vítimas Femininas ---")
            df_feminino_analise = df_para_analise[df_para_analise['GENERO_AGRUPADO'] == 'Feminino']
            gerar_grafico_6(df_feminino_analise, tipo_analise)


    elif escolha_principal == "3":
        executar_analise_geografica(df_crimes_original)

    print("\nOperação concluída com sucesso!")