import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Carregar os arquivos CSV
statistics_df = pd.read_csv('exp_data_frame/preprocessing_combinado_plate')
dados_imagens_df = pd.read_csv('dados_imagens.csv')

# Criar um dicionário para mapear OCR correto com base no nome da imagem
image_to_true_ocr = {row['Image Name']: row['OCR'] for _, row in dados_imagens_df.iterrows()}

# Adicionar a coluna 'Image Name' no dataframe statistics
statistics_df['Image Name'] = statistics_df['Image Path'].apply(lambda x: x.split('/')[-1])

# Adicionar a coluna 'True OCR' baseada no dicionário criado
statistics_df['True OCR'] = statistics_df['Image Name'].map(image_to_true_ocr)

# Filtrar linhas onde OCR verdadeiro e predito estão disponíveis e remover "Desconhecido"
valid_rows = statistics_df.dropna(subset=['True OCR', 'Plate Number'])
valid_rows = valid_rows[valid_rows['True Class'] != 'Desconhecido']

# Função para salvar DataFrame em CSV
def salvar_csv(df, nome_arquivo):
    df.to_csv(nome_arquivo, index=True)

# Função para gerar gráficos de métricas sem incluir acurácia
def gerar_graficos_sem_acuracia(df, colunas_metricas, titulo, y_label, nome_arquivo):
    df[colunas_metricas].plot(kind='bar', figsize=(10, 6))
    plt.title(titulo)
    plt.ylabel(y_label)
    plt.xlabel('Classe e Turno')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    plt.show()

# Função para calcular métricas por classe veicular e turno (dia/noite)
def calcular_metricas_por_classe_turno(df, classe, turno):
    # Filtrar para a classe veicular e turno em questão
    df_filtrado = df[(df['True Class'] == classe) & (df['Day/Night'] == turno)]

    # Calcular métricas se houver dados suficientes
    if len(df_filtrado) > 0:
        accuracy = accuracy_score(df_filtrado['True OCR'], df_filtrado['Plate Number'])
        precision = precision_score(df_filtrado['True OCR'], df_filtrado['Plate Number'], average='weighted', zero_division=0)
        recall = recall_score(df_filtrado['True OCR'], df_filtrado['Plate Number'], average='weighted', zero_division=0)
        f1 = f1_score(df_filtrado['True OCR'], df_filtrado['Plate Number'], average='weighted', zero_division=0)
        return round(accuracy * 100, 3), round(precision * 100, 3), round(recall * 100, 3), round(f1 * 100, 3)
    else:
        return None, None, None, None

# Função para calcular métricas de detecção de caracteres (7 caracteres no Plate Number)
def calcular_metricas_detecao_caracteres(df, classe, turno):
    # Filtrar para a classe veicular e turno em questão
    df_filtrado = df[(df['True Class'] == classe) & (df['Day/Night'] == turno)]

    # Calcular métricas de detecção de caracteres
    if len(df_filtrado) > 0:
        y_true = df_filtrado['Plate Number'].apply(lambda x: len(x) == 7)  # Condição de acerto de 7 caracteres
        y_pred = [True] * len(df_filtrado)  # Considerando todos como preditos corretamente para 7 caracteres

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return round(accuracy * 100, 3), round(precision * 100, 3), round(recall * 100, 3), round(f1 * 100, 3)
    else:
        return None, None, None, None

# Dicionário para armazenar métricas por classe e turno
metricas_por_classe_turno = {}
metricas_caracteres_por_classe_turno = {}

# Identificar as classes veiculares únicas
classes_veiculares = valid_rows['True Class'].unique()

# Calcular métricas para cada classe e turno (Dia e Noite)
for classe in classes_veiculares:
    for turno in ['Dia', 'Noite']:
        # Calcular métricas de OCR
        accuracy, precision, recall, f1 = calcular_metricas_por_classe_turno(valid_rows, classe, turno)
        metricas_por_classe_turno[(classe, turno)] = {
            'Acurácia OCR': accuracy,
            'Precisão OCR': precision,
            'Recall OCR': recall,
            'F1-Score OCR': f1
        }

        # Calcular métricas de detecção de caracteres
        accuracy_chars, precision_chars, recall_chars, f1_chars = calcular_metricas_detecao_caracteres(valid_rows, classe, turno)
        metricas_caracteres_por_classe_turno[(classe, turno)] = {
            'Acurácia Detecção de Caracteres': accuracy_chars,
            'Precisão Detecção de Caracteres': precision_chars,
            'Recall Detecção de Caracteres': recall_chars,
            'F1-Score Detecção de Caracteres': f1_chars
        }

# Transformar os dicionários de métricas em DataFrames para exibir
metricas_turno_df = pd.DataFrame(metricas_por_classe_turno).T
metricas_caracteres_turno_df = pd.DataFrame(metricas_caracteres_por_classe_turno).T

# Salvar os DataFrames em arquivos CSV
salvar_csv(metricas_turno_df, 'ID10_OCR.csv')
salvar_csv(metricas_caracteres_turno_df, 'ID10_CHAR.csv')

# Gerar gráficos de métricas de OCR por classe e turno (sem acurácia)
colunas_ocr = ['Precisão OCR', 'Recall OCR', 'F1-Score OCR']
gerar_graficos_sem_acuracia(metricas_turno_df, colunas_ocr, 'Métricas de OCR por Classe e Turno (Sem Acurácia)', 'Valor', 'metricas_ocr_sem_acuracia.png')

# Gerar gráficos de métricas de detecção de caracteres por classe e turno (sem acurácia)
colunas_caracteres = ['Precisão Detecção de Caracteres', 'Recall Detecção de Caracteres', 'F1-Score Detecção de Caracteres']
gerar_graficos_sem_acuracia(metricas_caracteres_turno_df, colunas_caracteres, 'Métricas de Detecção de Caracteres por Classe e Turno (Sem Acurácia)', 'Valor', 'metricas_caracteres_sem_acuracia.png')

# Exibir as métricas calculadas por classe e turno
print("Métricas de OCR por classe e turno:")
print(metricas_turno_df)

print("\nMétricas de Detecção de Caracteres por classe e turno:")
print(metricas_caracteres_turno_df)
