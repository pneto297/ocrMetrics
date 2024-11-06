import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Carregar os arquivos CSV
statistics_df = pd.read_csv('exp_data_frame/statistics_firts_moto_detector_preprocessing_combinado_plate_downsampling_general_mais_all.csv')
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

# Função para calcular métricas por classe veicular e turno (dia/noite) com número absoluto de casos
def calcular_metricas_por_classe_turno(df, classe, turno):
    # Filtrar para a classe veicular e turno em questão
    df_filtrado = df[(df['True Class'] == classe) & (df['Day/Night'] == turno)]

    # Calcular métricas se houver dados suficientes
    if len(df_filtrado) > 0:
        accuracy = accuracy_score(df_filtrado['True OCR'], df_filtrado['Plate Number'])
        precision = precision_score(df_filtrado['True OCR'], df_filtrado['Plate Number'], average='weighted', zero_division=0)
        recall = recall_score(df_filtrado['True OCR'], df_filtrado['Plate Number'], average='weighted', zero_division=0)
        f1 = f1_score(df_filtrado['True OCR'], df_filtrado['Plate Number'], average='weighted', zero_division=0)
        return len(df_filtrado), round(accuracy * 100, 3), round(precision * 100, 3), round(recall * 100, 3), round(f1 * 100, 3)
    else:
        return 0, None, None, None, None

# Função para calcular métricas de detecção de caracteres com número absoluto de casos
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

        return len(df_filtrado), round(accuracy * 100, 3), round(precision * 100, 3), round(recall * 100, 3), round(f1 * 100, 3)
    else:
        return 0, None, None, None, None

# Dicionário para armazenar métricas por classe e turno
metricas_por_classe_turno = {}
metricas_caracteres_por_classe_turno = {}

# Identificar as classes veiculares únicas
classes_veiculares = valid_rows['True Class'].unique()

# Calcular métricas para cada classe e turno (Dia e Noite)
for classe in classes_veiculares:
    for turno in ['Dia', 'Noite']:
        # Calcular métricas de OCR
        n_cases, accuracy, precision, recall, f1 = calcular_metricas_por_classe_turno(valid_rows, classe, turno)
        metricas_por_classe_turno[(classe, turno)] = {
            'Casos Absolutos': n_cases,
            'Acurácia OCR': accuracy,
            'Precisão OCR': precision,
            'Recall OCR': recall,
            'F1-Score OCR': f1
        }

        # Calcular métricas de detecção de caracteres
        n_cases_chars, accuracy_chars, precision_chars, recall_chars, f1_chars = calcular_metricas_detecao_caracteres(valid_rows, classe, turno)
        metricas_caracteres_por_classe_turno[(classe, turno)] = {
            'Casos Absolutos': n_cases_chars,
            'Acurácia Detecção de Caracteres': accuracy_chars,
            'Precisão Detecção de Caracteres': precision_chars,
            'Recall Detecção de Caracteres': recall_chars,
            'F1-Score Detecção de Caracteres': f1_chars
        }

# Transformar os dicionários de métricas em DataFrames para exibir e salvar
metricas_turno_df = pd.DataFrame(metricas_por_classe_turno).T
metricas_caracteres_turno_df = pd.DataFrame(metricas_caracteres_por_classe_turno).T

# Salvar os DataFrames em arquivos CSV
metricas_turno_df.to_csv('ID10_OCR_com_absolutos.csv', index=True)
metricas_caracteres_turno_df.to_csv('ID10_CHAR_com_absolutos.csv', index=True)

# Exibir as métricas calculadas por classe e turno
print("Métricas de OCR por classe e turno com número absoluto de casos:")
print(metricas_turno_df)

print("\nMétricas de Detecção de Caracteres por classe e turno com número absoluto de casos:")
print(metricas_caracteres_turno_df)
