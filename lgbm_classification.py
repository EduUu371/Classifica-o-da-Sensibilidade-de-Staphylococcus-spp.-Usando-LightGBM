# inicio

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import os
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import lightgbm as lgbm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder




SA_6538_A = np.load('dados\Staphylococcusaureus_6538_Plastico_A_180926-101242.npy')
SA_6538_B = np.load('dados\Staphylococcusaureus_6538_Plastico_B_180926-101357.npy')

SA_7903_A = np.load('dados\Staphylococcusaureus_7903_Plastico_A_180926-100656.npy')
SA_7903_B = np.load('dados\Staphylococcusaureus_7903_Plastico_B_180926-100548.npy')

SA_25923_A = np.load('dados\Staphylococcusaureus_25923_Plastico_A_Contaminado_180926-101609.npy')
SA_25923_B = np.load('dados\Staphylococcusaureus_25923_Plastico_B_Contaminado_180926-101712.npy')

SA_MRSA2_A = np.load('dados\Staphylococcusaureus_MRSA2_Plastico_A_180926-095538.npy')
SA_MRSA2_B = np.load('dados\Staphylococcusaureus_MRSA2_Plastico_B_180926-095714.npy')

SA_MRSA7_A = np.load('dados\Staphylococcusaureus_MRSA7_Plastico_A_180926-100020.npy')
SA_MRSA7_B = np.load('dados\Staphylococcusaureus_MRSA7_Plastico_B_180926-100250.npy')

SA_6538_A = SA_6538_A.transpose(1, 2, 0)
SA_6538_B = SA_6538_B.transpose(1, 2, 0)
SA_MRSA2_A = SA_MRSA2_A.transpose(1, 2, 0)
SA_MRSA2_B = SA_MRSA2_B.transpose(1, 2, 0)
SA_MRSA7_A = SA_MRSA7_A.transpose(1, 2, 0)
SA_MRSA7_B = SA_MRSA7_B.transpose(1, 2, 0)
SA_7903_A = SA_7903_A.transpose(1, 2, 0)
SA_7903_B = SA_7903_B.transpose(1, 2, 0)
SA_25923_A = SA_25923_A.transpose(1, 2, 0)
SA_25923_B = SA_25923_B.transpose(1, 2, 0)

print(f"SA_6538_A...........(x:{SA_6538_A.shape[1]},  y:{SA_6538_A.shape[0]},  λ:{SA_6538_A.shape[2]})")
print(f"SA_6538_B...........(x:{SA_6538_B.shape[1]},  y:{SA_6538_B.shape[0]},  λ:{SA_6538_B.shape[2]})")
print(f"SA_MRSA2_A..........(x:{SA_MRSA2_A.shape[1]}, y:{SA_MRSA2_A.shape[0]}, λ:{SA_MRSA2_A.shape[2]})")
print(f"SA_MRSA2_B..........(x:{SA_MRSA2_B.shape[1]}, y:{SA_MRSA2_B.shape[0]}, λ:{SA_MRSA2_B.shape[2]})")
print(f"SA_MRSA7_A..........(x:{SA_MRSA7_A.shape[1]}, y:{SA_MRSA7_A.shape[0]}, λ:{SA_MRSA7_A.shape[2]})")
print(f"SA_MRSA7_B..........(x:{SA_MRSA7_B.shape[1]}, y:{SA_MRSA7_B.shape[0]}, λ:{SA_MRSA7_B.shape[2]})")
print(f"SA_7903_A...........(x:{SA_7903_A.shape[1]},  y:{SA_7903_A.shape[0]},  λ:{SA_7903_A.shape[2]})")
print(f"SA_7903_B...........(x:{SA_7903_B.shape[1]},  y:{SA_7903_B.shape[0]},  λ:{SA_7903_B.shape[2]})")
print(f"SA_25923_A..........(x:{SA_25923_A.shape[1]}, y:{SA_25923_A.shape[0]}, λ:{SA_25923_A.shape[2]})")
print(f"SA_25923_B..........(x:{SA_25923_B.shape[1]}, y:{SA_25923_B.shape[0]}, λ:{SA_25923_B.shape[2]})")




# functions ----------------------------------------------------------------------------------------------------



def standard_normal_variate(data_matrix: np.ndarray):
    """
    Aplica a normalização Z-score a cada linha da matriz.
    
    Parâmetros:
    - data_matrix: Matriz de entrada em formato numpy.
    
    Retorno:
    - Matriz normalizada conforme SNV.
    """
    normalized_matrix = np.zeros_like(data_matrix)
    for row_index in range(data_matrix.shape[0]):
        row_mean = np.mean(data_matrix[row_index, :])
        row_std = np.std(data_matrix[row_index, :])
        normalized_matrix[row_index, :] = (data_matrix[row_index, :] - row_mean) / row_std
    return normalized_matrix



def centralize_by_mean(data_matrix: np.ndarray):
    """
    Centraliza os dados em zero subtraindo a média de cada linha.
    
    Parâmetros:
    - data_matrix: Matriz de entrada em formato numpy.
    
    Retorno:
    - Matriz centralizada.
    """
    centralized_matrix = np.zeros_like(data_matrix)
    for row_index in range(data_matrix.shape[0]):
        row_mean = np.mean(data_matrix[row_index, :])
        centralized_matrix[row_index, :] = data_matrix[row_index, :] - row_mean
    return centralized_matrix



def apply_savitzky_golay(data_matrix: np.ndarray, poly_order=2, window_size=21, derivative_order=1, boundary_mode='wrap'):
    """
    Aplica o filtro Savitzky-Golay à matriz.
    
    Parâmetros:
    - data_matrix: Matriz de entrada em formato numpy.
    - poly_order: Ordem do polinômio para o filtro.
    - window_size: Tamanho da janela do filtro.
    - derivative_order: Ordem da derivada a ser calculada.
    - boundary_mode: Modo de tratamento das bordas.
    
    Retorno:
    - Matriz suavizada/derivada conforme o filtro de Savitzky-Golay.
    """
    return savgol_filter(data_matrix, window_size, poly_order, deriv=derivative_order, mode=boundary_mode)



def filters(CUBE: np.array, poly_order=2, window_size=21, derivative_order=1, boundary_mode='wrap'):
    """
    Aplica métodos para filtrar, normalizar e suavizar uma matriz tridimensional.
    
    Parâmetros:
    - CUBE: Cubo de dados (matriz tridimensional).
    - poly_order: Ordem do polinômio para o filtro Savitzky-Golay.
    - window_size: Tamanho da janela do filtro Savitzky-Golay.
    - derivative_order: Ordem da derivada para o filtro Savitzky-Golay.
    - boundary_mode: Modo de tratamento das bordas.
    
    Retorno:
    - Matriz filtrada, normalizada e suavizada.
    """
    centralized_matrix = centralize_by_mean(CUBE)
    smoothed_matrix = apply_savitzky_golay(centralized_matrix, poly_order, window_size, derivative_order, boundary_mode)
    normalized_matrix = standard_normal_variate(smoothed_matrix)
    return normalized_matrix




def rgbscale(image):

    return (image * 255).astype(np.uint8)




def hsi2matrix(matrix: np.ndarray):

    return matrix.T.reshape((matrix.shape[1] * matrix.shape[2], matrix.shape[0]), order='F')



def matrix2hsi(matrix: np.ndarray, rows: int, cols: int):
    """
        Reorganizar a matriz 2D em uma matriz 3D
        forma final (-1, linhas, colunas)

        Parâmetros:
            - matriz: Matriz em formato numpy
            - rows: Número de linhas
            - cols: Número de colunas
        Retorno:
            - Matriz 3D
    """
    return matrix.T.reshape(-1, rows, cols)



def rev_idx_array(idx, rmv, shape=None, tfill=None):
    """
        Criar um array de idx de acordo com
        idx e rmv, matrizes de índices
    """
    if shape is None:
        out = np.zeros(idx.shape[0] + rmv.shape[0])
    else:
        out = np.zeros(shape)

    out[rmv] = 0

    if tfill is not None:
        for i, row in enumerate(idx):
            out[row] = tfill[i]
    else:
        out[idx] = 1

    return out.astype(int)



def sum_idx_array(idx):

    ind_r = []
    np.arange(idx.shape[0])
    for i, (j, ind) in enumerate(zip(idx, np.arange(idx.shape[0]))):
        if j != ind:
            ind_r.append(i)

    return np.delete(idx, ind_r), np.array(ind_r)



def read_HSI(dado):

    X = dado
    y = dado [:,:,-1]
    (f"X shape: {X.shape}\ny shape: {y.shape}")

    return X, y



def normalize_image(image):
    #Normaliza a imagem para garantir que os valores estejam no intervalo [0, 1].

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0

    return np.clip(image, 0, 1)



def moveAxis(data):
    '''
        ex: Se a matriz original tinha dimensões 
        (altura, largura, canais), 
        após a chamada de moveAxis 
        a matriz terá dimensões 
        (canais, altura, largura).
    '''
    data = np.moveaxis(data, 2, 0)

    return data



def clusteriza(CUBE = None, PCS = None):
#Retorna o rotulo de cada pixel ao seu correpondente clustercluster
#É executado empregando as classes com os algoritmos PCA e K-means

    scores = PCA(n_components = PCS).fit_transform(hsi2matrix(CUBE))
    # 'n_components()'Número de componentes principais a serem mantidos.
    # 'fit_transform()' Ajusta o modelo com CUBE e aplique a redução de dimensionalidade em CUBE.
    # 'hsi2matrix()' transforma o hipercubo em matriz 2D

    #Retorna a matriz com a remoção do fundo
    return  KMeans(n_clusters=2, init='k-means++', n_init=5, max_iter=300).fit(scores).labels_



def realIdx(idx, c):

    out = np.arange(idx.shape[0])
    for idx, (rid, vec) in enumerate(zip(out, idx)):
        if vec != c:
            out[idx] = -1
    out[out == -1] = 0

    return out



def getClusters(image = None,INDEX = None, c = None, rgb = None):

    ind = realIdx(INDEX, c)
    out_i = np.concatenate((ind, ind, ind), axis=0).reshape((3, *(image.shape)))

    image = MinMaxScaler(feature_range=(0, 1)).fit_transform(image)
    image = np.stack((image, image, image), axis=2)

    image[out_i[0] != 0, 0] = rgb[0]
    image[out_i[1] != 0, 1] = rgb[1]
    image[out_i[2] != 0, 2] = rgb[2]

    return image



def Remove_Background(CUBE=None):

    CUBE_amostra = moveAxis(CUBE)
    CLUSTERS_amostra = clusteriza(CUBE_amostra, 2) + 1
    imagem_amostra = CUBE_amostra[50, :, :]

    cluster_1 = getClusters(imagem_amostra, CLUSTERS_amostra, 1, (1, 0, 0))
    cluster_2 = getClusters(imagem_amostra, CLUSTERS_amostra, 2, (0, 1, 0))
    cluster_1 = normalize_image(cluster_1)
    cluster_2 = normalize_image(cluster_2)


    fig_clusters, axes_clusters = plt.subplots(1, 2, figsize=(15, 8))
    axes_clusters[0].axis('off')
    axes_clusters[1].axis('off')
    axes_clusters[0].imshow(cluster_1)
    axes_clusters[1].imshow(cluster_2)

    fig_clusters.savefig('fig_clusters.png', bbox_inches='tight')
    plt.close(fig_clusters)

    fig_result, axes_result = plt.subplots(2, 1, figsize=(10, 8))#, tight_layout=True)
    fig_result.suptitle('Selecione o cluster de interesse')

    ax_button1 = plt.axes([0.2, 0.48, 0.2, 0.05])
    ax_button2 = plt.axes([0.6, 0.48, 0.2, 0.05])
    button1 = Button(ax_button1, 'Cluster 1', color='red', hovercolor='lightcoral')
    button2 = Button(ax_button2, 'Cluster 2', color='green', hovercolor='lightgreen')

    axes_result[0].imshow(plt.imread('fig_clusters.png'))
    os.remove('fig_clusters.png')
    axes_result[0].axis('off')
    axes_result[1].axis('off')

    
    cluster_escolhido = None
    ind_amostra = None
    result_amostra = None

    def on_click_cluster1(event):
        nonlocal cluster_escolhido, ind_amostra, result_amostra
        cluster_escolhido = 1
        select_cluster(cluster_escolhido)


    def on_click_cluster2(event):
        nonlocal cluster_escolhido, ind_amostra, result_amostra
        cluster_escolhido = 2
        select_cluster(cluster_escolhido)


    def select_cluster(cluster_escolhido):
        nonlocal ind_amostra, result_amostra
        print((realIdx(CLUSTERS_amostra, int(cluster_escolhido))[:320]))
        ind_amostra, rm_amostra = sum_idx_array(realIdx(CLUSTERS_amostra, int(cluster_escolhido)))
        sample_cluster = rev_idx_array(ind_amostra, rm_amostra)
        result_amostra = getClusters(imagem_amostra, sample_cluster, 0, (0, 0, 0))
        result_amostra = normalize_image(result_amostra)
        axes_result[1].imshow(result_amostra)
        axes_result[1].set_title('Resultado')
        plt.draw()
        plt.pause(0.7)
        plt.close(fig_result)


    button1.on_clicked(on_click_cluster1)
    button2.on_clicked(on_click_cluster2)

    plt.show()

    return ind_amostra



# Aplicação do Filtro ------------------------------------------------------------------------------------------

def plot_cubo(CUBE, title='', colors=["black", "navy", "blue", "red", "orange", "yellow", "white"]):
    # CUBE = Dados(y, x, λ)
    # title = Título da plotagem
    # colors = range de cores da imagem
    
    # Segmentar o range de cores entre 0 e 1
    positions = np.linspace(0, 1, len(colors))

    # Criar o mapa de cores personalizado
    cmap_custom = LinearSegmentedColormap.from_list("custom_colormap", list(zip(positions, colors)))

    # Definir os comprimentos de onda em nanômetros
    num_bands = CUBE.shape[2]
    wavelengths = np.linspace(900, 2500, num_bands)

    # Inicializar a figura e os eixos
    fig, (ax_imagem, ax_espectro) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)
    camada_inicial = CUBE[:, :, 0]
    cax = ax_imagem.imshow(camada_inicial, cmap=cmap_custom)
    fig.colorbar(cax, ax=ax_imagem, orientation='vertical', shrink=0.8)
    ax_imagem.set_title('Arraste o slider para alternar entre camadas')
    
    # Configurar o slider
    ax_slider = plt.axes([0.1, 0.07, 0.38, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Banda Espectral', 0, num_bands - 1, valinit=0, valstep=1)

    # Subplot para o espectro
    spectrum_line, = ax_espectro.plot([], [])
    ax_espectro.set_title('Espectro do Pixel')
    ax_espectro.set_xlabel('Comprimento de Onda λ em nm')
    ax_espectro.set_ylabel('Intensidade')

    # Função para atualizar a camada exibida
    def update(val):
        camada_atual = int(slider.val)
        ax_imagem.images[0].set_array(CUBE[:, :, camada_atual])
        ax_imagem.set_title(f'Banda Espectral {camada_atual}/{num_bands - 1}')
        fig.canvas.draw_idle()

        # Converter o valor do slider para o comprimento de onda correspondente
        camada_atual_convertida = wavelengths[camada_atual]
        
        # Remover as linhas verticais existentes
        for line in ax_espectro.get_lines():
            if line.get_linestyle() == '--':
                line.remove()
        
        # Adicionar a nova linha pontilhada
        ax_espectro.axvline(camada_atual_convertida, color='r', linestyle='--')
        ax_espectro.figure.canvas.draw_idle()
    

    # Função para atualizar o espectro do pixel sob o cursor
    def on_hover(event):
        if event.inaxes == ax_imagem:
            ix_spec, iy_spec = int(event.xdata), int(event.ydata)
            if 0 <= ix_spec < CUBE.shape[1] and 0 <= iy_spec < CUBE.shape[0]:
                # Atualizar o gráfico do espectro
                spectrum = CUBE[iy_spec, ix_spec, :]
                spectrum_line.set_data(wavelengths, spectrum)
                ax_espectro.set_xlim(900, 2500)
                ax_espectro.set_ylim(np.min(spectrum), np.max(spectrum))

                # Converter o valor do slider para o comprimento de onda correspondente
                camada_atual = int(slider.val)
                camada_atual_convertida = wavelengths[camada_atual]
                
                # Remover as linhas verticais existentes
                for line in ax_espectro.get_lines():
                    if line.get_linestyle() == '--':
                        line.remove()
                
                # Adicionar a nova linha pontilhada
                ax_espectro.axvline(camada_atual_convertida, color='b', linestyle='--')
                ax_espectro.figure.canvas.draw_idle()

    # Conectar o evento de mudança de valor do slider à função update
    slider.on_changed(update)
    # Conectar o evento de movimento do mouse à função on_hover
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    plt.show()









# Visualizar os dados antes da filtragem do ruído.
plot_cubo(SA_7903_A, "Dados Não Filtrados")

# Lista das amostras de entrada e saída
input_output_pairs = {
    "SA_6538_A": "FSA_6538_A",
    "SA_6538_B": "FSA_6538_B",
    "SA_MRSA2_A": "FSA_MRSA2_A",
    "SA_MRSA2_B": "FSA_MRSA2_B",
    "SA_MRSA7_A": "FSA_MRSA7_A",
    "SA_MRSA7_B": "FSA_MRSA7_B",
    "SA_7903_A": "FSA_7903_A",
    "SA_7903_B": "FSA_7903_B",
    "SA_25923_A": "FSA_25923_A",
    "SA_25923_B": "FSA_25923_B"
}
# Aplicando a função filters a cada par de entrada e saída
for input_var, output_var in input_output_pairs.items():
    globals()[output_var] = filters(globals()[input_var], poly_order=2, window_size=21, derivative_order=1, boundary_mode='wrap')

# Visualizar os dados após da filtragem do ruído.
plot_cubo(FSA_7903_A, "Dados Filtrados")









# Remov backgroud ----------------------------------------------------------------------------------------------

dados_filtrados = [
    "FSA_6538_A", "FSA_6538_B", "FSA_MRSA2_A", "FSA_MRSA2_B",
    "FSA_MRSA7_A", "FSA_MRSA7_B", "FSA_7903_A", "FSA_7903_B",
    "FSA_25923_A", "FSA_25923_B"
]

dados_tratados = {}

# Verifica se os dados já estão slvos
if os.path.isfile('data_without_background.npz'):
    npzfile = np.load('data_without_background.npz')
    dados_tratados = dict(npzfile)
else:
    for df in dados_filtrados:
            IDX = Remove_Background(globals()[df])
            dados_tratados[f"IDX_{df}"] = IDX
    # Salva os dados tratados no arquivo
    np.savez('data_without_background.npz', **dados_tratados)

# Agora todos os resultados estão armazenados no dicionário 'resultados'









# get_xy -------------------------------------------------------------------------------------------------------

def get_xy(CUBE, index, target, razão):

    # CUBE = CUBE sem modificaçao
    # index = os pixels dejesejado sem fundo
    # target = targetget
    # razão = porcentagem do conjunto de teste
    # Retorna na cequencia y_test, y_train, X_test, X_train

    CUBE = moveAxis(CUBE)
    data = hsi2matrix(CUBE)

    y_test = np.array([])
    y_train = np.array([])

    id_train, id_test = train_test_split(index, test_size = razão, shuffle = True)

    # X_test = np.concatenate((xtest, data[id_test]))
    # X_train = np.concatenate((xtrain, data[id_train]))

    X_test = data[id_test]
    X_train = data[id_train]

    y = np.ones(id_test.shape) * target
    y_test = np.concatenate((y_test, y))

    y = np.ones(id_train.shape) * target
    y_train = np.concatenate((y_train, y))

    return X_train, y_train, X_test, y_test

x_nr_1, y_nr_1, _, _ = get_xy(FSA_6538_A,  dados_tratados['IDX_FSA_6538_A'],  0, 1)
x_r_1,  y_r_1,  _, _ = get_xy(FSA_MRSA2_A, dados_tratados['IDX_FSA_MRSA2_A'], 1, 1)
x_nr_2, y_nr_2, _, _ = get_xy(FSA_7903_A,  dados_tratados['IDX_FSA_7903_A'],  0, 1)
x_r_2,  y_r_2,  _, _ = get_xy(FSA_MRSA7_A, dados_tratados['IDX_FSA_MRSA7_A'], 1, 1)
x_nr_3, y_nr_3, _, _ = get_xy(FSA_25923_A, dados_tratados['IDX_FSA_25923_A'], 0, 1)


x_nr_4, y_nr_4, _, _ = get_xy(FSA_6538_B,  dados_tratados['IDX_FSA_6538_B'],  0, 1)
x_r_4,  y_r_4,  _, _ = get_xy(FSA_MRSA2_B, dados_tratados['IDX_FSA_MRSA2_B'], 1, 1)
x_nr_5, y_nr_5, _, _ = get_xy(FSA_7903_B,  dados_tratados['IDX_FSA_7903_B'],  0, 1)
x_r_5,  y_r_5,  _, _ = get_xy(FSA_MRSA7_B, dados_tratados['IDX_FSA_MRSA7_B'], 1, 1)
x_nr_6, y_nr_6, _, _ = get_xy(FSA_25923_B, dados_tratados['IDX_FSA_25923_B'], 0, 1)

xtrain = np.concatenate([x_nr_1, x_r_1, x_nr_2, x_r_2])#, x_nr_3])
ytrain = np.concatenate([y_nr_1, y_r_1, y_nr_2, y_r_2])#, y_nr_3])

xtest  = np.concatenate([x_nr_4,  x_r_4, x_nr_5,  x_r_5])#, x_nr_6])
ytest  = np.concatenate([y_nr_4,  y_r_4, y_nr_5,  y_r_5])#, y_nr_6])

print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

x_concat = np.concatenate([xtrain, xtest])
y_concat = np.concatenate([ytrain, ytest])
print(x_concat.shape, y_concat.shape)











# LightGbm -----------------------------------------------------------------------------------------------------------

# Modelo
clf_lgbm = lgbm.LGBMClassifier(boosting_type='gbdt', is_unbalance=True, force_col_wise=True)
clf_lgbm.fit(xtrain, ytrain)
predict = clf_lgbm.predict(xtest)

# Mapeamento das classes
class_mapping = {
    0: 'Não Resistente',
    1: 'Resistente'
}
# Convertendo os números das classes para os nomes correspondentes
ytest_names = [class_mapping[y] for y in ytest]
predict_names = [class_mapping[p] for p in predict]
# Criar uma lista de rótulos das classes
class_labels = [class_mapping[i] for i in range(len(class_mapping))]

# Imprimir as métricas
skl_report = classification_report(ytest_names, predict_names)
print(skl_report)
print()

# Plotar a matriz de confusão para o fold atual
conf_matrix = confusion_matrix(ytest, predict)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title(f"Matriz de Confusão")
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.show()

# Plotar a curva ROC multiclasse para o fold atual usando Yellowbrick
from yellowbrick.classifier import ROCAUC
encoder = LabelEncoder()
ytrain_encoded = encoder.fit_transform(ytrain)
visualizer = ROCAUC(clf_lgbm, micro=False, macro=False, per_class=True, classes=class_labels, is_fitted=True)
visualizer.fit(xtrain, ytrain_encoded)
visualizer.score(xtest, ytest)
visualizer.show()

input('Fim!')