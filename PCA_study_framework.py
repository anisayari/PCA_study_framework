import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.decomposition import PCA
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__")), os.pardir)))
print(os.getcwd())


def PCA_study(df):
    # number of variable
    p = df.shape[1]
    print('[INFO] Number of Variable: {}'.format(p))
    # nombre d'observations
    n = df.shape[0]
    print('[INFO] Number of Variable: {}'.format(n))

    # print(data.columns)
    print('[INFO] DATA DONE')
    # data.info()

    print('[INFO] Scaling in progress...')
    min_max_scaler = preprocessing.StandardScaler()
    # min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    print('[INFO] Check if means stay near 0')
    print(np.mean(np_scaled, axis=0))
    print('[INFO] Check if standars deviation is equal at 1')
    print(np.std(np_scaled, axis=0, ddof=0))

    data = pd.DataFrame(np_scaled)
    print('[INFO] SCALING DONE')
    sns.set(style='darkgrid')

    print('[INFO] PCA in progress...')

    pca = PCA(svd_solver='full')
    data = pca.fit_transform(data)
    index_columns = ['PCA-{}'.format(i) for i in range(1, p + 1)]
    pca_components_df = pd.DataFrame(pca.components_, columns=df.columns, index=index_columns)
    print('[INFO] PCA DONE')

    # standardize these 3 new features
    print('[INFO] Standardization in progress...')
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    print('[INFO] Standardization DONE')

    print('[INFO] Explained Variance')
    eigval = (n - 1) / n * pca.explained_variance_
    print(eigval)
    # proportion de variance expliquée
    print(pca.explained_variance_ratio_)
    # scree plot
    plt.plot(np.arange(1, p + 1), eigval)
    plt.title("Scree plot")
    plt.ylabel("Eigen values")
    plt.xlabel("Factor number")
    plt.savefig('data/output/{}_Scree_plot.png'.format(timestr))
    plt.show()
    plt.close()
    # cumul de variance expliquée
    plt.plot(np.arange(1, p + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.title("Explained variance vs. # of factors")
    plt.ylabel("Cumsum explained variance ratio")
    plt.xlabel("Factor number")
    plt.savefig('data/output/{}_Explained_variance.png'.format(timestr))
    plt.show()
    plt.close()
    # seuils pour test des bâtons brisés
    bs = 1 / np.arange(p, 0, -1)
    bs = np.cumsum(bs)
    bs = bs[::-1]
    # test des bâtons brisés
    print(pd.DataFrame({'Val.Propre': eigval, 'Seuils': bs}))

    # contribution des individus dans l'inertie totale
    di = np.sum(data ** 2, axis=1)
    print(pd.DataFrame({'ID': df.index, 'd_i': di}).sort_values(['d_i'], ascending=False).head(10))

    # qualité de représentation des individus - COS2
    cos2 = np_scaled ** 2
    for j in range(p):
        cos2[:, j] = cos2[:, j] / di
    print(pd.DataFrame({'id': df.index, 'COS2_1': cos2[:, 0], 'COS2_2': cos2[:, 1]}).head())

    # vérifions la théorie - somme en ligne des cos2 = 1
    print(np.sum(cos2, axis=1))

    # contributions aux axes
    ctr = np_scaled ** 2
    for j in range(p):
        ctr[:, j] = ctr[:, j] / (n * eigval[j])

    print(pd.DataFrame({'id': df.index, 'CTR_1': ctr[:, 0], 'CTR_2': ctr[:, 1]}).head())

    # représentation des variables
    # racine carrée des valeurs propres
    sqrt_eigval = np.sqrt(eigval)

    # corrélation des variables avec les axes
    corvar = np.zeros((p, p))
    for k in range(p):
        corvar[:, k] = pca.components_[k, :] * sqrt_eigval[k]

    # afficher la matrice des corrélations variables x facteurs
    print('[INFO] Correlation matrices (variable x factor)')
    print(corvar)

    # on affiche pour les deux premiers axes
    print(('[INFO] plot first two axes'))
    print(pd.DataFrame({'id': df.columns, 'COR_1': corvar[:, 0], 'COR_2': corvar[:, 1]}).head())

    # cercle des corrélations
    fig, axes = plt.subplots(figsize=(15, 15))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    # affichage des étiquettes (noms des variables)
    for j in range(p):
        plt.annotate(df.columns[j], (corvar[j, 0], corvar[j, 1]))

    # ajouter les axes
    plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
    plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

    # ajouter un cercle
    cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
    axes.add_artist(cercle)
    # affichage
    fig.savefig('data/output/{}_PCA.png'.format(timestr))
    plt.show()
    plt.close()

    for index_ in pca_components_df.index:
        print('------------- {} -------------'.format(index_))
        dict_ = {}
        for col_ in pca_components_df.columns:
            dict_[col_] = pca_components_df.loc[index_, col_]
        d_view = [(v, k) for k, v in dict_.items()]
        d_view.sort(reverse=True)  # natively sort tuples by first element
        for v, k in d_view:
            pass
            # print("{}: {}".format(k,v))
        plt.figure(figsize=(15, 6))
        sns.barplot(pd.Series(list(zip(*d_view))[1]), pd.Series(list(zip(*d_view))[0]), palette="BuGn_r").set_title('PCA-{0}_importance_variable'.format(index_))
        plt.xticks(rotation=70)
        plt.savefig('data/output/{0}_PCA-{1}_importance_variable.png'.format(timestr, index_))
        plt.show()

    return data, pca_components_df
