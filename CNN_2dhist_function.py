#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importaciones de bibliotecas de sistema
import os
import gc
import time
import shutil

# Importaciones de bibliotecas de terceros
import wget
import scipy.signal
import h5py
import psutil
import ray

# Importaciones de TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
from keras import backend as K 

# Importaciones de sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.datasets import make_classification
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier

# Importaciones de pandas
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# Importaciones de matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from IPython.core.pylabtools import figsize, getfigs
import matplotlib.ticker as ticker


# Importaciones de seaborn
import seaborn as sns


# Importaciones de plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importaciones de numpy
import numpy as np
from scipy import stats

# Importaciones de astropy
from astropy.io import fits
from astropy.timeseries import LombScargle
from astropy.coordinates import SkyCoord
import astropy.units as u

# Importaciones para el equilibrio de los datos
from imblearn.keras import BalancedBatchGenerator
from imblearn.under_sampling import RandomUnderSampler

import joblib


class BalancedDataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras Sequence based data generator. 
       Suitable for building data generator for training and prediction.
    """
    def __init__(self, x, y, batch_size=64):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.classes = np.unique(y)
        self.class_indices = [np.where(y == i)[0] for i in self.classes]
        self.length = min([len(i) for i in self.class_indices]) // self.batch_size * len(self.classes)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for class_index in self.class_indices:
            i = idx % (len(class_index) // self.batch_size)
            batch_x.append(self.x[class_index[i * self.batch_size:(i + 1) * self.batch_size]])
            batch_y.append(self.y[class_index[i * self.batch_size:(i + 1) * self.batch_size]])
        return np.concatenate(batch_x), np.concatenate(batch_y)

    def on_epoch_end(self):
        for class_index in self.class_indices:
            np.random.shuffle(class_index)


# Funciones
def descarga_wget(database,ID,path_3,path_4):
    _,field,types,_ = ID.lower().split("-")
    try :
        if types=="ell":
            types="ecl"
        if database==4:
            if ((field =="blg") |(field =="gd"))&((types =="ecl")|(types =="lpv")|(types =="dsct")):
                url = "http://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/"+field+"/"+types+"/phot_ogle4/I/"+ ID +".dat"
                wget.download(url,path_4)
                return 1
            else:
                url = "http://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/"+field+"/"+types+"/phot/I/"+ ID +".dat"
                wget.download(url,path_4)
                return 1
                
        if database==3:
            url = "http://ftp.astrouw.edu.pl/ogle/ogle3/OIII-CVS/" +field+"/"+types+"/phot/I/"+ ID +".dat"
            wget.download(url,path_3)
            return 1
    except:
        return 0
            
@ray.remote
def review_open_data(nomb,path_datos,database):
        path= path_datos[database]
        try :
            df = pd.read_csv(f"{path}/{nomb}.dat",delim_whitespace=True,names=["jd","mag","err"])
            df_sigma = df.loc[(df["mag"] < np.mean(df["mag"]) + 3*np.std(df["mag"])) & ( df["mag"] > np.mean(df["mag"]) - 3*np.std(df["mag"]) )]
            if len(df_sigma)>2000:
                df_sigma = df_sigma.sample(2000,random_state=42).reset_index(drop=True)
            obs_eliminadas = len(df) - len(df_sigma)
            amplitud = df_sigma["mag"].max() - df_sigma["mag"].min()
            mag_mean = df_sigma["mag"].mean()
            mag_std = df_sigma["mag"].std()
            err_mean = df_sigma["err"].mean()
            err_std = df_sigma["err"].std()
            obs_final = len(df_sigma)
            obs_inicial = len(df)
            baseline = df["jd"].max() - df["jd"].min()
            cadence = df.sort_values(by="jd")["jd"].diff().mean()
            cadence_sigma = df_sigma.sort_values(by="jd")["jd"].diff().mean()
            return 1,nomb,database,obs_eliminadas,amplitud,mag_mean,mag_std,err_mean,err_std,obs_final,obs_inicial, baseline,cadence,cadence_sigma
        except:
            return 0,nomb,database,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan, np.nan

        
def ra_dec_to_degrees(ra_str, dec_str):
    # Convertir las coordenadas RA y DEC en objetos SkyCoord
    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))

    # Obtener las coordenadas en grados
    ra_deg = coord.ra.degree
    dec_deg = coord.dec.degree

    return ra_deg, dec_deg

def fase_datos(path_datos=None, database=None, nomb=None, per_vsx=None, df=None):
    
    # Si no se pasa dataframe, leer desde archivo (modo antiguo)
    if df is None:
        path = path_datos[database]
        df = pd.read_csv(
            f"{path}/{nomb}.dat",
            delim_whitespace=True,
            names=["jd", "mag", "err"]
        )

    # filtro sigma clipping (3 sigma)
    df_sigma = df.loc[
        (df["mag"] < np.mean(df["mag"]) + 3*np.std(df["mag"])) &
        (df["mag"] > np.mean(df["mag"]) - 3*np.std(df["mag"]))
    ]

    # limitar tamaño
    if len(df_sigma) > 2000:
        df_sigma = df_sigma.sample(2000, random_state=42).reset_index(drop=True)

    # fase
    fase_vsx = np.mod(df_sigma.jd, per_vsx) / per_vsx

    mag_vsx = df_sigma.mag
    t_vsx = df_sigma.jd
    err_vsx = df_sigma.err

    return fase_vsx, mag_vsx, t_vsx

    
    
def fill_subplot(ax, df_plot, star, data):
    """
    Rellena un subplot específico con los datos proporcionados.

    :param ax: El eje en el que se dibujará el subplot.
    :param df_plot: DataFrame con los datos para el subplot.
    :param star: Identificador de la estrella para este subplot.
    :param data: Diccionario con los datos de la imagen para el subplot.
    """
    n_obs_inicial,n_obs_final, phi = df_plot.loc[star, ["obs_final","bins", "phi"]]
    if n_obs_inicial > 2000:
        n_obs_inicial = 2000.0

    # Insertar gráfico (reemplazar con el código de gráfico real)
    ax.imshow(data["Number_M_data"][star], aspect="auto")

    # Construir el texto con los valores y luego aplicar LaTeX
    text_str = r'$n\prime_{\mathrm{obs}} = ' + str(round((n_obs_final/n_obs_inicial),2)) + \
               r'$, $\phi\prime = ' + f'{phi:.2f}$'
    ax.text(0.5, 1.05, text_str, transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', horizontalalignment='center', 
            bbox=dict(facecolor='white', alpha=0.5))


def make_2d_histogram(n_bins_x,n_bins_y,data_mag,data_fase,norm_max="max"):
    bins_x = np.linspace(0,1, n_bins_x) # Curves in phase between 0 and 2.
    bins_y = np.linspace( data_mag.min(), data_mag.max(), n_bins_y)
    hist_data, _xbins, _ybins = np.histogram2d(data_fase, data_mag, bins=(bins_x, bins_y))
    # Data in histogram is transposed, then transpose it just once:
    if norm_max=="max":
        norm_max = hist_data.max()
        hist_data_norm = hist_data / norm_max
        hist_data_transposed = hist_data_norm.transpose()
        hdu = fits.PrimaryHDU(data=hist_data_transposed)
        return hdu
    if norm_max=="none":
        hist_data_norm = hist_data
        hist_data_transposed = hist_data_norm.transpose()
        hdu = fits.PrimaryHDU(data=hist_data_transposed)
        return hdu
    
    else:
        norm_max = float(norm_max)
        hist_data[hist_data > norm_max ] = norm_max
        hist_data_norm = hist_data / norm_max
        hist_data_transposed = hist_data_norm.transpose()
        hdu = fits.PrimaryHDU(data=hist_data_transposed)
        return hdu

def split_random(df,numero_dividir,col_name):
    for types in df["types"].unique():
        df_var = df.loc[df["types"]==types].sample(numero_dividir,random_state=42)
        df_train,df_test = train_test_split(df_var,random_state=42,test_size=0.13)
        df_train,df_val = train_test_split(df_train,random_state=42,test_size=0.15)
        df.loc[df_train.index,col_name] = "train"
        df.loc[df_val.index,col_name] = "val"
        df.loc[df_test.index,col_name] = "test"
    return df
    

def split_data_balanced(df,numero_dividir):
    df["combined"] = list(zip(df["obs_final"],
                          df["amplitud"],
                          df["mag_mean"],
                          df["mag_std"],
                          df["field"],
                         df["err_mean"],
                         df["per"],
                         df["err_std"]))
    combined_weight = df['combined'].value_counts(normalize=True)
    df['combined_weight'] = df['combined'].apply(lambda x: combined_weight[x])
    subsample = df.sample(numero_dividir, weights=df['combined_weight'])
    for types in df["types"].unique():
        df_var = df.loc[df["types"]==types].sample(numero_dividir,
                                                             weights=df['combined_weight'],
                                                             random_state=42)
        df_train = df_var.sample(frac=0.8,
                         weights=df['combined_weight'],
                         random_state=42)
        df_var = df_var.drop(df_train.index)
        
        df_val = df_var.sample(frac=0.5,
                         weights=df['combined_weight'],
                         random_state=42)
        
        df_test = df_var.drop(df_val.index)
        
        df.loc[df_train.index,"entrenamiento_8mil_balanced"] = "train"
        df.loc[df_val.index,"entrenamiento_8mil_balanced"] = "val"
        df.loc[df_test.index,"entrenamiento_8mil_balanced"] = "test"
            
        
    return df
    
def balance_data(input_df, exclude_df):
    # Remove IDs from exclude_df in input_df and reset the index
    input_df = input_df[~input_df["ID"].isin(exclude_df["ID"])].reset_index(drop=True)

    # Find the minimum number of examples for any type
    min_examples_per_type = input_df.groupby("types").count().sort_values(["ID"])["ID"].values[0]

    # Create an empty DataFrame for the new balanced dataset
    balanced_df = pd.DataFrame()

    # Balance the data for each type
    for types in input_df["types"].unique():
        df_types = input_df.loc[input_df["types"]==types].reset_index(drop=True)
        if len(df_types) >= min_examples_per_type:
            df_types = df_types.sample(min_examples_per_type, random_state=42).reset_index(drop=True)
        balanced_df = pd.concat([balanced_df, df_types])

    # Add training data from exclude_df
    training_data = exclude_df.loc[(exclude_df["Train_8"] != "test")&
                                  (exclude_df["Train_8"] != "val")].reset_index(drop=True)
    balanced_df = pd.concat([balanced_df, training_data])

    # Find the balanced number for each type
    balanced_number = balanced_df.groupby("types").count().sort_values(by="ID", ascending=False)["ID"][0]

    # Adjust the number of examples for each type
    for types in balanced_df["types"].unique():
        df_types = balanced_df.loc[balanced_df["types"]==types].reset_index(drop=True)
        remainder = balanced_number % len(df_types)
        if balanced_number // len(df_types) > 1:
            repeat = balanced_number // len(df_types) - 1
            balanced_df = pd.concat([balanced_df, pd.concat([df_types] * repeat)])
            balanced_df = pd.concat([balanced_df, df_types.sample(remainder, random_state=42)])
        if balanced_number // len(df_types) == 1:
            balanced_df = pd.concat([balanced_df, df_types.sample(remainder, random_state=42)])
    balanced_df["Train_8"] = "train"
    input_df = input_df[~input_df["ID"].isin(balanced_df["ID"])].reset_index(drop=True)


    return input_df,balanced_df
    

def metrics_per_model(tests,name,path):
    # Load data
    data = h5py.File(f"{path}/Data.hdf5", 'r+')
    df_8mil = pd.read_csv(f"{path}/prueba_8mil.csv")
    idx_test = df_8mil.loc[df_8mil["Train_8"]=="test"].index.values
    test = df_8mil.loc[df_8mil["Train_8"]=="test"]
    test = test.drop(columns={"Train_8","aug","g","bins"})
    
    # Prepare data generator
    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow(
        data[name+"_data"][idx_test],
        data[name+"_label"][idx_test],
        batch_size=32
    )
    
    model = make_model()
    acc = []
    f1 = []
    rec = []
    prec = []
    for i, prueba in enumerate(tests):
        model.load_weights(f"{path}/{prueba}/cp.ckpt")
        prediction(test, test_gen, model, prueba)

        # Calculate F1 Score
        rec.append(recall_score(test_gen.y, test[f"label_predict_{prueba}"],average="macro"))
        f1.append(f1_score(test_gen.y, test[f"label_predict_{prueba}"],average="macro"))
        acc.append(accuracy_score(test_gen.y, test[f"label_predict_{prueba}"]))
        prec.append(precision_score(test_gen.y, test[f"label_predict_{prueba}"],average="macro"))

    return acc,f1,rec,prec
    
def data_augmented_parameter_creator(df):
    rng = np.random.default_rng(42) 

    # Mark duplicates based on 'ID' and assign 'aug' column
    df.loc[df.duplicated(subset="ID"), "aug"] = 1
    df.loc[df["aug"].isna(), "aug"] = 0

    # Count occurrences of each ID and merge with the original DataFrame
    df_aux = df.groupby("ID").count()[["RA"]].reset_index().rename(columns={"RA": "count"})
    df = df.merge(df_aux, how="left", on="ID")

    # Apply the augmented_to function to rows with more than one occurrence
    df.loc[df["count"] > 1].drop_duplicates(subset="ID").parallel_apply(
        lambda row: augmented_to(row["ID"], row["count"], df), axis=1
    )

    # Assign 'bins' values for rows with 'aug' == 1
    df.loc[df["aug"] == 1, "bins"] = df.loc[df["aug"] == 1].parallel_apply(
        lambda row: rng.choice(np.arange(int(row["obs_final"] * 0.5), int(row["obs_final"] * 0.9))), axis=1
    )
    
    df.loc[(df["bins"] < 60) & (df["aug"] == 1), "bins"] = df.loc[(df["bins"] < 60) & (df["aug"] == 1), "obs_final"].apply(lambda x: np.random.randint(60, x))
    
    df["g"] = rng.uniform(low=0 + 1/32, high=1 - 1/32, size=len(df))*df["per"]
    
    df.loc[df["Train_8"].isna(),"Train_8"] = "train"



    # Create a DataFrame without augmented data
    df_without_aug = df.loc[df["aug"] == 0].reset_index(drop=True)
    

    return df, df_without_aug
    
def randomize_per(group):
    # Mezclar aleatoriamente los valores de 'per' sin reemplazo
    shuffled_per = group['per'].sample(frac=1, random_state=1).reset_index(drop=True)

    # Reasignar los valores mezclados a la columna 'per'
    group['per'] = shuffled_per.values

    return group



def make_random_period(df,n_star_split="auto"):
    if n_star_split=="auto":
        for split in ["train","val","test"]:
            df_var = df.loc[(df["Train_8"]==split)&(df["aug"]==0)]
            n_star_split = df_var.groupby("categorical_label").count()["ID"].max()
            if np.isnan(n_star_split):
                continue
            if int(1+n_star_split/7) <= df_var.groupby("types").count()["ID"].min():
                df_randoms = df_var.groupby("types").sample(int(np.ceil(n_star_split/7)))
                df_randoms = df_randoms.sample(n_star_split)
            else:
                print("error")
            df_randoms["categorical_label"] = 7
            df_randoms = df_randoms.groupby('types', group_keys=True).apply(randomize_per).reset_index(drop=True)
            df_randoms["types"] = df_randoms["types"] +"_"+ "random"
            df = pd.concat([df,df_randoms])
                
    else:
        for split in ["train"]:
            df_var = df.loc[(df["Train_8"]==split)&(df["aug"]==0)]
            if int(1+ n_star_split/7) <= df_var.groupby("types").count()["ID"].min():
                df_randoms = df_var.groupby("types").sample(1+int(n_star_split /7))
                df_randoms = df_randoms.sample(n_star_split)
            else:
                min_class_name = df_var.groupby("types").count().sort_values(by="ID").index.values[0]
                df_var_min_class = df_var.loc[df_var["types"]==min_class_name]
                df_var = df_var.loc[df_var["types"]!=min_class_name]
                df_randoms = df_var.groupby("types").sample(int(1 + (n_star_split - len(df_var_min_class))/6))
                df_randoms = pd.concat([df_randoms,df_var_min_class])
                df_randoms = df_randoms.sample(n_star_split)


                
            df_randoms["categorical_label"] = 7
            df_randoms = df_randoms.groupby('types', group_keys=True).apply(randomize_per).reset_index(drop=True)
            df_randoms["types"] = df_randoms["types"] +"_"+ "random"
            df = pd.concat([df,df_randoms])

    return df

def plot_obs_dist(df, split_name,path):
    sns.set_context("paper")
    gyr = ['#FFCF3D', "#129675", "#890B96"]
    sns.set_palette(gyr)

    columns = ["obs_final", "amplitud", "mag_mean", "field", "err_mean", "per", "mag_std", "err_std"]
    labels = [r'$n_{obs}$', r'$Amplitude$', 'Mean Magnitude', "Field", 'Mean Error', 'Period', 
              'Magnitude standard deviations', 'Error standard deviations']
    log_scales = [True, True, False, False, True, True, True, False]
    x_ticks = [[10**2, 10**3, 10**4], [10**-1, 1, 10**1], None, None, [10**-2, 10**-1, 10**0], [10**-1, 10**1, 10**3],
               None, [0, 0.3, 0.6]]
    y_scale_log = [True, True, True, True, True, True, True, True]

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    # Crear las leyendas una vez
    legend_labels = df[split_name].unique()
    legend_colors = gyr[:len(legend_labels)]

    for i, ax in enumerate(axes.flatten()):
        sns.histplot(ax=ax, data=df, x=columns[i], hue=split_name, bins=30, log_scale=log_scales[i], fill=True, common_norm=True, multiple="stack")
        ks_test = stats.kstest(df.loc[df["Train_8"]=="train"][columns[i]],df.loc[df["Train_8"]=="test"][columns[i]])
        ks_test = np.round(ks_test[0],3)
        ks_val = stats.kstest(df.loc[df["Train_8"]=="train"][columns[i]],df.loc[df["Train_8"]=="val"][columns[i]])
        ks_val = np.round(ks_val[0],3)

        ax.set(xlabel=labels[i], ylabel="")
        ax.set_title(f' K-S: T={ks_test}, V={ks_val}')
        if x_ticks[i] is not None:
            ax.set_xticks(x_ticks[i])
        if y_scale_log[i]:
            ax.set_yscale("log")

            # Calcular y establecer 4 y-ticks para cada subplot
            ymin, ymax = ax.get_ylim()
            yticks = np.logspace(np.log10(ymin+1e-3), np.log10(ymax), 4)  # Añadir un pequeño offset para evitar log(0)            print(yticks)# Añadir un pequeño offset para evitar log(0)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: '${{10^{{{:d}}}}}$'.format(int(np.log10(y)))))

        ax.get_legend().remove()  # Remove the individual legend from each subplot

    plt.rc('xtick', labelsize=13) 

    # Crear una leyenda para toda la figura
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=label)
                       for label, color in zip(["Train", "Test", "Validation"], legend_colors)]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_labels), bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    plt.savefig(f"{path}/Distribution_splits.pdf", bbox_inches="tight")
    plt.show()
    return 
  

def make_lc_hist_with_time(nomb,
                 per_vsx,
                 path_datos,
                 database,
                 aug,
                 rng,
                 g,
                 bins):
    start_time = time.time()

    # Inicialización de tiempos
    times = {'task': [],
             'time': []}

    # Carga de datos
    t0 = time.time()
    path = path_datos[database]
    df = pd.read_csv(f"{path}/{nomb}.dat", delim_whitespace=True, names=["d", "mag", "e"])
    times['task'].append('load_csv')
    times['time'].append(time.time() - t0)

    # Limpieza de datos
    t0 = time.time()
    df_sigma = df.loc[(df["mag"] < np.mean(df["mag"]) + 3*np.std(df["mag"])) & (df["mag"] > np.mean(df["mag"]) - 3*np.std(df["mag"]))].reset_index(drop=True)
    df_sigma["fase"] = np.mod(df_sigma.d, per_vsx) / per_vsx
    if len(df_sigma) > 2000:
        df_sigma = df_sigma.sample(2000, random_state=42)
    times['task'].append('Prepare Data')
    times['time'].append(time.time() - t0)
    # Proceso dependiendo de la augmentación
    t0 = time.time()
    #  make_2d_histogram
    hdu = make_2d_histogram(32+1, 32+1, df_sigma.mag, df_sigma.fase, norm_max="max")
    times['task'].append('histogram')
    times['time'].append(time.time() - t0)
    total_time = time.time() - start_time
    times_df = pd.DataFrame(times)
    times_df['percentage'] = (times_df['time'] / total_time) * 100
    print(times_df)
    return hdu.data, times_df

    

def plot_histograms(estrellas_plot,path_datos, norm,path):
    
    fig, ax = plt.subplots(len(estrellas_plot), 2, figsize=(6, 10), sharex="col")
    for i in range(len(estrellas_plot)):
        fase, mag, t_vsx = fase_datos(path_datos, estrellas_plot["database"][i], estrellas_plot["ID"][i], estrellas_plot["per"][i])

        ax[i, 0].set_ylim(mag.max(), mag.min()- (mag.max() - mag.min())/4)
        ax[i, 0].set_yticks(np.linspace(mag.min() + (mag.max() - mag.min()) / 10, mag.max() - (mag.max() - mag.min()) / 10, 4))
        ax[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        sns.scatterplot(x=fase, y=mag, c=t_vsx, s=15, ax=ax[i, 0])
        ax[i, 0].set(xlabel='', ylabel='')
        # Agregar el nombre de la estrella y el tipo en la esquina superior izquierda
        text = f"{estrellas_plot['ID'][i]} [{estrellas_plot['types'][i]}]"
        ax[i, 0].text(0.05, 0.95, text, transform=ax[i, 0].transAxes, fontsize=9, verticalalignment='top')

        hdu = make_2d_histogram(32 + 1, 32 + 1, mag, fase, norm_max=norm)
        ax[i, 1].imshow(hdu.data, interpolation='nearest', aspect='auto')
        ax[i, 1].set_yticklabels([])
        ax[i, 1].set_xticklabels([])
        ax[i,1].set_xlabel("")
        ax[i,1].set_ylabel("")

    fig.text(0.5, 0, "Phase", size=13)
    fig.text(-0.01, 0.5, "I Mag", size=13, rotation=90)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(f"{path}hist_2d_fase.pdf", bbox_inches="tight", pad_inches=0)
    return hdu



@ray.remote
def make_lc_hist(nomb,
                 per_vsx,
                 path_datos,
                 database,
                 aug,
                 rng,
                 g,
                 bins):
        path= path_datos[database]
        df = pd.read_csv(f"{path}/{nomb}.dat",delim_whitespace=True,names=["d","mag","e"])
        df_sigma = df.loc[(df["mag"] < np.mean(df["mag"]) + 3*np.std(df["mag"])) & ( df["mag"] > np.mean(df["mag"]) - 3*np.std(df["mag"]) )].reset_index(drop=True)
        if int(aug) == 0:
            df_sigma["fase"] = np.mod(df_sigma.d, per_vsx) / per_vsx
            if len(df_sigma)>2000:
                n_obs = int(np.random.uniform(50, 2000))
                df_sigma = df_sigma.sample(n_obs, random_state=42).sort_values("d").reset_index(drop=True)          
            hdu =make_2d_histogram(32+1,32+1,df_sigma.mag,df_sigma.fase, norm_max="max")
            return hdu.data
        
        if int(aug) == 1:
            df_sigma["fase"] =  np.mod(df_sigma.d - g, per_vsx) / per_vsx
            df_sigma["mag"] = df_sigma["mag"] + rng.normal(0, df_sigma["e"],len(df_sigma))
            df_sigma["fase_bin"] = pd.cut(df_sigma["fase"],bins=int(bins))
            df_bins = pd.DataFrame(df_sigma.groupby("fase_bin")["fase"].mean())
            df_bins["mag"] = df_sigma.groupby("fase_bin")["mag"].mean()
            df_bins["e"] = df_sigma.groupby("fase_bin")["e"].mean()
            df_bins["d"] = df_sigma.groupby("fase_bin")["d"].mean()
            hdu =make_2d_histogram(32+1,32+1,df_bins.mag,df_bins.fase)
            return hdu.data

def create_hdf5(df,path_datos,rng):
    results_ids = []
    rng = np.random.default_rng(42)
    for i in range(len(df)):
        hdu = make_lc_hist.remote(df["ID"][i],
                                  df["per"][i],
                                  path_datos,
                                  df["database"][i],
                                  df["aug"][i],
                                  rng,
                                  df["g"][i],
                                  df["bins"][i])
        results_ids.append((hdu))
    x = np.empty((len(df), 32, 32))
    for i,key in enumerate(results_ids):
        ima = ray.get(key)
        x[i] = ima
    x = np.expand_dims(x, axis=3)
    return x

def make_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), input_shape=(32, 32, 1),activation="relu",padding="same"),
    tf.keras.layers.Conv2D(16, (3,3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3),activation="relu",padding="same"),
    tf.keras.layers.Conv2D(32, (3, 3),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024,activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512,activation="relu"),
    tf.keras.layers.Dropout(0.3),
   # tf.keras.layers.Dense(1, activation='sigmoid')
    tf.keras.layers.Dense(8, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=0.1), loss="sparse_categorical_crossentropy", metrics=['acc'])
    return model

def train_models(df_lista, keys_lista, data, prueba_8mil,path,epochs=200, use_balanced_generator=False):
    tf.random.set_seed(42)

    validation_datagen = ImageDataGenerator()
    idx_val = prueba_8mil.loc[prueba_8mil['Train_8']=="val"].index.values
    val_label = data['Number_CEP_label'][idx_val]
    val_data = data["Number_CEP_data"][idx_val]
    val_gen = validation_datagen.flow(val_data, val_label, batch_size=128, shuffle=True)

    for df, test_name in zip(df_lista, keys_lista):
        K.clear_session()
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

        model_history_log_file = f"{path}history_softmax_{'batchBalanced_' if use_balanced_generator else ''}{test_name}.csv"
        csv_logger = CSVLogger(model_history_log_file, append=False)

        checkpoint_path = f"{path}training_softmax_{'batchBalanced_' if use_balanced_generator else ''}{test_name}/cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=False, verbose=1)

        callbacks = [csv_logger, cp_callback, early_stopping]
        
        if use_balanced_generator:
            idx_train = df.loc[(df["Train_8"]!="test")&(df["Train_8"]!="val")&(df["aug"]==0)].index.values
        else :
            idx_train = df.loc[(df["Train_8"]!="test")&(df["Train_8"]!="val")].index.values

        bz = 128 #int((len(idx_train) * 96)/ len(prueba_8mil.loc[prueba_8mil['Train_8']=="train"]))
    
        train_label = data[f'{test_name}_label'][idx_train]
        
        train_data = data[f'{test_name}_data'][idx_train]
        
        if use_balanced_generator:
            train_gen = BalancedDataGenerator(train_data, train_label, batch_size=128)
        else:
            train_datagen = ImageDataGenerator()
            train_gen = train_datagen.flow(train_data, train_label, batch_size=bz, shuffle=True)

        model = make_model()
        print(f"Use balanced Generator [{use_balanced_generator}] \n Data: {len(train_data)} \n -----------------------------------------------------------------------------------")
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    return

    
def augmented_to(ID,count,df):
    rng = np.random.default_rng(42)
    a = (np.linspace(0,9,count)*100).astype(int)
    np.sort(rng.uniform(low=0 + 1/32, high=1 - 1/32, size=1000))[a]
    df.loc[df["ID"]==ID,"g"] = np.sort(rng.uniform(low=0 + 1/32, high=1 - 1/32, size=1000))[a]
    return

def plot_accuracy_and_loss(path,file_names, title_names, amarillo_train, purpura_val, output_file="training_.pdf"):
    plt.rcParams["figure.figsize"] = (18,8)
    sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2})
    sns.set_style("white")
    fig, axs = plt.subplots(2, len(file_names), sharex="col", sharey="row")
    metrics = ["acc", "val_acc", "loss", "val_loss"]
    labels = ['Training D.A.', 'Validation D.A.', 'Training D.A.', 'Validation D.A.']
    labels_batch = ['Training B.B.', 'Validation B.B.', 'Training B.B.', 'Validation B.B.']    
    colors = [amarillo_train, purpura_val, amarillo_train, purpura_val]
    linestyles = ["-.", ".as", "dashdot", "dashdot"]

    for i, file_name in enumerate(file_names):
        batch_name = file_name.split("softmax")[0]+"softmax_batchBalanced"+file_name.split("softmax")[1]
        df = pd.read_csv(f"{path}/{file_name}.csv")
        if file_name != "history_softmax_Number_CEP":
            df_batch = pd.read_csv(f"{path}/{batch_name}.csv")

        for j, metric in enumerate(metrics):
            sns.lineplot(ax=axs[j//2, i], data=df, x="epoch", y=metric, 
                         color=colors[j], label=labels[j], linestyle="solid")
            if i > 0:
                sns.lineplot(ax=axs[j//2, i], data=df_batch, x="epoch", y=metric, 
                             color=colors[j], label=labels_batch[j], linestyle="dashed")

        axs[0,i].set_title(title_names[i])
        axs[0,i].set_ylim([0.6,1])
        axs[1,i].set_ylim([0.6,1])
        axs[0,i].set_yticks(np.linspace(0.6,1,8))
        axs[1,i].set_yticks(np.linspace(0,0.9,8))
        axs[0,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[1,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[1,i].set_xlabel('Epoch')

        axs[0,i].get_legend().remove()
        axs[1,i].get_legend().remove()

    axs[0,0].set_ylabel('Accuracy')
    axs[1,0].set_ylabel('Loss')

    handles, labels = axs[1,2].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.1,0.5), ncol=4, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(output_file, bbox_inches="tight")
    return
    

def plot_accuracy_and_loss(path,file_names, title_names, amarillo_train, purpura_val, output_file="training_.pdf"):
    plt.rcParams["figure.figsize"] = (18,9)
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2})
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(2, len(file_names), sharex="col", sharey="row")
    metrics = ["acc", "val_acc", "loss", "val_loss"]
    labels = ['Training D.A.', 'Validation D.A.', 'Training D.A.', 'Validation D.A.']
    labels_batch = ['Training B.B.', 'Validation B.B.', 'Training B.B.', 'Validation B.B.']    
    colors = [amarillo_train, purpura_val, amarillo_train, purpura_val]
    linestyles = ["-.", ".as", "dashdot", "dashdot"]

    for i, file_name in enumerate(file_names):
        batch_name = file_name.split("softmax")[0]+"softmax_batchBalanced"+file_name.split("softmax")[1]
        df = pd.read_csv(f"{path}/{file_name}.csv")
        if file_name != "history_softmax_Number_CEP":
            df_batch = pd.read_csv(f"{path}/{batch_name}.csv")

        for j, metric in enumerate(metrics):
            sns.lineplot(ax=axs[j//2, i], data=df, x="epoch", y=metric, 
                         color=colors[j], label=labels[j], linestyle="solid")
            if i > 0:
                sns.lineplot(ax=axs[j//2, i], data=df_batch, x="epoch", y=metric, 
                             color=colors[j], label=labels_batch[j], linestyle="dashed")

        axs[0,i].set_title(title_names[i])
        axs[0,i].set_ylim([0.6,1])
        axs[1,i].set_ylim([0.6,1])
        axs[0,i].set_yticks(np.linspace(0.6,1,8))
        axs[1,i].set_yticks(np.linspace(0,0.9,8))
        axs[0,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[1,i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axs[1,i].set_xlabel('')
        axs[1, i].xaxis.set_major_locator(MaxNLocator(4, integer=True, min_n_ticks=4))
        axs[0, i].xaxis.set_major_locator(MaxNLocator(4, integer=True, min_n_ticks=4))

        axs[0,i].get_legend().remove()
        axs[1,i].get_legend().remove()

    axs[0,0].set_ylabel('Accuracy',fontsize=25)
    axs[1,0].set_ylabel('Loss',fontsize=25)

    handles, labels = axs[1,2].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.15,0.5), ncol=4, fancybox=True, shadow=True)
    fig.text(0.5, 0.01, 'Epoch', ha='center', va='center', fontsize=25)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(output_file, bbox_inches="tight")
    return
    
def run_analysis(tests,titles,name,path):
    # Load data
    data = h5py.File(f"{path}/Data.hdf5", 'r+')
    df_8mil = pd.read_csv(f"{path}/prueba_8mil.csv")
    idx_test = df_8mil.loc[df_8mil["Train_8"]=="test"].index.values
    test = df_8mil.loc[df_8mil["Train_8"]=="test"]
    test = test.drop(columns={"Train_8","aug","g","bins"})
    
    # Prepare data generator
    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow(
        data[name+"_data"][idx_test],
        data[name+"_label"][idx_test],
        batch_size=32
    )
    
    sns.set_context("paper",font_scale=3)
    model = make_model()

    num_tests = len(tests)
    rows = 3  # Ahora queremos 2 filas
    cols = 2  # Y 3 columnas

    fig, ax = plt.subplots(rows, cols, figsize=(25, 25), sharey="row")
    plt.subplots_adjust(wspace=0, hspace=0.2, right=0.7)

    # Aplanar el array de ejes para iterar fácilmente
    ax = ax.ravel()

    for i, prueba in enumerate(tests):
        model.load_weights(f"{path}/{prueba}/cp.ckpt")
        prediction(test, test_gen, model, prueba)

        # Calculate F1 Score
        f1 = f1_score(test_gen.y, test[f"label_predict_{prueba}"], average='weighted')

        # Plots
        array, annot = C_M(test_gen.y, test[f"label_predict_{prueba}"])
        sns.heatmap(array, annot=annot, fmt='', vmin=0, vmax=np.sum(array, axis=1)[0], cmap="BuPu",
                    annot_kws={"fontsize":25}, linewidth=1, ax=ax[i], cbar=False)
        ax[i].set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
        ax[i].set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
        ax[i].set_yticklabels(['ELL', 'M', 'CEP', 'DST', 'E', 'LPV', 'RR', "Rndm"])
        ax[i].set_xticklabels(['ELL', 'M', 'CEP', 'DST', 'E', 'LPV', 'RR', "Rndm"])
        # Add title and F1 Score
        ax[i].set_title(f'{titles[i]}\nF1 Score: {f1:.3f}')

    # Eliminar el último subplot si el número de tests no llena todos los subplots
    if num_tests < rows * cols:
        fig.delaxes(ax[-1])

    # Etiqueta general para el eje y
    fig.text(-0.02, 0.5, 'True Label', va='center', rotation='vertical', fontsize=30)

    # Etiqueta general para el eje x
    fig.text(0.5, -0.02, 'Predicted Label', ha='center', fontsize=30)

    fig.tight_layout(pad=0)
    plt.savefig(f"{path}CM.pdf", bbox_inches="tight")
    return test
    
    
def undersampling_CM(prueba,titles,name,path):
    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 2})
    data = h5py.File(f"{path}/Data.hdf5", 'r+')
    df_8mil = pd.read_csv(f"{path}/prueba_8mil.csv")
    idx_test = df_8mil.loc[df_8mil["Train_8"]=="test"].index.values
    test = df_8mil.loc[df_8mil["Train_8"]=="test"]
    test = test.drop(columns={"Train_8","aug","g","bins"})
    
    # Prepare data generator
    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow(
        data[name+"_data"][idx_test],
        data[name+"_label"][idx_test],
        batch_size=64
    )
    
    model = make_model()

    model.load_weights(f"{path}/{prueba}/cp.ckpt")
    
    prediction(test, test_gen, model, prueba)
    
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))


        # Calculate F1 Score
    f1 = f1_score(test_gen.y, test[f"label_predict_{prueba}"], average='weighted')

        # Plots
    array, annot = C_M(test_gen.y, test[f"label_predict_{prueba}"])
    sns.heatmap(array, annot=annot, fmt='', vmin=0, vmax=np.sum(array, axis=1)[0], cmap="BuPu",
                annot_kws={"fontsize":15}, linewidth=1, ax=ax, cbar=False)
    ax.set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
    ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
    ax.set_yticklabels(['ELL', 'M', 'CEP', 'DST', 'E', 'LPV', 'RR', "Rndm"])
    ax.set_xticklabels(['ELL', 'M', 'CEP', 'DST', 'E', 'LPV', 'RR', "Rndm"])
    # Add title and F1 Score
    ax.set_title(f'{titles}\nF1 Score: {f1:.3f}')

    # Etiqueta general para el eje y
#    fig.text(-0.02, 0.5, 'True Label', va='center', rotation='vertical', fontsize=20)

    # Etiqueta general para el eje x
 #   fig.text(0.5, -0.02, 'Predicted Label', ha='center', fontsize=20)

    plt.savefig(f"{path}UndersamplingCM.pdf", bbox_inches="tight")
    return test
    
def train_random_forest(X_train, y_train, X_test, y_test,path, n_estimators):
    # Crea el clasificador Random Forest
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    # Entrena el clasificador
    clf.fit(X_train, y_train)
    
    # Predice las clases para el conjunto de test
    y_pred = clf.predict(X_test)
    
    # Crea una figura y ejes para la trama
    fig, ax = plt.subplots(figsize=(8, 8))

    array, annot = C_M(y_test, y_pred)
    sns.heatmap(array, annot=annot, fmt='', vmin=0, vmax=np.sum(array, axis=1)[0], cmap="BuPu",
                annot_kws={"fontsize":15}, linewidth=1, ax=ax, cbar=False)
    
    ax.set_yticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
    ax.set_yticklabels(['ELL', 'M', 'CEP', 'DST', 'E', 'LPV', 'RR'], fontsize=15)
    ax.set_ylabel('True Label', fontsize=20)
    
    ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5])
    ax.set_xticklabels(['ELL', 'M', 'CEP', 'DST', 'E', 'LPV', 'RR'], fontsize=15)
    ax.set_xlabel('Predicted Label', fontsize=20)
    
    # Add title and F1 Score
    # Calculate F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    ax.set_title(f'CNN+RF\nF1 Score: {f1:.2f}', fontsize=20)

    fig.tight_layout(pad=0)
    plt.savefig(f"{path}CNN_And_RF.pdf", bbox_inches="tight")
    
    # Guarda el modelo entrenado en el mismo path
    joblib.dump(clf, f"{path}trained_random_forest.joblib")
    return



def prediction(df,image_gen,model,prueba):
    grupos = ['ELL', 'Mira', 'cep', 'dsct', 'ecl', 'lpv', 'rrlyr',"random"]
    label_predict = []
    porcentaje_predict = []
    nombres = []
    for i in model.predict(image_gen.x):
        idx = np.argmax(i)
        label_predict.append(np.argmax(i))
        porcentaje_predict.append(i[idx])
        nombres.append(grupos[np.argmax(i)])
    df[f"label_predict_{prueba}"] = label_predict
    df[f"porcentaje_predict_{prueba}"] = porcentaje_predict
    df[f"nombres_predict_{prueba}"] = nombres
    return

def C_M(label,predict_label):
    array = np.array(tf.math.confusion_matrix(label,predict_label) )
    df = pd.DataFrame(array)
    perc = df.copy()
    cols=perc.columns.values
    perc[cols]=perc[cols].div(perc[cols].sum(axis=1), axis=0).multiply(100)
    annot=df.round(2).astype(str) + "\n" + perc.round(1).astype(str) + "%"
    return array,annot

def metricas(labels,predict):
    print("Accuracy:", "%0.2f" % metrics.accuracy_score(labels,predict))
    print("macro precision: ","%0.2f" %  metrics.precision_score(labels,predict, average='macro'))
    print("macro recall: ","%0.2f" %  metrics.recall_score(labels,predict, average='macro'))
    print("macro F1: ","%0.2f" %  metrics.f1_score(labels,predict, average='macro'))
    print(metrics.classification_report(labels,predict, digits=2))
    report = metrics.classification_report(labels,predict, output_dict=True, digits=2)
    return report

    

