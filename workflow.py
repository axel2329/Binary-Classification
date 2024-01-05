import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE

##########################################
#            PRE PROCESSING              #
##########################################

# This function will be used to convert the .txt in .csv in order to facilitate further operations using pandas, such as filling missing values.
def convert_csv(dataset):
    filename = dataset.split(".")[0]
    extension = dataset.split(".")[1]
    if extension == "txt" :
        with open(dataset, "r") as txt_file :
            lines = txt_file.readlines()
        header = lines[0].strip().split(', ')
        data = [line.strip().split(', ') for line in lines[1:]]

        with open(os.path.join(filename + ".csv"), 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(header)
            csv_writer.writerows(data)


def pre_processing(data_path) :
    # Check the format and check if there is an "id" column to get rid of it
    convert_csv(data_path)
    try :
        dataset = pd.read_csv(data_path, index_col="id") 
    except :
        dataset = pd.read_csv(data_path)

    for header in dataset.columns :

        # Fill missing values differently regarding the type of the data and Normalize data for float data.

        if dataset[header].dtype == "int64" :   # Int can be transformed into Float to facilitate the computation in a single case
            dataset[header] = dataset[header].astype("float")

        elif dataset[header].dtype == "float64":
            median = dataset[header].median()
            dataset[header].fillna(median, inplace=True) 
            dataset[header] = (dataset[header] - dataset[header].mean()) / dataset[header].std()   
        
        if dataset[header].dtype != "float64" and dataset[header].dtype != "int64":
            # Deal with "\tno", "\tyes" and " yes" values (spotted in "dm class") or with "\tcdk"" in Classification". More generally, deal with string columns where there errors might happen.
            try :
                dataset[header] = dataset[header].str.replace('\t', '')   
                dataset[header] = dataset[header].str.replace(' ', '')
            except :
                pass

            # LabelEncoder to encode in different classes the different options available for the column. The output format is "Int32"
            encoder = LabelEncoder()
            dataset[header] = encoder.fit_transform(dataset[header])

    assert dataset.isnull().any().sum() == 0   # Ensure no missing value left

    return dataset

##########################################
#           FEATURE SELECTION            #
##########################################

def feature_selection_RFE(models, Nb_features, x, y):
    relevant_features = np.zeros((len(models), x.shape[1]))
    for i in range(len(models)) :
        if models[i] == "Logistic Regression" :
            estimator = LogisticRegression(max_iter=1000)
        
        elif models[i] == "Random Forest":
            estimator = RandomForestClassifier()
        
        elif models[i] == "Gradient Boosting":
            estimator = GradientBoostingClassifier()

        else :
            raise TypeError(" Model not recognized ")
        
        selector = RFE(estimator=estimator, n_features_to_select=Nb_features)
        selector = selector.fit(x,y)

        relevant_features[i,:] = selector.ranking_
    return relevant_features

def feature_selection_KBest(x,y,K):
    selector = SelectKBest(score_func=f_classif, k=K)
    x_transformed = selector.fit_transform(x, y)
    dataset_transformed = np.column_stack((x_transformed, y))
    feature_relevance = np.argsort(selector.scores_)[::-1]
    return dataset_transformed, feature_relevance




##########################################
#               TRAINING                 #
##########################################

# Function to split the dataset in train and test sets.
def pre_train(dataset) :
    try :
        x = dataset.to_numpy()[:,:-1]
        y = dataset.to_numpy()[:,-1]
    except :
        x = dataset[:,:-1]
        y = dataset[:,-1]
        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return x_train, x_test, y_train, y_test

# For the Deep Network (tensorflow)
def build_model(x_train) :
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=(x_train.shape[1],), activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

def training(x_train, x_test, y_train, choix_model) :
    
    if choix_model == "Logistic Regression" :
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
    elif choix_model == "Linear SVC":
        model = make_pipeline(StandardScaler(), LinearSVC())
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    
    elif choix_model == "Random Forest":
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
    
    elif choix_model == "Gradient Boosting":
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

    elif choix_model == "Neural Networks":
        model = build_model(x_train)
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=tf.keras.metrics.Recall())   # We want to maximize the recall 
        model.fit(x_train, y_train, batch_size=32, epochs=80, validation_split=0.2, verbose=0)  
        y_proba = model.predict(x_test)
        threshold = 0.5
        y_pred = (y_proba > threshold).astype(int)

    else :
        raise TypeError(" Model not recognized ")
    
    return y_pred


##########################################
#          DISPLAY     RESULTS           #
##########################################


def score_prediction(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return accuracy, recall

# Function to plot confusion matrices for One or Several models
def plot_conf_matrix(y_test, Y_pred, model):
    n = len(model)
    assert len(Y_pred) == n
    fig, axes = plt.subplots(1, n, figsize=(16,4))
    for i in range(n) :
        conf_matrix = confusion_matrix(y_test, Y_pred[i], labels=[1,0]) # ATTENTION here need to be [0,1] for banknote dataset, [1,0] for kidney disease
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, ax=axes[i])
        axes[i].set_xlabel("Predicted Labels")
        axes[i].set_ylabel("True Labels")
        axes[i].set_title(model[i])

    plt.tight_layout()
    plt.show()

# Function to plot bar chart of our metrics
def plot_metrics(model, accuracy, recall):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    bars_1 = axes[0].bar(model, accuracy, color='skyblue', align='center')
    axes[0].bar_label(bars_1, fmt='%.2f', fontsize=8, color='black')
    axes[0].set_xticks(np.arange(len(model)))
    axes[0].set_xticklabels(model, rotation=20, ha='right')
    axes[0].set_title("Accuracy")

    bars_2 = axes[1].bar(model, recall, color='skyblue', align='center')
    axes[1].bar_label(bars_2, fmt='%.2f', fontsize=8, color='black')
    axes[1].set_xticks(np.arange(len(model)))
    axes[1].set_xticklabels(model, rotation=20, ha='right')
    axes[1].set_title("Recall")
    
    plt.tight_layout()
    plt.show()

# Correlation Matrix used for feature selection
def correlation_matrix(dataset):
    correlation_matrix = np.corrcoef(dataset.T)
    plt.figure(figsize=(12,12))
    sns.heatmap(correlation_matrix, annot=True, cmap="Blues", annot_kws={"size": 6})
