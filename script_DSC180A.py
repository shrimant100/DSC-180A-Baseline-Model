import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

def preprocessing(df):
    """ (Optional) Helper method to calculate total packet sizes
    Params:
        df: dataframe object
    Returns:
        processed dataframe object
    """
    
    def packet_size_total(packets):
        """ Returns the total size (sum) of packets
        """
        return list(map(lambda x : int(x), list(filter(lambda x : x != '', packets.split(";")))))
    
    df['sum_packets'] = df.packet_sizes.apply(packet_size_total).apply(sum)

    
    return df

def chunking(df, step):
    """ Chunks the df into 'time-sequences' of size 'step'
    Params:
        df: dataframe object to be chunked
        step: step size
    Returns:
        formatted dataframe object with summary stats for each chunk
    """
    bytes_1_2 = []
    bytes_2_1 = []

    packets_1_2 = []
    packets_2_1 = []

    total_packets = []

    start = df.Time.iloc[0]
    end = df.Time.iloc[len(df) - 1]
    curr = start
    
    i = 0
    while i < len(df):
        curr = df.Time.iloc[i]
        sub_df = df.loc[(curr <= df.Time) & (df.Time <= curr + step)]

        bytes_1_2.append(sub_df['1->2Bytes'].mean())
        bytes_2_1.append(sub_df['2->1Bytes'].mean())
        packets_1_2.append(sub_df['1->2Pkts'].mean())
        packets_2_1.append(sub_df['2->1Pkts'].mean())
        total_packets.append(sub_df.sum_packets.mean())
        
        i += len(sub_df)
        
    df_formatted = pd.DataFrame(data={'1->2Bytes':bytes_1_2, '2->1Bytes':bytes_2_1, 
                                        '1->2Pkts':packets_1_2, '2->1Pkts':packets_2_1,})
                                        
        
    return df_formatted


def train(train_csv):
    """ Trains the model
    Params:
        train_csv: the path of dataframe object used to train the model
    
    Returns:
        trained classifier object
    """
    combined_df = pd.read_csv(train_csv)
    X, y = combined_df.drop(columns=['streaming']), combined_df.streaming
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    clf = RandomForestClassifier(max_depth = 3, random_state=42).fit(X_train, y_train)
    predicted_train = clf.predict(X_train)
    predicted_test = clf.predict(X_test)

    print("Training Accuracy: " + str(np.mean(predicted_train == y_train)))
    print("Validation Accuracy: " + str(np.mean(predicted_test == y_test)))

    return clf

def test(train_csv, test_csv):
    """ Tests model performance on input data
    Params:
        train_csv: the path of dataframe object used to train the model
        test_csv: the path of dataframe object used to test the model

    Returns:
        Performance of model
    """

    clf = train(train_csv)

    temp = pd.read_csv(test_csv)
    temp_proc = preprocessing(temp)
    temp_chunked = chunking(temp_proc, 10)

    actual = [0]*len(temp_chunked)
    predicted = clf.predict(temp_chunked)

    accuracy = (predicted == actual).mean()
    print("Accuracy: " + str(accuracy))


    
