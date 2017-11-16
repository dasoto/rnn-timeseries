from process_data import clean_data, create_index, pivot_data
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def data_lstm(X,y,lookback=1):
    newX = []
    newY = []
    for i in range(len(y)-lookback):
        newX.append(X[i:i+1+lookback,:])
        newY.append(y[i+lookback])
    return np.array(newX), np.array(newY)


def create_rnn():
    model = Sequential()
    model.add(LSTM(return_sequences=True, input_shape=(169,141), units=64))
    model.add(Dropout(0.2))

    model.add(LSTM(
        units=32,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print('compilation time : {}'.format(time.time() - start))
    return model


def train_rnn(df,date_predict,epochs=100):
    #Preparting data to Train
    #print('Preparing data to Train')
    date_p = datetime.datetime.strptime(date_predict, "%Y-%m-%d").date()
    date_limit = (date_p - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    date_start = (date_p - datetime.timedelta(days=42)).strftime('%Y-%m-%d')
    sk = df[:date_limit].copy()
    sk = sk_df.drop(['TRADEDATE', 'RTENERGY'], axis=1)
    sk.hourofday = sk.hourofday.dt.seconds/3600
    sk = sk[date_start:]
    y = sk.pop('DAENERGY').values
    X = temp.values[:,1:]
    X,y = data_lstm(X,y,168)

    model = create_rnn()

    #X = X.reshape(X.shape[0],X.shape[1],1)
    filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    filepath="best_model_lstm.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    # #Training Model
    model.fit(
        X,
        y,
        batch_size=1,  #168
        epochs=epochs,
        validation_split=0.05, callbacks=[early_stop, checkpoint])

    #preparing data to Predict
    return model, X, y


def predict_next_day(df,date_predict, filename):
    model = load_model(filename)
    date_p = datetime.datetime.strptime(date_predict, "%Y-%m-%d").date()
    date_limit = (date_p - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    df = df.copy()
    df_new = create_features(df)
    df_new = df_new.dropna()
    df_new = df_new[['Ph-1', 'Ph-2', 'Ph-3', 'Ph-24', 'Ph-25', 'Ph-48', 'Ph-49',
                'Ph-72','Ph-73', 'Ph-96', 'Ph-97', 'Ph-120', 'Ph-121', 'Ph-144',
                'Ph-145', 'Ph-168']]

    input_p = df_new[date_predict].copy()
    pred = []
    ph1 = np.zeros(25)
    ph2 = np.zeros(25)
    ph3 = np.zeros(25)

    ph1[0] = input_p.iloc[0]['Ph-1']
    ph2[0] = input_p.iloc[0]['Ph-2']
    ph2[1] = input_p.iloc[1]['Ph-2']
    ph3[0] = input_p.iloc[0]['Ph-3']
    ph3[1] = input_p.iloc[1]['Ph-3']
    ph3[2] = input_p.iloc[2]['Ph-3']

    for x in range(24):
        X_pred = input_p.iloc[x].values
        X_pred[0]=ph1[x]
        if x>1:
            X_pred[1]=ph2[x]
        if x>2:
            X_pred[2]=ph3[x]
        p = model.predict(X_pred.reshape(1,16,1))[0][0]
        pred.append(p)
        ph1[x+1]=p
        ph2[x+1]=ph1[x]
        ph3[x+1]=ph2[x]

    pred = np.array(pred)
    results = df[date_predict].copy()
    # pred_ada = model.predict(input_p.values)
    results['forecast'] = pred
    RMSE = get_rmse(df[date_predict]['DAENERGY'], pred)
    # print('RMSE for AdaBoost Model: ', RMSE)
    # #preparing to plot
    # #print('Ploting results')
    fig, ax = plt.subplots(figsize=(9,4))
    # npre = 24
    ax.set(title='DAENERGY '+ 'RMSE: {:.3f} for {}'.format(RMSE, date_predict), xlabel='Date', ylabel='Price DAENERGY')
    #
    # # Plot data points
    df.loc[date_predict, 'DAENERGY'].plot(ax=ax, style='o', label='Observed')
    #
    # # Plot predictions
    results.forecast.plot(ax=ax, style='r--', label='forecast')
    # #ci = predict_ci.loc[date_predict]
    # #ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
    legend = ax.legend(loc='lower right')

    return results, RMSE


def get_rmse(pred,real):
    return np.sqrt(((pred-real)**2).sum()/pred.shape[0])

def create_date_range(start_date, end_date):
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    dates = []
    cur_dt = start_dt
    dates.append(start_date)
    while cur_dt<end_dt:
        cur_dt = cur_dt + datetime.timedelta(days=1)
        dates.append(cur_dt.strftime('%Y-%m-%d'))
    return dates


def create_features(df):
    df = df.copy()
    features = ['Ph-1', 'Ph-2', 'Ph-3', 'Ph-24', 'Ph-25', 'Ph-48', 'Ph-49',
                'Ph-72','Ph-73', 'Ph-96', 'Ph-97', 'Ph-120', 'Ph-121', 'Ph-144',
                'Ph-145', 'Ph-168']
    deltas = [1, 2, 3, 24, 25, 48, 49, 72, 73, 96, 97, 120, 121, 144, 145, 168]

    for col, shift in zip(features, deltas):
        df[col] = df.DAENERGY.shift(shift)
    return df


def MAPE(y_true, y_pred):
    return 1/len(y_true)*(np.abs((y_true-y_pred))/y_true*100).sum()

if __name__ == '__main__':
    df = pivot_data('data/Data.txt')
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(df.DAENERGY)
    df.DAENERGY = scaler.transform(df.DAENERGY)
    df = clean_data(df)
    df = create_index(df)
    model, X, y = train_rnn(df,'2017-10-01',epochs=20)
    results , RMSE = predict_next_day(df,'2017-10-01', 'best_model.hdf5')
    y_true = scaler.inverse_transform(results.DAENERGY.values.reshape(1,-1))[0]
    y_pred = scaler.inverse_transform(results.forecast.values.reshape(1,-1))[0]
    print(MAPE(y_true, y_pred))
