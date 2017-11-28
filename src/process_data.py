import pandas as pd
import numpy as np

def pivot_data(filename, since_date='2017-07-05', to_date='2017-10-31'):
    df = pd.read_csv(filename, sep="\t")
    new_df = pd.pivot_table(df, index = ['TRADEDATE','HOUROFWEEK'],
                            columns='REGION')
    mynew_df = new_df.TEMPERATURE.copy()
    mynew_df['LOAD'] = new_df.LOAD.CYWG.values
    mynew_df['DAENERGY'] = new_df.DAENERGY.CYWG.values
    mynew_df['RTENERGY'] = new_df.RTENERGY.CYWG.values
    mynew_df['ISOWIND'] = new_df.ISOWIND.CYWG.values
    mynew_df['OUTAGE'] = new_df.OUTAGE.CYWG.values
    mynew_df = mynew_df.reset_index()
    mynew_df.TRADEDATE = pd.to_datetime(mynew_df.TRADEDATE)

    mynew_df = mynew_df[(mynew_df.TRADEDATE >= since_date)]
    mynew_df = mynew_df[(mynew_df.TRADEDATE < to_date)]
    mynew_df.sort_values(['TRADEDATE','HOUROFWEEK'], inplace=True)
    return mynew_df

def clean_data(df):
    i = df.shape[0]
    toadd = pd.DataFrame(index=[i], columns= df.columns)
    toadd2 = pd.DataFrame(index=[i+1], columns= df.columns)
    toadd3 = pd.DataFrame(index=[i+2], columns= df.columns)
    toadd4 = pd.DataFrame(index=[i+3], columns= df.columns)
    # toadd5 = pd.DataFrame(index=[i+4], columns= df.columns)
    # toadd6 = pd.DataFrame(index=[i+5], columns= df.columns)
    # toadd7 = pd.DataFrame(index=[i+6], columns= df.columns)

    for x in df.columns.values:
        toadd[x] = df[(df.TRADEDATE=='2017-08-15')&(df.HOUROFWEEK==56)][x].values
    toadd.HOUROFWEEK = 57
    for x in df.columns.values:
        toadd2[x] = df[(df.TRADEDATE=='2017-08-15')&(df.HOUROFWEEK==61)][x].values
    toadd2.HOUROFWEEK = 62
    for x in df.columns.values:
        toadd3[x] = df[(df.TRADEDATE=='2017-10-25')&(df.HOUROFWEEK==92)][x].values
    toadd3.HOUROFWEEK = 93
    for x in df.columns.values:
        toadd4[x] = df[(df.TRADEDATE=='2017-10-26')&(df.HOUROFWEEK==109)][x].values
    toadd4.HOUROFWEEK = 110
    df = pd.concat([df, toadd])
    df = pd.concat([df, toadd2])
    df = pd.concat([df, toadd3])
    df = pd.concat([df, toadd4])
    df = df.reset_index()
    df.sort_values(['TRADEDATE','HOUROFWEEK'], inplace = True)
    return df

def create_index(df):
    df['hourofday'] = df.HOUROFWEEK.apply(lambda x:
                                          np.timedelta64(x%(24*(x//24)),'h')
                                          if x>=24 else np.timedelta64(x,'h'))
    df['new_index'] = df.TRADEDATE + df.hourofday
    df.loc[df.hourofday =='00:00:00','new_index'] = df.loc[df.hourofday =='00:00:00'].new_index + \
    np.timedelta64(1,'D')
    df.set_index(['new_index'], inplace=True)
    df.index.name='dt'
    df = df.drop(['index'],axis=1)
    return df


if __name__ == '__main__':
    df = pivot_data('data/Data.txt')
