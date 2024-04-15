from datetime import date
from re import S
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector

#get and split data
w_df = pd.read_csv('seattle-weather.csv')
print(w_df.head(1))
w_df['date'] = pd.to_datetime(w_df['date'])
train_df = w_df[w_df.columns].iloc[:1096]
val_df = w_df[w_df.columns].iloc[1096:]

#args for datasets
target_col = 'wind'
feat_cols = ['date','precipitation','temp_max','temp_min','weather']
min_max = (-1,1)
seq_len = 8


class SeqDataset(Dataset):
    def __init__(self,dataframe,target_col,feat_cols,min_max, datetimes, seq_length):
        self.dataframe = dataframe
        self.target = target_col
        self.feats = feat_cols
        print(self.feats)
        self.min_max = min_max
        self.seq_length = seq_length
        self.datetimes = datetimes
        
        self.x = torch.Tensor(self.feat_preprocessing().values).float()
        self.y = torch.Tensor(self.dataframe[self.target]).float()
        

    def feat_preprocessing(self):
        #check if datetimes should be converted to unix time
        init_dataframe = self.dataframe.copy()
        feat_df = init_dataframe[self.feats]

        if type(self.datetimes) != bool:
            raise ValueError('datetimes argument passed is not of type bool (True/False)')

        elif self.datetimes == True:
            def date2unix():
                datetime_cols = feat_df.select_dtypes(include='datetime').columns
                
                #double check if there are datetime cols
                if len(datetime_cols) > 0:
                    unix_epoch = pd.Timestamp('1970-01-01')
                    feat_df.loc[:,datetime_cols] = (feat_df.loc[:,datetime_cols]-unix_epoch)//pd.Timedelta(seconds=1)
                else:
                    raise TypeError("No columns with type 'datetime' were found, even though parameter 'datetimes' was set to 'True'")
            date2unix()
        
        else:
            pass

        num_transformer_name = 'num'
        cat_transformer_name = 'cat'

        #minmax for quant feats & one-hot encoding for categorical feats:
        preprocessor = ColumnTransformer(
                            transformers=[
                    ('num',MinMaxScaler(self.min_max),make_column_selector(dtype_include='number')),
                    ('cat',OneHotEncoder(),make_column_selector(dtype_include='object'))
                ], remainder='passthrough'
            )
        
    #preprocessor object credit: ChatGPT - 'Preprocessing data', and scikit-learn docs

        unlabeled_preprocessed_df = preprocessor.fit_transform(feat_df)
        print(f'Unlabeled preprocessed df: {unlabeled_preprocessed_df.shape}')

        #Return column names:
        num_cols = preprocessor.named_transformers_[num_transformer_name].get_feature_names_out()
        OH_col_array = preprocessor.named_transformers_[cat_transformer_name].categories_
        
        OH_cols = []

        for in_feature in OH_col_array:
            OH_cols.extend(in_feature)

        preprocessed_feat_cols = list(num_cols) + list(OH_cols)
        processed_feat_df = pd.DataFrame(unlabeled_preprocessed_df,columns=preprocessed_feat_cols)

        print(processed_feat_df.head(3))

        return processed_feat_df

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, i):
        '''
        #get sequence before 
        if i >= self.seq_length-1:
            seq_start = i - self.seq_length + 1
            x = self.x[seq_start:(i+1),:]
        else:
        padding = self.x[0].repeat(self.seq_length-i-1,1)
        x=self.x[0:(i+1),:]
        x=torch.cat((padding,x),0)
        '''

        #when i < seq_length, the values before i (inclusive) are returned, 
        #the first values are padded with data[0], and the rest is returned in the seq
        #when i > seq_length, the values after i (inclusive) are returned
        if i >= self.seq_length-1:
            x = self.x[i:i+self.seq_length,:]
            y = self.y[i:i+self.seq_length]
        else:
            x_padding = self.x[0].repeat(self.seq_length-i-1,1)
            y_padding = self.y[0].repeat(self.seq_length-i-1)
            x=self.x[0:i+1,:]
            x=torch.cat((x_padding,x),0)
            y=self.y[0:i+1,]
            y=torch.cat((y_padding,y),0)
            x=x.float()
            y=y.float()

        return x, y
    
    #credit except preprocessing: https://www.crosstab.io/articles/time-series-pytorch-lstm/#create-datasets-that-pytorch-dataloader-can-work-with

dataset = SeqDataset(w_df,target_col=target_col,feat_cols=feat_cols,min_max=min_max,datetimes=True,seq_length=seq_len)
print(dataset.__getitem__(10))
