from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import dataloader
import pandas as pd
import torch
import itertools

#import data to pandas dataframe 
sw_df = pd.read_csv('seattle-weather.csv')
sw_df['date'] = pd.to_datetime(sw_df['date'])

#convert to unix timestamps (GMT) - date becomes quant -> scaling
unix_epoch = pd.Timestamp('1970-01-01')

unix_timestamps = (sw_df['date']-unix_epoch)//pd.Timedelta(seconds=1)
sw_df['unix_timestamps'] = unix_timestamps
print(sw_df.head())
#print(sw_df.date.dtype)


# y column - wind
y_col = 'wind'
y_data = pd.DataFrame(sw_df[y_col])

# x quantitative columns - all except weather
x_q_cols = ['unix_timestamps','precipitation','temp_max','temp_min',]

x_q_data = sw_df[x_q_cols]
scaler = StandardScaler()
x_q_scaled_data = scaler.fit_transform(x_q_data)

# x categorical columns (weather) - one hot encoding
cat_pre_data = pd.DataFrame(sw_df['weather'])
enc = OneHotEncoder(handle_unknown='ignore')
OHE_transform = enc.fit_transform(X=cat_pre_data).toarray()
OHE_cols = enc.get_feature_names_out()
OHE_df = pd.DataFrame(OHE_transform,columns=OHE_cols,index=sw_df.index)

# Split data into train, val and test sets
#Train data - start to row 731 = 2012-01-01 to 2013-12-31
x_q_train = x_q_scaled_data[:731]
#x_c_train = OHE_df[:731]
y_train = y_data.iloc[:731]
#print(x_q_train)

#Validation data - row 731 to row 1096 = 2014-01-01 to 2014-12-31
x_q_val = x_q_scaled_data[731:1096]
#x_c_val = OHE_df[731:1096]

y_val = y_data.iloc[731:1096]
#print(x_q_val)

#Test data - row 1096 to end = 2015-01-01 to 2015-12-31
x_q_test = x_q_scaled_data[1096:]
x_c_test = OHE_df.iloc[1096:]
y_test = y_data.iloc[1096:]

# --- Tensor creation ---
#Train tensors

xq_train_tensor = torch.from_numpy(x_q_train)
#xc_train_tensor = torch.tensor(x_c_train)

y_train_tensor = torch.tensor(y_train.values)
#x_train_tensor = torch.concat([xq_train_tensor,xc_train_tensor])


#Validation tensors
xq_val_tensor = torch.from_numpy(x_q_val)
#xc_val_tensor = torch.tensor(x_c_val)

y_val_tensor = torch.tensor(y_val.values)
#x_val_tensor = torch.concat([xq_val_tensor,xc_val_tensor])

#Test tensors
xq_test_tensor = torch.from_numpy(x_q_test)
#xc_test_tensor = torch.tensor(x_c_test)

y_test_tensor = torch.tensor(y_test.values)

#x_test_tensor = torch.concat([xq_test_tensor,xc_test_tensor])

train_tensorset = [xq_train_tensor,y_train_tensor]
val_tensorset = [xq_val_tensor,y_val_tensor]
test_tensorset = [xq_test_tensor,y_test_tensor]

tensor_list = [train_tensorset,val_tensorset,test_tensorset]
flat_tensor_list = list(itertools.chain.from_iterable(tensor_list))
print(f"Num tensors:{len(flat_tensor_list)}")

for i in range(0,len(flat_tensor_list)):
    flat_tensor_list[i] = flat_tensor_list[i].float() 
    print(f"{i}: {flat_tensor_list[i].type()}")

# ***** new *****
# ----- Dataloader -----


