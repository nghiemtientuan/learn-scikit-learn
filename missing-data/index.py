import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('dt.csv', header=None)
print(data)
# transform from dataframe to array
x = data.values

# mean - trung binh
# most_frequent - lay gia tri xuat hien nhieu nhat
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(x)
result = imp.transform(x)
print('---')
print(result)
