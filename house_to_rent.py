#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd


# In[53]:


data=pd.read_csv('houses_to_rent.csv')
data.head()


# In[54]:


data.shape


# In[55]:


data.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[56]:


data.head()


# In[57]:


data['floor'].replace(to_replace='-',value=0, inplace=True)


# In[58]:


data['animal'].replace(to_replace='not acept',value=0,inplace=True)
data['animal'].replace(to_replace='acept',value=1,inplace=True)


# In[59]:


data.head()


# In[60]:


data['furniture'].replace(to_replace='furnished',value=1,inplace=True)
data['furniture'].replace(to_replace='not furnished',value=0,inplace=True)


# In[61]:


data.head()


# In[62]:


data.columns


# In[63]:


for col in['hoa', 'rent amount', 'property tax','fire insurance', 'total']:
    data[col].replace(to_replace='R\$', value='',regex=True,inplace=True)
    data[col].replace(to_replace=',', value='',regex=True,inplace=True)


# In[64]:


data['hoa'].replace(to_replace='Sem info',value=0, inplace=True)


# In[68]:


data['hoa'].replace(to_replace='Incluso',value=0, inplace=True)
data['property tax'].replace(to_replace='Incluso',value=0, inplace=True)


# In[69]:


data.head()


# In[70]:


data.isin(['Incluso']).any()


# In[71]:


data=data.astype(dtype=np.int64)


# In[72]:


data.info()


# In[73]:


data=data.sample(frac=1).reset_index(drop=True)


# In[74]:


y=data['city']
X=data.drop('city',axis=1)


# In[75]:


y


# In[77]:


from sklearn.model_selection import train_test_split


# In[78]:


from sklearn.preprocessing import MinMaxScaler


# In[80]:


scaler=MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)


# In[83]:


pd.DataFrame(X)


# In[84]:


X_test,X_train,y_test,y_train=train_test_split(X,y,train_size=0.80)


# In[89]:


X_test.shape


# In[90]:


X_train.shape


# In[92]:


y_train.shape


# In[94]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# In[98]:


log_model=LogisticRegression(penalty='l2',verbose=1)
svm_model=SVC(kernel='rbf',verbose=1)
nn_model=MLPClassifier(hidden_layer_sizes=(16,16),activation='relu',solver='adam',verbose=1)


# In[99]:


log_model.fit(X_train,y_train)
svm_model.fit(X_train,y_train)
nn_model.fit(X_train,y_train)


# In[101]:


print(log_model.score(X_test,y_test))
print(svm_model.score(X_test,y_test))
print(nn_model.score(X_test,y_test))


# In[103]:


data[data.columns[0]].sum()/data.shape[0]


# In[104]:


from sklearn.metrics import f1_score


# In[110]:


log_pred=log_model.predict(X_test)
svm_pred=svm_model.predict(X_test)
nn_pred=nn_model.predict(X_test)


# In[111]:


print(log_pred)


# In[112]:


print(f1_score(log_pred,y_test))
print(f1_score(svm_pred,y_test))
print(f1_score(nn_pred,y_test))


# In[ ]:





# In[ ]:




