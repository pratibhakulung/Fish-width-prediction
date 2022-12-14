#!/usr/bin/env python
# coding: utf-8

# ## Predicting Width of different fish

# In[59]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ### loading data

# In[60]:


df=pd.read_csv('Desktop\MachineLearning\LineaRegression\Fish.csv')
df.reset_index()


# In[61]:


new_dataframe=pd.get_dummies(df,columns=['Species'],drop_first=True)


# In[62]:


new_dataframe


# In[63]:


X=new_dataframe.drop(['Width'],axis=1)
y=new_dataframe['Width']


# ### splitting data into train and test

# In[64]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)


# ### prediction model

# In[65]:


predictor=LinearRegression(n_jobs=-1)


# In[66]:


model=predictor.fit(X_train,y_train)


# In[67]:


y_prediction=model.predict(X_test)
y_prediction


# In[68]:


y_train


# ### performance

# In[69]:


mean_squared_error(y_prediction,y_test)


# In[ ]:




