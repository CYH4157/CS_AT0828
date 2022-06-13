#!/usr/bin/env python
# coding: utf-8

# ## HW1: Linear Regression using Gradient Descent
# In hw1, you need to implement linear regression by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
# 
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.linear_model.LinearRegression

# In[238]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# ## Load data

# In[239]:


train_df = pd.read_csv("train_data.csv")
x_train, y_train = train_df['x_train'], train_df['y_train']
train_df.head()


# In[240]:


plt.plot(x_train, y_train, '.')


# In[241]:


def read_data(df):
    name = df.columns
    x, y = df[name[0]], df[name[1]]
    
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    return x, y 
    


# In[242]:


def my_mse(y_batch, y_pred):
    error = y_batch - y_pred
    mse =  (error ** 2).sum() / error.shape[0]
    return mse


# In[243]:


def gd_fit(x, y, learning_rate=1e-6, iteration=50, batch_size=None):
    x = np.hstack((np.ones(x.shape), x))
    n = x.shape[0]
    d = x.shape[1]
    beta = np.random.randn(d, 1)
    beta_log = [beta.copy()]
    mse = []

    for i in range(iteration):
        if batch_size is not None:
            idx = np.random.randint(0, n, batch_size)
            idx.sort()
        else:
            idx = np.arange(n)
        x_batch = x[idx]
        y_batch = y[idx]
        y_pred = x_batch @ beta
        error = y_batch - y_pred
        beta -= learning_rate * -2 * (x_batch.T @ error) / n
        beta_log.append(beta.copy())
        mse.append(my_mse(y_batch, y_pred))

    return beta, beta_log, mse


# In[ ]:


def predict(x_test, y_test, beta):
    x_test = np.hstack((np.ones(x_test.shape), x_test))
    y_pred = x_test @ beta
    error = my_mse(y_test, y_pred)    
    # print(f'weight: {beta[1][0]}\nintercept: {beta[0][0]}')
    
    return y_pred, error
  


# In[244]:


train_df = pd.read_csv("train_data.csv")

x, y = read_data(train_df)


parameters={'0':{'learning_rate':1e-1,'iteration':200,'batch_size':None},
            '1':{'learning_rate':1e-4,'iteration':200,'batch_size':None}}






for key in parameters:
    # print(key)
    learning_rate = parameters[key]['learning_rate']
    iteration = parameters[key]['iteration']
    batch_size = parameters[key]['batch_size']
    
    beta, beta_log, mse = gd_fit(x, y, learning_rate=learning_rate, iteration=iteration, batch_size=batch_size)
    
    
    


learning_rate=1e-1
iteration=200
batch_size= None
# x_train_ext = np.hstack((np.ones(x.shape), x))
beta, beta_log, mse = gd_fit(x, y, learning_rate=learning_rate, iteration=iteration, batch_size=batch_size)


# ## Test the performance on the testing data
# Inference the test data (x_test) by your model and calculate the MSE of (y_test, y_pred)

# In[245]:


test_data = pd.read_csv("test_data.csv")
x_test, y_test = test_data['x_test'], test_data['y_test']


# In[246]:


x_test = test_data['x_test'].values
y_test = test_data['y_test'].values

x_test = np.asarray(x_test).reshape(-1, 1)
y_test = np.asarray(y_test).reshape(-1, 1)


# In[247]:





# In[248]:


y_pred, error = predict(x_test, y_test, beta)


# In[249]:


upper_bound = np.ceil(np.max(x_test)) + 1
lower_bound = np.floor(np.min(x_test)) - 1

x_pred = np.linspace(lower_bound, upper_bound, len(x_test)).reshape(-1, 1)
y_pred, _ = predict(x_pred, y_test, beta)


# In[250]:


plt.figure()

for i in range(len(beta_log)):
    y_pred, _ = predict(x_pred, y_test, beta_log[i])
    alpha = np.sqrt(i/len(beta_log))
    plt.plot(x_pred.ravel(), y_pred.ravel(), c=(0.85, 0.7, 0.3, alpha))

    
plt.scatter(x_train, y_train, alpha=0.5)
plt.plot(x_pred.ravel(), y_pred.ravel(), c='red')
plt.show()

# plt.figure()
# plt.scatter(np.arange(len(mse)), mse, s=10)

# plt.tight_layout()
# plt.show()


# In[ ]:
''''''



