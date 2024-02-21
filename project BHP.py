#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('Bengaluru.csv')
df


# In[3]:


g = df.groupby('area_type')
g.count()


# In[4]:


df.isna().sum()


# In[5]:


df2 = df.drop(['area_type','availability','balcony','society'],axis='columns')
df2


# In[6]:


df2.isna().sum()


# In[7]:


df3 = df2.dropna()
df3


# In[8]:


df3 = df3.replace({'size':'[A-Za-z]'},'',regex=True)
df3


# In[9]:


df3['size'] = df3['size'].astype(int)


# In[10]:


df3


# In[11]:


df3[df3['size']>20] 


# In[12]:


df3['total_sqft'].unique()


# In[13]:


def isFloat(x):
    try:
        float(x)
        return False
    except:
        return True
    


# In[14]:


df3[df3['total_sqft'].apply(isFloat)].head(10)


# In[15]:


def convertToNum(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[16]:


df3['total_sqft']=df3['total_sqft'].apply(convertToNum)
df3


# In[17]:


df3.loc[410,'total_sqft']


# In[18]:


df3.loc[30,'total_sqft']


# In[19]:


loca = df3.groupby('location')['location'].agg('count')


# In[20]:


loca.sort_values(ascending=False).head(50)


# In[21]:


len(df3['location'].unique())


# In[22]:


loca[loca<=10]


# In[23]:


loca<=10


# In[24]:


len(loca[loca<=10])


# In[25]:


df3.location = df3.location.apply(lambda x: 'other' if x in loca[loca<=10] else x)
len(df3.location.unique())


# In[26]:


df3


# In[27]:


len(df3.location.unique())


# In[28]:


df3['pricePerSqft'] = df3['price']*100000/df3['total_sqft']


# In[29]:


df3


# In[30]:


df4 = df3[~(df3['total_sqft']/df3['size']<300)]


# In[31]:


df4.shape


# In[32]:


df4.pricePerSqft.describe()


# In[33]:


def remove_pps_outlier(df):
    dfout = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = subdf.pricePerSqft.mean()
        st = subdf.pricePerSqft.std()
        reduceddf = subdf[(subdf.pricePerSqft>(m-st)) & (subdf.pricePerSqft<=(m+st))]
        dfout = pd.concat([dfout,reduceddf],ignore_index=True)
        
    return dfout
df5 = remove_pps_outlier(df4)
df5.shape


# In[34]:


df5


# In[35]:


import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df['size']==2)]
    bhk3 = df[(df.location==location) & (df['size']==3)]
    matplotlib.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2bhk')
    plt.scatter(bhk3.total_sqft,bhk3.price,color='green',label='3bhk',marker='+')
    plt.xlabel('Total_sqft')
    plt.ylabel('price')
    plt.title(location)
    plt.legend()
    plt.show()

plot_scatter_chart(df5,"Hebbal")    
    


# In[37]:


import numpy as np


# In[38]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location,location_df in df.groupby('location'):
        bhk_stats={}
        for bhk,bhk_df in location_df.groupby('size'):
            bhk_stats[bhk]={
                'mean': bhk_df.pricePerSqft.mean(),
                'std': bhk_df.pricePerSqft.std(),
                'count': bhk_df.shape[0]
            }
        for bhk,bhk_df in location_df.groupby('size'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.pricePerSqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df6 = remove_bhk_outliers(df5)


# In[39]:


df6.shape


# In[40]:


plot_scatter_chart(df6,"Hebbal") 


# In[41]:


df6.bath.unique()


# In[42]:


df6[df6.bath>10]


# In[43]:


df7 = df6[df6.bath<(df6['size']+2)]


# In[44]:


df7.shape


# In[45]:


df8 = df7.drop('pricePerSqft',axis='columns')
df8.shape


# In[46]:


dummies = pd.get_dummies(df8.location)
dummies


# In[47]:


df9 = pd.concat([df8,dummies],axis='columns')


# In[48]:


df9 = df9.drop(['location','other'],axis='columns')


# In[49]:


df9


# In[50]:


x = df9.drop('price',axis='columns')
y = df9.price


# In[51]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[52]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[53]:


model.score(x_test,y_test)


# In[54]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
cross_val_score(LinearRegression(),x,y,cv=cv).mean()


# In[55]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
def find_best_model(x,y):
    algos={
        'LinearRegression':{
            'model':LinearRegression(),
            'params':{}
        },
        'lasso':{
            'model':Lasso(),
            'params':{
                'alpha':[1,2],
                'selection':['random','cyclic']
            }
        },
        'DecisionTreeRegressor':{
            'model':DecisionTreeRegressor(),
            'params':{
                'criterion':['mse','friedman_mse'],
                'splitter':['best','random']
            }
        }
    }
    scores=[]
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name,config in algos.items():
        model = GridSearchCV(config['model'],config.get('params'),cv=cv,return_train_score=False)
        model.fit(x,y)
        scores.append({'model':algo_name, 'best_score':model.best_score_,'best_params':model.best_params_})
    return pd.DataFrame(scores)    
find_best_model(x,y)


# In[56]:


def predict_price(location,sqft,bath,size):
    loc_index=np.where(x.columns==location)[0][0]
    X=np.zeros(len(x.columns))
    X[0]=size
    X[1]=sqft
    X[2]=bath
    if loc_index>=0:
        X[loc_index]=1
    return  model.predict([X])[0]


# In[57]:


predict_price('1st Phase JP Nagar',1000,3,3)


# In[58]:


predict_price('Indira Nagar',1000,3,3)


# In[59]:


from joblib import dump,load
dump(model,'banglore_home_prices_model.joblib')


# In[60]:


import json
columns = {'data_columns':[col.lower() for col in x.columns]}
with open('columns.json','w') as f:
    json.dump(columns,f)


# In[ ]:




