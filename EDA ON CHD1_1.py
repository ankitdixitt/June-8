#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fcsparser
data=fcsparser.parse('CHD1_1.fcs')


# In[ ]:





# In[2]:


data


# In[ ]:


import pandas as pd
df=pd.DataFrame(data)


# In[3]:


get_ipython().run_line_magic('pylab', 'inline')


# In[4]:


data=fcsparser.parse('CHD1_1.fcs',meta_data_only=True)


# In[5]:


print(type(data))
print (data.keys())


# In[6]:


data = fcsparser.parse('CHD1_1.fcs', meta_data_only=True, reformat_meta=True)
data['_channels_']


# In[18]:


data, df = fcsparser.parse('CHD1_1.fcs', meta_data_only=False, reformat_meta=True)


# In[19]:


print(type(data))
print(type(df))


# In[20]:


df.head()


# In[21]:


df.info()


# In[22]:


data2=fcsparser.parse('CHD1_1.fcs',meta_data_only=True)


# In[23]:


data2 = fcsparser.parse('CHD1_1.fcs', meta_data_only=True, reformat_meta=True)
data2['_channels_']


# In[24]:


data2, df2 = fcsparser.parse('CHD1_1.fcs', meta_data_only=False, reformat_meta=True)


# In[25]:


df2.head()


# In[26]:


data3=fcsparser.parse('CHD1_1.fcs',meta_data_only=True)


# In[27]:


data3 = fcsparser.parse('CHD1_1.fcs', meta_data_only=True, reformat_meta=True)
data3['_channels_']


# In[28]:


data3, df3 = fcsparser.parse('CHD1_1.fcs', meta_data_only=False, reformat_meta=True)


# In[29]:


df3.head()


# In[30]:


import pandas as pd
import numpy as np


# In[31]:


#concatenating the dataframes
df_final=pd.concat([df,df2,df3],axis=0)


# In[32]:


df_final.head()


# In[33]:


df_final.info()


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


sns.set(style='whitegrid')
sns.scatterplot(x='FSC-A',y='SSC-A',data=df_final)


# In[36]:


df_final.columns


# In[43]:


sns.set(style='whitegrid')
sns.scatterplot(x='PE-A',y='PE-Cy5-A',data=df_final)
sns.scatterplot(x='PE-A',y='PE-Cy7-A',marker='*',s=200,data=df_final)


# In[44]:


sns.histplot(df_final['FITC-A'],kde='False')


# In[46]:


sns.histplot(df_final['PE-Cy5-A'],kde='False')


# In[47]:


sns.kdeplot(x='FSC-A',y='SSC-A',data=df_final)
plt.figure(figsize=(20,20))


# In[50]:


sns.kdeplot(x='PE-Cy5-A',y='PE-Cy7-A',data=df_final)


# In[ ]:





# In[ ]:


#Kmeans clustering


# In[48]:


from sklearn.cluster import KMeans


# In[51]:


km=KMeans()
sse=[]
k_range=range(1,20)
for k in k_range:
    km=KMeans(n_clusters=k)
    km.fit(df_final[['FSC-A','SSC-A','AmCyan-A']])
    sse.append(km.inertia_)


# In[52]:


plt.xlabel('K')
plt.ylabel('sum of squared errors')
plt.plot(k_range,sse)


# In[53]:


#putting K=5
km1=KMeans(n_clusters=5)
km1
y_pred=km1.fit_predict(df_final[['FSC-A','SSC-A','AmCyan-A']])
df_final['cluster']=y_pred
df_final.head()


# In[56]:


d1=df_final[df_final.cluster==0]
d2=df_final[df_final.cluster==1]
d3=df_final[df_final.cluster==2]
d4=df_final[df_final.cluster==3]
d5=df_final[df_final.cluster==4]
plt.scatter(d1['FSC-A'],d1['SSC-A'],d1['AmCyan-A'],color='red')
plt.scatter(d2['FSC-A'],d2['SSC-A'],d2['AmCyan-A'],color='purple')
plt.scatter(d3['FSC-A'],d3['SSC-A'],d3['AmCyan-A'],color='green')
plt.scatter(d4['FSC-A'],d4['SSC-A'],d4['AmCyan-A'],color='yellow')
plt.scatter(d5['FSC-A'],d5['SSC-A'],d5['AmCyan-A'],color='pink')
plt.figure(figsize=(100,100))


# In[57]:


#putting K=2
km2=KMeans(n_clusters=2)
y_pred=km2.fit_predict(df_final[['FSC-A','SSC-A','AmCyan-A']])
del df_final['cluster']
df_final.head()


# In[58]:


df_final['cluster']=y_pred
df_final.head()


# In[59]:


d1_1=df_final[df_final.cluster==0]
d2_1=df_final[df_final.cluster==1]
plt.scatter(d1_1['FSC-A'],d1_1['SSC-A'],d1_1['AmCyan-A'],color='red')
plt.scatter(d2_1['FSC-A'],d2_1['SSC-A'],d2_1['AmCyan-A'],color='black')


# In[60]:


df_final.head()


# In[61]:


#Putting K=3
km3=KMeans(n_clusters=3)
y_pred=km3.fit_predict(df_final[['FSC-A','SSC-A','AmCyan-A']])
del df_final['cluster']
df_final.head()


# In[62]:


df_final['cluster']=y_pred
df_final.head()


# In[63]:


dF1=df_final[df_final.cluster==0]
dF2=df_final[df_final.cluster==1]
dF3=df_final[df_final.cluster==2]
plt.scatter(dF1['FSC-A'],dF1['SSC-A'],dF1['AmCyan-A'],color='red')
plt.scatter(dF2['FSC-A'],dF2['SSC-A'],dF2['AmCyan-A'],color='black')
plt.scatter(dF3['FSC-A'],dF3['SSC-A'],dF3['AmCyan-A'],color='green')

