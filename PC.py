#Remove Outliers for Machine Learning


# In[14]:

import numpy as np
import pandas as pd


# In[15]:

# citirea datelor din 10_log_extended2.csv
my_data = pd.read_csv(r"D:\POLI\Master2\PC3\GIT\PC\10_log_extended2.csv")
my_data.shape


# In[16]:


# print a summary of the data
my_data.describe()


# In[17]:


my_data


# In[12]:


# get the number of missing data points per column
missing_values_count = my_data.isnull().sum()


# In[13]:


# look at the # of missing points
missing_values_count


# In[18]:


# how many total missing values do we have?
total_cells = np.product(my_data.shape)
total_missing = missing_values_count.sum()


# In[19]:


# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)


# In[30]:


# replace all NA's with 0
my_data = my_data.fillna(0)
my_data


# In[58]:


#using quantile to find the outliers
# importing the statistics module
import statistics
#calculate the mean value for NoIP
mean(my_data.NoIP)


# In[59]:


#NoIP
max_NoIP_access = my_data.NoIP.quantile(0.9)
my_data[my_data.NoIP > max_NoIP_access]
#there can be seen values bigger than double of the mean value for NoIP


# In[60]:


min_NoIP_access = my_data.NoIP.quantile(0.1)
my_data[my_data.NoIP < min_NoIP_access]
#the NoIP in this case it's okei


# In[57]:


#Viewed
#calculate the mean Value: how many accesses during the year did the students do?
mean(my_data.Viewed)


# In[71]:


max_Viewed_accesses = my_data.Viewed.quantile(0.9)
max_Viewed_Values = my_data[my_data.Viewed > max_Viewed_accesses]
max_Viewed_Values.Viewed


# In[72]:


min_Viewed_accesses = my_data.Viewed.quantile(0.10)
min_Viewed_Values = my_data[my_data.Viewed < min_Viewed_accesses]
min_Viewed_Values.Viewed


# In[73]:


mean(my_data['Class days'])


# In[75]:


max_ClassDays_accesses = my_data['Class days'].quantile(0.9)
my_data[my_data['Class days'] > max_ClassDays_accesses]
#we can observe there are users who acceses during the class 10 times more than the mean value


# In[76]:


min_ClassDays_accesses = my_data['Class days'].quantile(0.1)
my_data[my_data['Class days'] < min_ClassDays_accesses]
#we can observe there are users who acceses during the class 10 times more than the mean value


# In[ ]:




