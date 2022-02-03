#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Remove Outliers for Machine Learning


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


# citirea datelor din 10_log_extended2.csv
my_data = pd.read_csv(r"D:\POLI\Master2\PC3\GIT\PC\10_log_extended2.csv")
my_data.shape


# In[4]:


# print a summary of the data
my_data.describe()


# In[5]:


my_data


# In[6]:


# get the number of missing data points per column
missing_values_count = my_data.isnull().sum()


# In[7]:


# look at the # of missing points
missing_values_count


# In[8]:


# how many total missing values do we have?
total_cells = np.product(my_data.shape)
total_missing = missing_values_count.sum()


# In[9]:


# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)


# In[10]:


# replace all NA's with 0
my_data = my_data.fillna(0)
my_data


# In[11]:


#USING QUANTILE TO FIND THE OUTLIERS
# importing the statistics module
import statistics
from statistics import mean
#calculate the mean value for NoIP
mean(my_data.NoIP)


# In[12]:


#NoIP
max_NoIP_access = my_data.NoIP.quantile(0.9)
my_data[my_data.NoIP > max_NoIP_access]
#there can be seen values bigger than double of the mean value for NoIP


# In[13]:


min_NoIP_access = my_data.NoIP.quantile(0.1)
my_data[my_data.NoIP < min_NoIP_access]
#the NoIP in this case it's okei


# In[14]:


#Viewed
#calculate the mean Value: how many accesses during the year did the students do?
mean(my_data.Viewed)


# In[15]:


max_Viewed_accesses = my_data.Viewed.quantile(0.9)
max_Viewed_Values = my_data[my_data.Viewed > max_Viewed_accesses]
max_Viewed_Values.Viewed


# In[16]:


min_Viewed_accesses = my_data.Viewed.quantile(0.10)
min_Viewed_Values = my_data[my_data.Viewed < min_Viewed_accesses]
min_Viewed_Values.Viewed


# In[17]:


mean(my_data['Class days'])


# In[18]:


max_ClassDays_accesses = my_data['Class days'].quantile(0.9)
my_data[my_data['Class days'] > max_ClassDays_accesses]
#we can observe there are users who acceses during the class 10 times more than the mean value


# In[19]:


min_ClassDays_accesses = my_data['Class days'].quantile(0.1)
my_data[my_data['Class days'] < min_ClassDays_accesses]
#we can observe there are users who acceses during the class 10 times more than the mean value


# In[20]:


#STANDARD DEVIATION METHOD
# calculate summary statistics
from numpy import mean
from numpy import std
my_data_mean = mean(my_data)


# In[21]:


my_data_std = std(my_data)
my_data_std


# In[22]:


# identify outliers
cut_off = my_data_std * 3
lower, upper = (my_data_mean - cut_off), (my_data_mean + cut_off)
lower


# In[23]:


upper


# In[24]:


my_data.dtypes


# In[25]:


type(upper)


# In[26]:


#lower.iloc[2]
#FILTER AFTER OUTLINERS FOR NoIP, VIEWED AND OTHERS
standard_deviation_method = my_data[
    ((my_data['NoIP'] < lower.iloc[2]) | (my_data['NoIP'] > upper.iloc[2])) | 
    #((my_data['Created'] < lower.iloc[3]) | (my_data['Created'] > upper.iloc[3])) |
    #((my_data['Deleted'] < lower.iloc[4]) | (my_data['Deleted'] > upper.iloc[4])) |
    #((my_data['Downloaded'] < lower.iloc[5]) | (my_data['Downloaded'] > upper.iloc[5])) | 
    #((my_data['Ended'] < lower.iloc[6]) | (my_data['Ended'] > upper.iloc[6])) |
    #((my_data['Reset'] < lower.iloc[7]) | (my_data['Reset'] > upper.iloc[7])) | 
    #((my_data['Shown'] < lower.iloc[8]) | (my_data['Shown'] > upper.iloc[8])) |
    #((my_data['Started'] < lower.iloc[9]) | (my_data['Started'] > upper.iloc[9])) |
    #((my_data['Submitted'] < lower.iloc[10]) | (my_data['Submitted'] > upper.iloc[10])) |
    #((my_data['Updated'] < lower.iloc[11]) | (my_data['Updated'] > upper.iloc[11])) |
    #((my_data['Uploaded'] < lower.iloc[12]) | (my_data['Uploaded'] > upper.iloc[12])) |
    ((my_data['Viewed'] < lower.iloc[13]) | (my_data['Viewed'] > upper.iloc[13])) |
    #((my_data['Holiday'] < lower.iloc[14]) | (my_data['Holiday'] > upper.iloc[14])) |
    #((my_data['Week 01'] < lower.iloc[15]) | (my_data['Week 01'] > upper.iloc[15])) |
    #((my_data['Week 02'] < lower.iloc[16]) | (my_data['Week 02'] > upper.iloc[16])) |
    #((my_data['Week 03'] < lower.iloc[17]) | (my_data['Week 03'] > upper.iloc[17])) |
    #((my_data['Week 04'] < lower.iloc[18]) | (my_data['Week 04'] > upper.iloc[18])) |
    #((my_data['Week 05'] < lower.iloc[19]) | (my_data['Week 05'] > upper.iloc[19])) |
    #((my_data['Week 06'] < lower.iloc[20]) | (my_data['Week 06'] > upper.iloc[20])) |
    #((my_data['Week 07'] < lower.iloc[21]) | (my_data['Week 07'] > upper.iloc[21])) |
    #((my_data['Week 08'] < lower.iloc[22]) | (my_data['Week 08'] > upper.iloc[22])) |
    #((my_data['Week 09'] < lower.iloc[23]) | (my_data['Week 09'] > upper.iloc[23])) |
    #((my_data['Week 10'] < lower.iloc[24]) | (my_data['Week 10'] > upper.iloc[24])) |
    #((my_data['Week 11'] < lower.iloc[25]) | (my_data['Week 11'] > upper.iloc[25])) |
    #((my_data['Week 12'] < lower.iloc[26]) | (my_data['Week 12'] > upper.iloc[26])) |
    #((my_data['Week 13'] < lower.iloc[27]) | (my_data['Week 13'] > upper.iloc[27])) |
    #((my_data['Week 14'] < lower.iloc[28]) | (my_data['Week 14'] > upper.iloc[28])) |
    #((my_data['Mean events'] < lower.iloc[29]) | (my_data['Mean events'] > upper.iloc[29])) |
    #((my_data['Max events'] < lower.iloc[30]) | (my_data['Max events'] > upper.iloc[30])) |
    #((my_data['Min events'] < lower.iloc[31]) | (my_data['Min events'] > upper.iloc[31])) |
    #((my_data['Total events'] < lower.iloc[32]) | (my_data['Total events'] > upper.iloc[32])) |
    #((my_data['1st half sem'] < lower.iloc[33]) | (my_data['1st half sem'] > upper.iloc[33])) |
    #((my_data['2nd half sem'] < lower.iloc[34]) | (my_data['2nd half sem'] > upper.iloc[34])) |
    #((my_data['Visit days'] < lower.iloc[35]) | (my_data['Visit days'] > upper.iloc[35])) |
    #((my_data['Day events min'] < lower.iloc[36]) | (my_data['Day events min'] > upper.iloc[36])) |
    #((my_data['Day events max'] < lower.iloc[37]) | (my_data['Day events max'] > upper.iloc[37])) |
    #((my_data['Weekend days'] < lower.iloc[38]) | (my_data['Weekend days'] > upper.iloc[38])) |
    ((my_data['Class days'] < lower.iloc[39]) | (my_data['Class days'] > upper.iloc[39])) #|
    #((my_data['Out days'] < lower.iloc[40]) | (my_data['Out days'] > upper.iloc[40])) |
    #((my_data['Afternoon'] < lower.iloc[41]) | (my_data['Afternoon'] > upper.iloc[41])) |
    #((my_data['Evening'] < lower.iloc[42]) | (my_data['Evening'] > upper.iloc[42])) |
    #((my_data['Morning'] < lower.iloc[43]) | (my_data['Morning'] > upper.iloc[43])) |
    #((my_data['Night'] < lower.iloc[44]) | (my_data['Night'] > upper.iloc[44])) |
    #((my_data['Class'] < lower.iloc[45]) | (my_data['Class'] > upper.iloc[45])) |
    #((my_data['Out'] < lower.iloc[46]) | (my_data['Out'] > upper.iloc[46]))
    
]


# In[30]:


standard_deviation_method.drop(['Year', 'Created', 'Shown', 'Started', 'Deleted', 'Downloaded', 'Ended', 'Reset', 'Submitted', 'Updated', 'Uploaded', 'Frequency', 'Recency', 'T'], axis = 1)
#del standard_deviation_method.Year


# In[53]:


#INTERQUARTILE RANGE METHOD
# calculate interquartile range
#VIEWED
q25_Viewed, q75_Viewed = np.percentile(my_data.Viewed, 25), np.percentile(my_data.Viewed, 75)
iqr_Viewed = q75_Viewed - q25_Viewed
iqr_Viewed


# In[55]:


# calculate the outlier cutoff
cut_off_Viewed = iqr_Viewed * 1.5
iqr_lower_Viewed, iqr_upper_Viewed = q25_Viewed - cut_off_Viewed, q75_Viewed + cut_off_Viewed
iqr_lower_Viewed, iqr_upper_Viewed


# In[56]:


#NoIP
q25_NoIP, q75_NoIP = np.percentile(my_data.NoIP, 25), np.percentile(my_data.NoIP, 75)
iqr_NoIP = q75_NoIP - q25_NoIP
iqr_NoIP


# In[57]:


# calculate the outlier cutoff
cut_off_NoIP = iqr_NoIP * 1.5
iqr_lower_NoIP, iqr_upper_NoIP = q25_NoIP - cut_off_NoIP, q75_NoIP + cut_off_NoIP
iqr_lower_NoIP, iqr_upper_NoIP


# In[59]:


#NoIP
q25_ClassDays, q75_ClassDays = np.percentile(my_data['Class days'], 25), np.percentile(my_data['Class days'], 75)
iqr_ClassDays = q75_ClassDays - q25_ClassDays
iqr_ClassDays


# In[60]:


# calculate the outlier cutoff
cut_off_ClassDays = iqr_ClassDays * 1.5
iqr_lower_ClassDays, iqr_upper_ClassDays = q25_ClassDays - cut_off_ClassDays, q75_ClassDays + cut_off_ClassDays
iqr_lower_ClassDays, iqr_ClassDays


# In[61]:


#FILTER AFTER OUTLINERS FOR NoIP, VIEWED AND OTHERS
interquartile_range_method = my_data[
    ((my_data['NoIP'] < iqr_lower_NoIP) | (my_data['NoIP'] > iqr_upper_NoIP)) | 
    ((my_data['Viewed'] < iqr_lower_Viewed) | (my_data['Viewed'] > iqr_upper_Viewed)) |
    ((my_data['Class days'] < iqr_lower_ClassDays) | (my_data['Class days'] > iqr_upper_ClassDays)) 
    
]


# In[62]:


interquartile_range_method.drop(['Year', 'Created', 'Shown', 'Started', 'Deleted', 'Downloaded', 'Ended', 'Reset', 'Submitted', 'Updated', 'Uploaded', 'Frequency', 'Recency', 'T'], axis = 1)


# In[ ]:





# In[ ]:




