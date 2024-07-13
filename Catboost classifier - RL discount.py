#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install catboost')


# In[2]:


import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[9]:


df = pd.read_csv('with_agent_action.csv',index_col=0)
print(df.shape)
df.columns


# In[4]:


df[['approved_payment','agent_discount']].value_counts(normalize=True)


# In[10]:


from sklearn.preprocessing import LabelEncoder

# Assuming X is your DataFrame and 'agent_discount' is a column in it

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'agent_discount' column
df['agent_discount_encoded'] = label_encoder.fit_transform(df['agent_discount'])

df[['agent_discount', 'agent_discount_encoded']].value_counts()


# In[11]:


# Split the dataset into features (X) and labels (y)
X = df[[#'user_id',
        'age',
        'current_browsing_time',
        'pages_visited',
        #'discount',
        #'suggested_amount_after_discount', 
        'onboarding_time',
        'referral_source_Facebook',
        'referral_source_Google',
        'referral_source_Instagram',
        'referral_source_Telegram',
        'referral_source_Tiktok',
        'referral_source_URL',
        'payment_method_Credit Card',
        'payment_method_Cryptocurrency',
        'payment_method_PayPal', 
        'time_of_day_Afternoon', 
        'time_of_day_Evening',
        'time_of_day_Morning', 
        'time_of_day_Night',
        'device_Android',
        'device_MacOS', 
        'device_Others', 
        'device_Windows',
#         'device_iOS',
        #'agent_action', 
#         'agent_discount',
        'agent_discount_encoded'
        #'approved_payment'
]]
y = df['approved_payment']


# In[59]:


df['approved_payment'].fillna(value=1,inplace=True)


# In[12]:


# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print(X_train.columns)


# In[13]:


from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# Initialize the CatBoost classifier
catboost_model = CatBoostClassifier(iterations=5000, 
                                    learning_rate=0.05, 
                                    depth=16, 
                                    loss_function='Logloss',
                                    l2_leaf_reg=1.0,
                                    border_count=254,
                                    bagging_temperature=2,
                                    random_strength=1.0,
                                    verbose=100)

# Train the model
catboost_model.fit(X_train, y_train)

# Predict on the test set
y_pred = catboost_model.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))


# In[14]:


# Get feature importance
feature_importances = catboost_model.get_feature_importance()
feature_names = X.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
bars = plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
# Annotate bars with the numerical values
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}',  # Formatting to 2 decimal places
             va='center', ha='left')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


# In[ ]:




