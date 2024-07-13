#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install catboost')


# In[1]:


import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('RL_dataset.csv',index_col=0)
print(df.shape)
df.columns


# In[3]:


df['approved_payment'] = np.where(df['user_action']==100,1,0)

# Encode categorical features using dummy encoding
categorical_features = [
                        #'ip_location',
                        'referral_source', 
                        #'kyc_type',
                        #'browser_type',
                        'payment_method',
                        'time_of_day',
                        #'planned_monthly_deposit',
                        'device',
                        #'device_model'
                       ]
df_encoded = pd.get_dummies(df, columns=categorical_features)

# Normalize numerical features
numerical_features = ['age',
                      'discount',
                      'current_browsing_time',
                      'pages_visited',
                      'onboarding_time',
                     # 'base_suggested_amount'
                     ]
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])

# Split the dataset into features (X) and labels (y)
X = df_encoded.drop(columns=['user_action'])
y = df_encoded['approved_payment']


# In[4]:


# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print(X_train.columns)


# In[5]:


relevant_cols = ['age',
                'current_browsing_time',
                'pages_visited', 
                'discount', 
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
                 'device_iOS']
X_train = X_train[relevant_cols]
X_test = X_test[relevant_cols]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[6]:


X_train['discount'].value_counts()


# In[7]:


# from sklearn.model_selection import GridSearchCV
# from catboost import CatBoostClassifier

# # Define the parameter grid
# param_grid = {
#     'iterations': [10000, 5000, 7000],
#     'learning_rate': [0.01, 0.03, 0.1],
#     'depth': [4, 6, 8, 10, 12],
#     'l2_leaf_reg': [1, 3, 5, 7],
#     'bagging_temperature': [0, 0.5, 1, 2],
#     'border_count': [32, 64, 128, 255],
#     'random_strength': [1, 2, 5, 10],
#     'scale_pos_weight': [1, 2, 3],  # Useful if dealing with imbalanced datasets
#     'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']  # Different tree growing policies
# }

# # Initialize the CatBoost classifier
# catboost_model = CatBoostClassifier(loss_function='Logloss', random_seed=42, verbose=0)

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, cv=3, scoring='accuracy')

# # Fit GridSearchCV
# grid_search.fit(X_train, y_train)

# # Best parameters
# print("Best Parameters: ", grid_search.best_params_)

# # Best estimator
# best_model = grid_search.best_estimator_

# # Predict on the test set
# y_pred = best_model.predict(X_test)

# # Evaluate the model's performance
# print(classification_report(y_test, y_pred))


# In[11]:


from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

# Initialize the CatBoost classifier
catboost_model = CatBoostClassifier(iterations=5000, 
                                    learning_rate=0.05, 
                                    depth=4, 
                                    loss_function='Logloss',
                                    l2_leaf_reg=3.0,
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


# In[12]:


# Get feature importance
feature_importances = catboost_model.get_feature_importance()
feature_names = relevant_cols

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





# In[ ]:




