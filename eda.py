import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/review_dataset.csv')
# print(df.head())
# print(df.columns)
# print(df.shape)
# print(df.info())
# print(df.describe())

features = df.columns.drop('Outcome Type')
target = 'Outcome Type'

# getting numerical and categorical features
numerical_feats = df[features].select_dtypes(include=np.number).columns
# print(numerical_feats)
cat_feats = df[features].select_dtypes(include='object').columns
# print(cat_feats)

# check for data imbalance
print(df[target].value_counts())

# df[target].value_counts().plot.bar()
# plt.show()

# check cat feature properties
# for c in cat_feats:
#     # print(df[c].value_counts())
#     print(c)
#     df[c].value_counts().plot.bar()
#     plt.show()

# for c in numerical_feats:
#     print(c)
#     df[c].value_counts().plot.hist(bins=5)
#     plt.show()

# for c in numerical_feats:
#     print(c)
#     print('min: {}, max: {}'.format(df[c].min(), df[c].max()))

# handling outliers
# Q1-1.5IQR and Q2+1.5IQR
# removing any values in the upper 1% is another way
# for c in numerical_feats:
#     print(c)
#
#     # Drop values below Q1 - 1.5 IQR and beyond Q3 + 1.5 IQR
#     # Q1 = df[c].quantile(0.25)
#     # Q3 = df[c].quantile(0.75)
#     # IQR = Q3 - Q1
#     # print(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
#
#     # dropIndexes = df[df[c] > Q3 + 1.5*IQR].index
#     # df.drop(dropIndexes , inplace=True)
#     # dropIndexes = df[df[c] < Q1 - 1.5*IQR].index
#     # df.drop(dropIndexes , inplace=True)
#
#     # Drop values beyond 90% of max()
#     dropIndexes = df[df[c] > df[c].max() * 9 / 10].index
#     df.drop(dropIndexes, inplace=True)
#
#
# for c in numerical_feats:
#     print(c)
#     df[c].plot.hist(bins=100)
    # plt.show()


# scatter plots and correlation
# fig, axes = plt.subplots(len(numerical_feats), len(numerical_feats), figsize=(16, 16), sharex=False, sharey=False)
# for i in range(0,len(numerical_feats)):
#     for j in range(0,len(numerical_feats)):
#         axes[i,j].scatter(x = df[numerical_feats[i]], y = df[numerical_feats[j]])
# fig.tight_layout()
# plt.show()
# the numerical features correlate well

# adding target values to scatter plot
# import seaborn as sns
# X1 = df[[numerical_feats[0], numerical_feats[1]]][df[target] == 0]
# X2 = df[[numerical_feats[0], numerical_feats[1]]][df[target] == 1]
#
# # for label 0
# plt.scatter(X1.iloc[:,0],
#             X1.iloc[:,1],
#             s=50,
#             c='blue',
#             marker='o',
#             label='0')
#
# # for label 1
# plt.scatter(X2.iloc[:,0],
#             X2.iloc[:,1],
#             s=50,
#             c='red',
#             marker='v',
#             label='1')
#
# plt.xlabel(numerical_feats[0])
# plt.ylabel(numerical_feats[1])
# plt.legend()
# plt.grid()
# plt.show()

# correlation of features
cols = [numerical_feats[0], numerical_feats[1]]
print(df[cols].corr())  # highly correlated

# fancy heatmap
# from string import ascii_letters
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.set(style="white")
#
# # Generate a large random dataset
# rs = np.random.RandomState(33)
# d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                  columns=list(ascii_letters[26:]))
#
# # Compute the correlation matrix
# corr = d.corr()
#
# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=np.bool))
#
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
#
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(220, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# plt.show()


# handling missing values
# print(df.isna().sum())
#
# # drop features/columns having more than 20% null values
# print(df.isna().sum()/len(df.index))
# threshold = 2/10
# cols_to_drop = df.loc[:, list((df.isna().sum()/len(df.index)) >= threshold)].columns
# print(cols_to_drop)
# df.drop(cols_to_drop, inplace=True, axis=1)
# print(df.head())
#
# print(df.isna().sum())
# print(df.shape)
#
# # drop rows with missing values
# df.dropna(inplace=True)
# print(df.isna().sum())
# print(df.shape)
# print(df.columns)

# imputing data: adding values for missing data
df_imputed = df.copy()
print(df.shape)
# fill missing values with mean
df_imputed[numerical_feats] = df_imputed[numerical_feats].fillna(df_imputed[numerical_feats].mean())
print(df_imputed[numerical_feats].isna().sum())
print("--------------------------------------------")

# imputing categorical data
# replace with most common cat: use mode
print(df_imputed[cat_feats].isna().sum())
for c in cat_feats:
    if c != "Pet ID":
        mode = df_imputed[c].mode().array[0]
        df_imputed[c].fillna(mode, inplace=True)


print(df_imputed.isna().sum())
# print(df_imputed.Name.head(100))

print("-----------------------------------------------------")
# using sklearn for imputation
from sklearn.impute import SimpleImputer

df_skl_impute = df.copy()
imputer = SimpleImputer(strategy="mean")
df_skl_impute[numerical_feats] = imputer.fit_transform(df_skl_impute[numerical_feats])
print(df_skl_impute[numerical_feats].isna().sum())

# Pick some categorical features you desire to impute with this approach
categoricals_missing_values = df_skl_impute[cat_feats].loc[:,list(((df_skl_impute[cat_feats].isna()
                                                                    .sum()/len(df_skl_impute.index)) > 0.0))].columns
columns_to_impute = categoricals_missing_values[1:3]
print(columns_to_impute)

imputer_cat = SimpleImputer(strategy="most_frequent")
df_skl_impute[columns_to_impute] = imputer_cat.fit_transform(df_skl_impute[columns_to_impute])
print(df_skl_impute[columns_to_impute].isna().sum())

print(df.Name.head(100))

# fill na with placeholder value
# imputer = SimpleImputer(strategy='constant', fill_value="Missing")










