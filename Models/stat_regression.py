import pandas as pd
import pickle as p
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
from sklearn.svm import SVR
import scipy
from scipy import stats
import argparse

#specify whether to use the small, medium or large feature set.
parser = argparse.ArgumentParser(description='Run regression analysis on edit types.')
parser.add_argument('--feature_set', type=str, choices=['small', 'medium', 'large'], default='small',
					help='Choose the feature set to use: small, medium, or large.')
args = parser.parse_args()


#Load Pandas Dataframe with all the data.
pandas_df_loc = "../Dataset/combined_dataframe.pkl"
dfFull = pd.read_pickle(pandas_df_loc)

#List of the 4, 25 and 55 Edit-Type Feature Set.

smallFeat = ['R', 'M', 'U', 'NONE']
medFeat = ['MORPH', 'PUNCT', 'ADJ', 'ADJ:FORM', 'ADV', 'NOUN:NUM', 'CONTR', 'PART', 'SPELL', 'NOUN', 'VERB', 'VERB:INFL', 'WO', 'NOUN:POSS', 'OTHER', 'ORTH', 'PREP', 'VERB:FORM', 'DET', 'NOUN:INFL', 'VERB:TENSE', 'VERB:SVA', 'PRON', 'CONJ', 'NONE']
bigFeat = ['U:VERB:FORM', 'U:NOUN', 'R:NOUN:POSS', 'U:VERB', 'R:NOUN:INFL', 'NONE', 'U:CONJ', 'U:NOUN:POSS', 'R:PART', 'M:PREP', 'R:CONJ', 'M:OTHER', 'U:VERB:TENSE', 'M:PART', 'R:VERB', 'R:OTHER', 'R:WO', 'U:CONTR', 'R:PREP', 'M:DET', 'M:NOUN:POSS', 'U:PREP', 'R:PRON', 'U:PRON', 'U:ADJ', 'R:ADV', 'U:PART', 'R:NOUN:NUM', 'M:PRON', 'M:ADV', 'R:SPELL', 'R:ORTH', 'R:CONTR', 'R:VERB:INFL', 'R:MORPH', 'R:VERB:TENSE', 'U:PUNCT', 'M:VERB', 'U:ADV', 'M:CONTR', 'R:PUNCT', 'R:ADJ', 'R:DET', 'M:VERB:TENSE', 'M:PUNCT', 'U:OTHER', 'R:ADJ:FORM', 'M:CONJ', 'U:DET', 'M:ADJ', 'R:VERB:SVA', 'M:VERB:FORM', 'R:VERB:FORM', 'R:NOUN', 'M:NOUN']

if args.feature_set == 'small':
	feature_set = smallFeat
elif args.feature_set == 'medium':
	feature_set = medFeat
elif args.feature_set == 'large':
	feature_set = bigFeat

#Filter the Dataset to use < 250 seconds threshold.
#Duplicate records are averaged and then combined.

dfDS = dfFull[dfFull['use_MO_TRG'] == 1]

#change here for small/med/big features for the model.
#Remove extra features to calculate coefficients for just edit types.
checkColumns = feature_set + ['NumWordsS', 'NumWordsT', 'NumEditedWords']

Xval = dfDS[checkColumns]
Yval = dfDS['time_avg_MO_TRG']

X_train, X_test, y_train, y_test = train_test_split(Xval, Yval, test_size=0.2, random_state=42, shuffle=True)

#Linear - Ridge Regression (L2)
lmR = Ridge(alpha=1.0)
svR = SVR(kernel='linear')

lmR.fit(X_train,y_train)
svR.fit(X_train, y_train)


#Calucalte the statistics of the models:
predictionslmR = lmR.predict(X_test)
predictionssvR = svR.predict(X_test)

#Print the results of the models:
print("Linear Regression Model Results :\n")
print('MAE-test:', metrics.mean_absolute_error(y_test, predictionslmR))
#Correlation between the predicted time and the actual time:
print("\tSpearman : "+str(stats.spearmanr(y_test,predictionslmR)))
print("\tPearson : "+str(stats.pearsonr(y_test,predictionslmR)))
print("\tkendalltau : "+str(stats.kendalltau(y_test,predictionslmR)))

print("\nSVR Linear Model Results :\n")
print('MAE-test:', metrics.mean_absolute_error(y_test, predictionssvR))
#Correlation between the predicted time and the actual time:
print("\tSpearman : "+str(stats.spearmanr(y_test,predictionssvR)))
print("\tPearson : "+str(stats.pearsonr(y_test,predictionssvR)))
print("\tkendalltau : "+str(stats.kendalltau(y_test,predictionssvR)))


#Save the model for future use.
filenameLR = 'model-LR.sav'
p.dump(lmR, open(filename, 'wb'))
filenameSVR = 'model-SVR.sav'
p.dump(svR, open(filenameSVR, 'wb'))

#Check the coefficients of the linear regression model.

print(lmR.intercept_)
coeff_df = pd.DataFrame(lmR.coef_.squeeze(),Xval.columns, columns=['Coefficient'])
coeff_df.sort_values(by='Coefficient', key=abs, ascending=False)

print(coeff_df)