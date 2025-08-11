import pandas as pd
import pickle as p
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics
from sklearn.svm import SVR
import scipy
from scipy import stats
import argparse

#Parse the arguments:
parser = argparse.ArgumentParser(description='Calculate Regression Coefficients for Edit Types')
#Linear Regression model passed to evaluate.
parser.add_argument('--model', type=str, default='linear', choices=['linear', 'ridge'], help='Model type to use for regression')
args = parser.parse_args()

#Load the pickle model passed as input.
lmR = p.load(open(args.model, 'rb'))

print("Loaded Model: ", args.model)
print("Intercept Value :")
print(lmR.intercept_)

coeff_df = pd.DataFrame(lmR.coef_.squeeze(),lmR.feature_names_in_, columns=['Coefficient'])
coeff_df = coeff_df.sort_values(by='Coefficient', key=abs, ascending=False)
print(coeff_df)
