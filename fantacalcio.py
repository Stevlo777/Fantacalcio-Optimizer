
#!C:/Users/Stevan/Documents/fanta 21/VirtualEnvironments/bin/python
# coding: utf-8

# In[78]:


#cross validation to optimize fit
#normalize
#explore different regressions with potential exponential term
#k-means clustering to split players into different budgets
#optimizer to find best way to spread budget

#add playerm minutes, xG?


# #Imports
# 

# In[121]:


import csv
import pandas as pd
import numpy as np
from patsy import dmatrices
from pulp import *


from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import cross_val_score

import seaborn as sns
from sklearn.cluster import KMeans

import statsmodels.api as sm


# In[122]:


calcio = pd.read_csv("calcio.csv")

calcio = calcio.reset_index()

calcio = calcio.drop(range(147,535))
calcio = calcio.drop(columns = ["PREDICTED VALUE"])



scaler = MinMaxScaler(feature_range=(0,1))


scaled1 = scaler.fit_transform((np.array(calcio["BID"])).reshape(-1,1))
calcio["scaled_bid"] = scaled1.reshape(-1,1)


calcio = calcio.sample(frac=1)


calcio_D = calcio.loc[calcio["R"]=="D"]
calcio_C = calcio.loc[calcio["R"]=="C"]
calcio_A = calcio.loc[calcio["R"]=="A"]
calcio_P = calcio.loc[calcio["R"]=="P"]





# In[123]:


def fitting(data):
    global X_train
    global X_test
    global y_train
    global y_test
    
    global X
    global y
    global z
    
    X = np.array(data["FV 19/20"])
    y = np.array(data["scaled_bid"])
    
    z = np.array(data["2 YR AVG"])



    X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1,1), y.reshape(-1,1), random_state=0)



# In[124]:


#def score_estimator(estimator, X_test):

#  y_pred = estimator.predict(X_test)

#  print("MSE: %.3f" % mean_squared_error(y_pred, y_test))
#  print("MAE: %.3f" % mean_absolute_error(y_pred, y_test))

#  scores = -1 * cross_val_score(estimator, y_pred, y_test, cv=5, scoring='neg_mean_absolute_error')
#  print("CV MAE: %.3f" % scores.mean())
  


# In[125]:




linear = Pipeline(steps=[("regressor", LinearRegression())])



players = [calcio_A, calcio_C, calcio_D, calcio_P]
for i in players:

    fitting(i)

    linear.fit(X_train, y_train)
    
    #Xnew = np.array(i["FV 19/20"])
    i["y_hat"] = linear.predict(X.reshape(-1,1))
 
    i["z_hat"] = linear.predict(z.reshape(-1,1))



    i["20/21 Value"] = scaler.inverse_transform(np.array(i["y_hat"]).reshape(-1,1))
    i["21Value"] = scaler.inverse_transform(np.array(i["z_hat"]).reshape(-1,1))

#adding price compensation for extra 20% of people in league
    i["21/22 Value"] = 1.1*i["21Value"]
    
    i["residual"] = i["20/21 Value"] - i["BID"]

    #print("str(i) "+ "linear evaluation:")
    #score_estimator(linear, X_test)






players = [calcio_A, calcio_C, calcio_D] 

for i in players:

    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(np.array(i["21Value"]).reshape(-1,1))
    
    
    i["cluster"] = y_kmeans.tolist()
    
    i["Tier"] = "Mid"
    
    
    
#Adding compensation for defense modifier

calcio["2 YR AVG"].loc[calcio["R"] == "D"] += 0.16
    



# In[133]:


calcio = pd.DataFrame()

for i in [calcio_A, calcio_C, calcio_D]:

    Tier0 = i.loc[i["cluster"] == 0]
    Tier1 = i.loc[i["cluster"] == 1] 
    Tier2 = i.loc[i["cluster"] == 2]

    mean0 = (Tier0["21Value"].mean())
    mean1 = (Tier1["21Value"].mean())
    mean2 = (Tier2["21Value"].mean())
    
    #TierHigh
    if mean0 > mean1 and mean0 > mean2:
        
        Tier0["Tier"] = "Hig"
        
    elif mean1 > mean0 and mean1 > mean2:
        
        Tier1["Tier"] = "Hig"
    
    else:
        Tier2["Tier"] = "Hig"
        
    #TierLow
    if mean0 < mean1 and mean0 < mean2:
        
        Tier0["Tier"] = "Low"
    
    elif mean1 < mean0 and mean1 < mean2:
        
        Tier1["Tier"] = "Low"
        
    else:
        Tier2["Tier"] = "Low"
        
    
    #TierMid
    #for j, k in i.iterrows():
        
        #if k["Tier"] == "Na":
            #k["Tier"] = "Mid"
            
    i = Tier0.append([Tier1, Tier2])
    
    calcio = calcio.append(i)
     

    






# In[134]:


#dict has 25 players, 3 GK, 8 D, 8 M, 6 S
#2021/22 total value < 1000
#maximize avg points


# In[135]:


calcio = calcio.loc[calcio["21Value"] > 0]
calcio = calcio.reset_index(drop=True)


dict = calcio.set_index("Nome").T.to_dict("index")

keys_to_remove = ["Id", "Squadra", "Pg", "Mv", "Mf", "Gf", "Gs", "Rp", "Rc", "R+", "R-", "Ass", "Asf", "Amm", "Esp", "Unnamed: 18",
                 "IV","Au", "FV", "D", "BID", "ln(Bid)", "FV 19/20", "scaled_bid", "y_hat", "z_hat", "20/21 Value", "residual", "cluster"]

for i in keys_to_remove:
    
    dict.pop(i)

    calcio = calcio.drop(columns = i)
    
calcio = calcio.sort_values(by = ["Nome"])

calcio = calcio.dropna()

newindex = []
for i in range(calcio.shape[0]):
    newindex.append(i)
    


calcio = pd.merge(pd.DataFrame(newindex), calcio, how='left', left_index = True, right_index = True)

pd.set_option('display.max_rows', 1000)




# In[136]:


dummies = pd.get_dummies(calcio["R"])


calcio = pd.merge(calcio, dummies, how='right', left_index = True, right_index = True)


# In[137]:


#create problem to optimize



prob = pulp.LpProblem('PointsMaximizer', LpMaximize)

decision_variables = []

for rownum, row in calcio.iterrows():
    variable = str('x' + str(row["Nome"])+str(row["R"])+str(row["Tier"]))
    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat= 'Integer') #make variables binary
    decision_variables.append(variable)
    
#print ("Total number of decision_variables: " + str(len(decision_variables)))
#print ("Array with Decision Variables:" + str(decision_variables))


# In[138]:



total_points = ""



for i, schedule in enumerate(decision_variables):
    for rownum, row in calcio.iterrows():
        if rownum  == i:
                   
                      
            formula = row["2 YR AVG"]*schedule
                 
            total_points = lpSum([total_points, formula])
        
            
            
prob += total_points
#print ("Optimization function: " + str(total_points))


# In[139]:


budget = 800
totalbudget = ""


for rownum, row in calcio.iterrows():
    for i, schedule in enumerate(decision_variables):
        if rownum == i:
            formula = row["21Value"]*schedule
            totalbudget = lpSum([totalbudget, formula])
            

                
            
prob += (totalbudget <= budget)






            


# In[140]:


Aplayers = ""
for rownum, row in calcio.iterrows():
    for i, schedule in enumerate(decision_variables):
        if rownum == i:
            formula = row["A"]*schedule
            Aplayers = formula + Aplayers

Cplayers = ""
for rownum, row in calcio.iterrows():
    for i, schedule in enumerate(decision_variables):
        if rownum == i:
            formula = row["C"]*schedule
            Cplayers = formula + Cplayers
            
Dplayers = ""
for rownum, row in calcio.iterrows():
    for i, schedule in enumerate(decision_variables):
        if rownum == i:
            formula = row["D"]*schedule
            Dplayers = formula + Dplayers
            



prob += (Aplayers == 3)
prob += (Cplayers == 3)
prob += (Dplayers == 4)





            


# In[141]:


#print(prob)
prob.writeLP("PointsMaximizer.lp")


# In[142]:


optimization_result = prob.solve()


#print("Status:", LpStatus[prob.status])
#print("Optimal Solution to the problem: ", value(prob.objective))
#print("Individual decision_variables: ")


print(calcio)

optimization_result = prob.solve()


print("Status:", LpStatus[prob.status])
print("Optimal Solution to the problem: ", value(prob.objective))
print("Individual decision_variables: ")


players = []
players2 = []
Dbudget=0
Cbudget=0
Abudget=0

for i in calcio["Nome"]:

    i = i.replace(" ", "_")
    
print(calcio)

for v in prob.variables():
    if v.varValue == 1.0:
        #print(str(v.name)[-4:])
        players.append(str(v.name)[-4:])
        players2.append(str(v.name[1:-4]))
        print(v.name[1:-4])
        
        
        
        
for i in players2:
        
    print(i)
    if calcio.loc[calcio["Nome"] == i.replace("_", " ")].R.max() == "D":

        Dbudget += calcio["21Value"].loc[calcio["Nome"] == i.replace("_", " ")].max()

    elif calcio.loc[calcio["Nome"] == i.replace("_", " ")].R.max() == "C":

        Cbudget += calcio["21Value"].loc[calcio["Nome"] == i.replace("_", " ")].max()

    elif calcio.loc[calcio["Nome"] == i.replace("_", " ")].R.max() == "A":

        Abudget += calcio["21Value"].loc[calcio["Nome"] == i.replace("_", " ")].max()
    
    print(Dbudget)
    print(Cbudget)
    print(Abudget)

       
       
        
AHig = 0
AMid = 0
ALow = 0
CHig = 0
CMid = 0
CLow = 0
DHig = 0
DMid = 0
DLow = 0



for i in players:
    
    if i == "AHig":
        AHig += 1
    elif i == "AMid":
        AMid += 1
    elif i == "ALow":
        ALow += 1
        
    elif i == "CHig":
        CHig += 1
    elif i == "CMid":
        CMid += 1
    elif i == "CLow":
        CLow += 1

    elif i == "DHig":
        DHig += 1
    elif i == "DMid":
        DMid += 1
    elif i == "DLow":
        DLow += 1
        
print("Striker High: " + str(AHig))
print("Striker Mid: " + str(AMid))
print("Striker Low: " + str(ALow))

print("Midfielder High: " + str(CHig))
print("Midfieler Mid: " + str(CMid))
print("Midfielder Low " + str(CLow))

print("Defender High: " + str(DHig))
print("Defender Mid: " + str(DMid))
print("Defender Low " + str(DLow))

print(Dbudget)
print(Cbudget)
print(Abudget)



print(players)
        


# In[143]:


with open("budget.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    list = [str(AHig), str(AMid), str(ALow), str(CHig), str(CMid), str(CLow), str(DHig), str(DMid), str(DLow), str(Abudget), str(Cbudget), str(Dbudget)]
    writer.writerow(list)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




