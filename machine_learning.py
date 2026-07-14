import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor,plot_tree
import os 

OUTPUT="ML plots"
os.makedirs(OUTPUT,exist_ok=True)

url="datasets/ml_df.csv"

# LOAD DATA
#---------------------------------------------
def load_ml_data(url):
    df=pd.read_csv(url)
    return df

# INSPECT & EDIT DATA
#---------------------------------------------
def inspect_and_edit_dataset(df):
    print(df.columns)

    df.drop(columns=['Unnamed: 0'],inplace=True)

    df=df[df["sqft"]<5000]
    df=df[df["type_single_family"]==1]
    df=df[(df["year_built"]==2023)|(df["year_built"]==2024)|(df["year_built"]==2026)]
    df=df[(df["garage"]==2)| (df["garage"]==3)]

    return df

# PRE-MODELING PREPARATION
#---------------------------------------------
def pre_modeling_preparation(df):

    x=df.drop(columns=["listPrice",'type_condo_townhome_rowhome_coop', 'type_condos', 'type_multi_family','type_single_family', 'type_townhomes'])
    y=df['listPrice']

    scaler=StandardScaler()
    x=scaler.fit_transform(x)
    
    return x,y

# RANDOM FOREST REGRESSOR
#---------------------------------------------
def random_forest_ml(x_train,x_test,y_train,y_test):

    regr=RandomForestRegressor(n_estimators=200,
                            max_depth=10,
                            min_samples_split=5,
                            random_state=42)
    regr.fit(x_train,y_train)

    y_predi=regr.predict(x_test)

    print(r2_score(y_predi,y_test))  # 0.7173216497895358

    plt.scatter(y_predi,y_test,s=1)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{OUTPUT}/random_forest_plot.png")
    plt.show()

# MULTIPLE REGRESSION
#---------------------------------------------
def multiple_regr_ml(x_train,x_test,y_train,y_test):
    regr=LinearRegression()
    regr.fit(x_train,y_train)

    y_predi=regr.predict(x_test)

    print(r2_score(y_predi,y_test))  # 0.6560235305503466

    plt.scatter(y_predi,y_test,s=1)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# DECISION TREE REGRESSOR
#---------------------------------------------
def decision_tree_ml(x_train,x_test,y_train,y_test):
    dtree=DecisionTreeRegressor()
    dtree=dtree.fit(x_train,y_train)

    y_predi=dtree.predict(x_test)

    features=['sqft', 'stories', 'beds', 'baths',
       'baths_full', 'baths_full_calc', 'garage', 'year_built', 'house_age',
       'baths_per_bedroom']

    plt.figure(figsize=(15,9))
    plot_tree(dtree,feature_names=features)
    plt.savefig(f"{OUTPUT}/decision_tree_plot.png")

    mae = mean_absolute_error(y_test, y_predi)
    r2 = r2_score(y_test, y_predi)

    print(f"MAE: {mae}")
    print(f"R² Score: {r2}")
    """
    MAE: 94164.81914165351
    R² Score: 0.16241670740575254
    """

# MAIN PIPELINE
#---------------------------------------------
def main():
    df= load_ml_data(url)
    df=inspect_and_edit_dataset(df)
    x,y=pre_modeling_preparation(df)

    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.1)

    random_forest_ml(x_train,x_test,y_train,y_test)
    multiple_regr_ml(x_train,x_test,y_train,y_test)
    decision_tree_ml(x_train,x_test,y_train,y_test)

# RUN
#---------------------------------------------
if __name__=="__main__":
    main()