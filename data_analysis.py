import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import os

url="datasets/Texas_Residential_Real Estate_Intelligenc_ 2026.csv"
OUTPUT="data analysis plots"
os.makedirs(OUTPUT,exist_ok=True)

# LOAD DATA
#---------------------------------------------
def load_data(data):
    df=pd.read_csv(url)
    return df

# INSPECT DATA
#---------------------------------------------    
def inspect_data(df):
    print("\n--- HEAD ---")
    print(df.head())
    """
--- HEAD ---
            type                sub_type                                               text  ...  garage  year_built  Price_Per_SqFt
0  single_family                     NaN  Experience The Best Of North Texas Living In T...  ...     2.0      2021.0          244.61
1  single_family                     NaN  Welcome home to [Redacted Entity], [Redacted E...  ...     NaN      2025.0          134.59
2  single_family                     NaN  Welcome to this beautiful home located in [Red...  ...     2.0      2020.0          138.79
3  single_family                     NaN  Welcome to [Redacted Entity] - where style, co...  ...     2.0      2005.0          149.34
4  single_family  single family detached  ***READY NOW*** Discover this stunning new con...  ...     2.0      2025.0          148.55
5  single_family                     NaN  This inviting property highlights the best of ...  ...     2.0         NaN          142.04
6  single_family                     NaN  Welcome to [Redacted Entity] in the Heart of [...  ...     3.0         NaN          205.76
7  single_family                     NaN  The [Redacted Entity] floor plan embodies the ...  ...     2.0         NaN          151.18
8  single_family                     NaN  Beautifully designed single-story home located...  ...     2.0      2021.0          225.74
9  single_family                     NaN  Incredible views of [Redacted Entity] from thi...  ...     2.0      2010.0          365.37"""

    print("\n--- TAIL ---")
    print(df.tail())

    """
--- TAIL ---
                type sub_type                                               text  ...  garage  year_built  Price_Per_SqFt
12132         condos    condo  Welcome to [Redacted Entity] in the Heart of [...  ...     NaN      1968.0          148.49
12133  single_family      NaN  Welcome home to this charming farmhouse-style ...  ...     2.0      2021.0          190.51
12134  single_family      NaN  Welcome to [Redacted Entity], a beautifully ma...  ...     2.0      1984.0          118.14
12135  single_family      NaN  BEACHSIDE GEM HAS IT ALL! Lovely 3/2 in the he...  ...     1.0      2008.0          316.67
12136  single_family      NaN  This beautifully maintained home sits on 1.83 ...  ...     3.0      2007.0          200.07"""

    print("\n--- COLUMNS ---")
    print(df.columns) 
    # ['type', 'sub_type', 'text', 'listPrice', 'sqft', 'stories', 'beds','baths', 'baths_full', 'baths_full_calc', 'garage', 'year_built','Price_Per_SqFt']

    print("\n--- ISNULL ---")
    print(df.isnull().sum())
    """
--- ISNULL ---
type                   0
sub_type           10388
text                   0
listPrice              3
sqft                  64
stories              507
beds                 128
baths                274
baths_full           149
baths_full_calc      146
garage              1913
year_built          2288
Price_Per_SqFt         2"""

    print("\n--- DUPLICATED ---")
    print(df.duplicated().sum()) # 63

    print("\n--- INFO ---")
    print(df.info())
    """
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   type             12137 non-null  str    
 1   sub_type         1749 non-null   str    
 2   text             12137 non-null  str    
 3   listPrice        12134 non-null  float64
 4   sqft             12073 non-null  float64
 5   stories          11630 non-null  float64
 6   beds             12009 non-null  float64
 7   baths            11863 non-null  float64
 8   baths_full       11988 non-null  float64
 9   baths_full_calc  11991 non-null  float64
 10  garage           10224 non-null  float64
 11  year_built       9849 non-null   float64
 12  Price_Per_SqFt   12135 non-null  float64 """

    print("\n--- DESCRIBE ---")
    print(df.describe())
    """
--- DESCRIBE ---
          listPrice          sqft       stories          beds  ...  baths_full_calc        garage   year_built  Price_Per_SqFt
count  1.213400e+04  12073.000000  11630.000000  12009.000000  ...     11991.000000  10224.000000  9849.000000    12135.000000
mean   4.998792e+05   2286.140230      1.442562      3.611791  ...         2.480527      2.134390  2013.043050      208.948589
std    5.409439e+05   1048.659141      0.549794      1.014476  ...         0.915769      0.605927    22.132438      139.697072
min    8.500000e+02    240.000000      1.000000      0.000000  ...         1.000000      0.000000  1654.000000        0.000000
25%    2.880922e+05   1640.000000      1.000000      3.000000  ...         2.000000      2.000000  2015.000000      154.915000
50%    3.749000e+05   2085.000000      1.000000      4.000000  ...         2.000000      2.000000  2023.000000      182.910000
75%    5.390000e+05   2686.000000      2.000000      4.000000  ...         3.000000      2.000000  2025.000000      220.840000
max    1.199500e+07  19600.000000      7.000000     29.000000  ...        26.000000     20.000000  2028.000000     6363.980000"""
    print("\n--- CORR ---")
    print(df.corr(numeric_only=True))

    """
--- CORR ---
                 listPrice      sqft   stories      beds     baths  baths_full  baths_full_calc    garage  year_built  Price_Per_SqFt
listPrice         1.000000  0.627351  0.216422  0.257624  0.505054    0.476207         0.476073  0.392970    0.036315        0.703671
sqft              0.627351  1.000000  0.360569  0.592764  0.738724    0.730893         0.732234  0.529573    0.080574        0.129935
stories           0.216422  0.360569  1.000000  0.246371  0.482285    0.329225         0.327436  0.058266    0.128611        0.055096
beds              0.257624  0.592764  0.246371  1.000000  0.661388    0.721586         0.724754  0.256475    0.170348       -0.048592
baths             0.505054  0.738724  0.482285  0.661388  1.000000    0.881891         0.884293  0.382357    0.207405        0.166772
baths_full        0.476207  0.730893  0.329225  0.721586  0.881891    1.000000         0.995504  0.381525    0.184597        0.147029
baths_full_calc   0.476073  0.732234  0.327436  0.724754  0.884293    0.995504         1.000000  0.382205    0.182488        0.145067
garage            0.392970  0.529573  0.058266  0.256475  0.382357    0.381525         0.382205  1.000000    0.101914        0.194671
year_built        0.036315  0.080574  0.128611  0.170348  0.207405    0.184597         0.182488  0.101914    1.000000       -0.002273
Price_Per_SqFt    0.703671  0.129935  0.055096 -0.048592  0.166772    0.147029         0.145067  0.194671   -0.002273        1.000000"""
    

# CLEAR DATA
#---------------------------------------------
def clear_data(df):

    df.drop(columns=["sub_type","text","Price_Per_SqFt"],inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["listPrice","sqft","stories","beds","baths","baths_full","baths_full_calc"],inplace=True)
    df['garage'] = df['garage'].fillna(df['garage'].median())
    df['year_built'] = df['year_built'].fillna(df['year_built'].median())
    

    return df

# BASIC DATA ANALYSES
#---------------------------------------------
def basic_analysis(df):
    types=df["type"].value_counts()
    plt.figure(figsize=(8,7))
    sb.barplot(x=types.index,y=types.values)
    plt.xticks(rotation=45,ha="right")
    plt.title("PRIMARY PROPERTY TYPES OF HOUSES IN TEXAS - Residential Real Estate")
    plt.xlabel("Types")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid(axis="y")
    plt.savefig(f"{OUTPUT}/analysis_of_primary_property_types_of_houses_in_Texas.png")
    plt.show()

    prices=df["listPrice"].value_counts()
    plt.figure(figsize=(15,7))
    sb.lineplot(x=prices.index,y=prices.values)
    plt.title("PRICES OF HOUSES IN TEXAS - Residential Real Estate")
    plt.xlabel("Prices")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_prices_of_houses_in_Texas.png")
    plt.show()

    area=df["sqft"].value_counts()
    plt.figure(figsize=(15,7))
    sb.lineplot(x=area.index,y=area.values)
    plt.title("TOTAL INTERIOR SQUARE FOOTAGE OF HOUSES IN TEXAS - Residential Real Estate")
    plt.xlabel("Total interior square footage")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_total_interior_square_footage_of_houses_in_Texas.png")
    plt.show()

    floor=df["stories"].value_counts()
    sb.barplot(x=floor.index,y=floor.values)
    plt.title("NUMBER OF FLOORS IN THE PROPERTY - Residential Real Estate")
    plt.xlabel("Floor")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_number_of_floors_of_houses_in_Texas.png")
    plt.show()

    bed=df["beds"].value_counts()
    plt.figure(figsize=(8,7))
    sb.barplot(x=bed.index,y=bed.values)
    plt.title("TOTAL BEDROOM COUNT OF HOUSES IN TEXAS - Residential Real Estate")
    plt.xlabel("Beds")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_total_bedroom_of_houses_in_Texas.png")
    plt.show()

    bath=df["baths"].value_counts()
    plt.figure(figsize=(10,5))
    sb.barplot(x=bath.index,y=bath.values)
    plt.title("TOTAL BATHROOM COUNT (including partial) OF HOUSES IN TEXAS - Residential Real Estate")
    plt.xlabel("Baths")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_total_bathroom_of_houses_in_Texas.png")
    plt.show()

    garages=df["garage"].value_counts()
    plt.figure(figsize=(10,5))
    sb.barplot(x=garages.index,y=garages.values)
    plt.title("GARAGE CAPACITY IN NUMBER OF CARS OF HOUSES IN TEXAS - Residential Real Estate")
    plt.xlabel("Garage Capacity")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_garage_capacity_of_houses_in_Texas.png")
    plt.show()

    years=df["year_built"].astype(int).value_counts().sort_values(ascending=False).head(10).sort_index(ascending=True)
    plt.figure(figsize=(10,5))
    sb.barplot(x=years.index,y=years.values)
    plt.xticks(rotation=45,ha="right")
    plt.title("YEAR OF CONSTRUCTION OF HOUSES IN TEXAS (TOP 10) - Residential Real Estate")
    plt.xlabel("Year of construction")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/analysis_of_year_of_construction_of_houses_in_Texas.png")
    plt.show()

    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(7,6))
    sb.heatmap(corr,annot=True,cmap="coolwarm")
    plt.title("CORRALETION HEATMAP")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT}/correlation_heatmap.png")
    plt.show()

# COMPREHENSIVE ANALYSES
#---------------------------------------------
def comprehensive_analyses(df):
    plt.figure(figsize=(7,5))
    plt.scatter(df["sqft"],df["listPrice"],s=2)
    plt.title("TOTAL INTERIOR SQUARE FOOTAGE VS PRICE")
    plt.xlabel("Total interior square footage")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/total_interior_square_footage_vs_Price.png")
    plt.show()

    plt.figure(figsize=(7,5))
    plt.scatter(df["baths"],df["listPrice"],s=2)
    plt.title("BATHS VS PRICE")
    plt.xlabel("Baths")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{OUTPUT}/baths_vs_Price.png")
    plt.show()

# FEATURE ENGINEERING
#---------------------------------------------
def feature_engineering(df):
    df["house_age"]=2026-df["year_built"]
    df["baths_per_bedroom"]=df["baths"] / df["beds"]

    return df

# MAIN PIPELINE
#---------------------------------------------
def main():
    df=load_data(url)
    inspect_data(df)
    df=clear_data(df)
    df=feature_engineering(df)
    basic_analysis(df)
    comprehensive_analyses(df)

    df=pd.get_dummies(df,columns=["type"],dtype=int)
    df.to_csv("datasets/ml_df.csv")

"""
type
Primary property type (single_family, condos, townhomes, multi_family, apartment, condo_townhome_rowhome_coop)

listPrice
Active listing price in USD

sqft
Total interior square footage

stories
Number of floors in the property

beds
Total bedroom count

baths
Total bathroom count (including partial)

baths_full
Full bathrooms only

baths_full_calc
Calculated full bathroom count (MLS derived field)

garage
Garage capacity in number of cars

year_built
Year of construction"""

# RUN
#---------------------------------------------
if __name__=="__main__":
    main()