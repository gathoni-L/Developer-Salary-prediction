"""
Data cleaning utilities for developers salary raw csv
We'll use the function in train.py and app.py
"""

import pandas as pd
import numpy as np

# Constants
TARGET ='ConvertedCompYearly'
LOG_TARGET='log_salary'
Salary_MIN= 10_000
Salary_MAX=500_000
TOP_COUNTRIES= 25

SELECTED_FEATURES= ['Country','YearsCode','EdLevel','Employment','LanguageHaveWorkedWith',
                    # New selected features,v2
                    'DevType', # developer role
                    'OrgSize', # Company size
                    'RemoteWork', #remote/hybrid/in-person
                    'WorkExp', # years of professional experience
                    'Industry', # eg tech,financehealthcare
                    'Age',
                    'ICorPM',# Individual contributor or manager
                    'DatabaseHaveWorkedWith',
                    'PlatformHaveWorkedWith',
                    'ToolCountWork'
                    ]
# Target encoding
# features that we want to target encode
TARGET_ENC_FEATURES = ["Country", "DevType", "Industry"]


# Ordinal mappings
ED_LEVEL_ORDINAL:dict [str,int]={
        "Bachelor's":0,
        "Master's":1,
        "Some college":2,
        "Assoiciates":3,
        "Professional":4,
        "High school":5,
        "Primary School":6,
        "Other":7

        }

REMOTE_ORDINAL:dict[str,int]={

    "In_person":0,
    "Hybrid":1,
    "Remote":2,
    "Other":1 # Can be mapped as hyvbrid
    }

# Cleaning features
def clean_years_code(series:pd.Series) -> pd.Series:
    series = pd.to_numeric(series,errors='coerce')
    return(series)

def clean_work_exp(series:pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce')
    return(series)


def clean_education(series: pd.Series)-> pd.Series:
    """Standardizing Edlevel into a set of clean categories"""
    mapping ={
        "Bachelor's degree(B.A.,, B.S., B.Eng., etc.)":"Bachelor's",
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)":"Master's",
        "Some college/university study without earning a degree":"Some college",
        "Associate degree (A.A., A.S., etc.)":"Assoiciates",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)":"Professional",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)":"High school",
        "Other (please specify)":"Other",
        "Primary/elementary school":"Primary School",
                                                                                  
    }
    incomplete=series.map(mapping).fillna('Other')
    return incomplete.map(ED_LEVEL_ORDINAL)

def clean_employment(series: pd.Series) -> pd.Series:
    """Clean employment column"""
    
    def simplify(val):
        if pd.isna(val):
            return np.nan
        val=str(val)
        if 'employed' in val.lower() or 'full-time' in val.lower():
            return 'Full-time'
        elif 'Independent contractor,freelancer, or self-employed'in val:
            return 'Freelance/Self-employed'
        elif 'student' in val.lower():
            return 'Student'
        else:
            return 'Other'
        
    return series.apply(simplify)

def clean_age(series:pd.Series) -> pd.Series:

    """
    Mapping age bands to ordinal intergers
    """
    mapping={
        "18-24 years old":0,
        "25-34 years old":1,
        "35-44 years old":2,
        "45-54 years old":3,
        "55-64 years old":4,
        "65 years or older":5,
        "Prefer not to say":np.nan
    }
    return series.map(mapping)


def clean_icorpm (series:pd.Series) ->pd.Series:
    """
    Clean ic or pm role to binary 1 for manager and 0 for independent contractor
    """
    # Underscore before to show the fn is inside another function
    def _map(val):
        if pd.isna(val):
            return np.nan
        v = str(val).lower()
        if "manager" in v or "lead in v":
            return 1
        return 0
    return series.apply(_map)

    

def count_language(series: pd.Series) -> pd.Series:
    """
    Convert comma separated list to count

    Example: 'Bash/Shell(all shells): Dart;Sql'-> 3
    """
    def _count(val):
        if pd.isna(val) or val == '':
            return np.nan
        return len(str(val).split(";"))
    
    return series.apply(_count)
# Languages that consistenty pay above average in stack overflow salary surveys
HIGH_PAY_LANGUAGES ={
    "Go","Rust","Scala","Exilir","Clojure","Kotlin","Swift","F#","Erlang","Zig",
    "OCaml","Haskell"
}
def has_high_pay_language(series:pd.Series)->pd.Series:
    """
    return 1 if the respondent has a high paying language
    """
    def _check(val):
        if pd.isna(val):
            return 0
        langs={lang.strip() for lang in str(val).split(";")}
        return 1 if langs & HIGH_PAY_LANGUAGES else 0
    return series.apply(_check)

def group_rare_countries(series: pd.Series, top_n: int=TOP_COUNTRIES)-> pd.Series:
    """
    Keep only the top N most common countries(default = 25). Replace all other columns with 'other'
    """
    top_countries = series.value_counts().head(top_n).index.tolist()
    return series.apply(lambda x:x if x in top_countries else 'Other')

def clean_remote_work(series:pd.Series) -> pd.Series:
    """
    Map remote work values to ordinal integers
    """
    mappings={
        "Remote": "Remote",
        "Hybrid (some in-person, leans heavy to flexibility)": "Hybrid",
        "Hybrid (some remote, leans heavy to in-person)": "Hybrid",
        "In-person": "In-person",
        "Your choice (very flexible, you can come in when you want or just as needed)": np.nan
    }
    incomplete =series.map(mappings).fillna("Other")
    return incomplete.map(REMOTE_ORDINAL)

def count_platformsworkedwith(series:pd.Series) -> pd.Series:
    """
    Convert the semi colon separated list to count
    """
    def _count(val):
        if pd.isna(val) or val == '':
            return np.nan
        return len(str(val).split(";"))
    
    return series.apply(_count)

def count_databasesused(series:pd.Series) -> pd.Series:
    """
    Counts the number of databases used
    """
    def _count(val):
        if pd.isna(val) or val == '':
            return np.nan
        return ;len(str(val).split(";"))

    return series.apply(_count)

def clean_dev_type(series: pd.Series) -> pd.Series:
    """
      Picking the primary dev role
      Mapping using conditional checks
      """
    def _primary(val): 
        if pd.isna(val):
            return "Other"
        low = str(val).lower()
        if "full-stack" in low:
            return "Full-stack"
        if "back-end" in low:
            return "Back-end"
        if "front-end" in low:
            return "Front-end"
        if "data scientist" in low or "machine learning" in low or "ml" in low:
            return "Data/ML"
        if "data engineer" in low or "data analyst" in low:
            return "Data/ML"
        if "devops" in low or "cloud" in low or "site reliability" in low:
            return "DevOps/Cloud"
        if "mobile" in low:
            return "Mobile"
        if "embedded" in low or "hardware" in low:
            return "Embedded/Hardware"
        if "security" in low:
            return "Security"
        if "manager" in low or "executive" in low or "director" in low:
            return "Management"
        return "Other"
    return series.apply(_primary)
        

def clean_industries(series:pd.Series)->pd.Series:
    """
    Keep only top industries
    """
    top =series.value_counts().head(10).index.tolist()
    return series.apply(lambda x:x if x in top else 'other').fillna('others')
# The code above means we pick only the top 10 industry counts and the rest are filled with other

def cleaned_orgsize(series:pd.Series) ->pd.Series:
    mapping ={
        'Just me - I am a freelancer, sole proprietor, etc.':0,
        'Less than 20 employees':1,
        '20 to 99 employees':2,
        '100 to 499 employees':3,
        '500 to 999 employees': 4,
        '1,000 to 4,999 employees':5,
        '5,000 to 9,999 employees':6,
        '10,000 or more employees':7,
        'I don’t know':np.nan     
    }

    return series.map(mapping)


# Add interaction feature
def add_interaction_features(df):
        df['yearsCode_sq'] = df['YearsCode'] ** 2
        df['WorkExp_sq'] = df['WorkExp'] ** 2
        df['Exp_ratio'] = df['WorkExp']/ (df['YearsCode'] + 1)
        df['Tech_breadth'] = (df['LanguageCount'] + df['DatabaseCount'] + df['PlatformCount']).fillna(0)
        return df


       


def load_and_clean(filepath:str) -> pd.DataFrame:
    """
    Loading the raw stack overflow  survey and return a clean DF
    The df will be ready for the scikit -learn pipeline

    Pass a filepath to the raw csv file
    returns pd.DataFrame(DF with features+ target column) 
    """


    df = pd.read_csv(filepath,low_memory=False)
    print(f"This is the raw shape: {df.shape}")

    # Step 1- Keep only the valid salary
    df = df.dropna(subset=[TARGET])
    df= df[df[TARGET].between(Salary_MIN,Salary_MAX)]
    print(f"Shape after salary filter:{df.shape}")

    # Step 1.5-v2 of model
    df['log_salary']=np.log1p(df[TARGET])

    # Step 2- Check whether the feature and the target column exist
    cols_needed = SELECTED_FEATURES + ['log_salary']
    cols_available = [c for c in cols_needed if c in df.columns]

    missing_cols = set(cols_needed)- set(cols_available)
    if missing_cols:
        print(f"Columns not found in dataset: {missing_cols}")

    df =df[cols_available].copy()
    print(f"selected {len(cols_available)} columns, expecting 6")

    # 3. Cleaning specific columns
    if 'YearsCode' in df.columns:
        df['YearsCode'] = clean_years_code(df['YearsCode'])
    if 'WorkExp' in df.columns:
        df['WorkExp']=clean_work_exp(df["WorkExp"])

    if 'EdLevel' in df.columns:
        df['EdLevel'] = clean_education(df['EdLevel'])

    if 'Employment' in df.columns:
        df['Employment'] = clean_employment(df['Employment'])

    if 'Age' in df.columns:
        df['Age'] = clean_age(df['Age'])

    if 'OrgSize' in df.columns:
        df['OrgSize'] = cleaned_orgsize(df['OrgSize'])
    
    if 'ICorPM' in df.columns:
        df['ICorPM'] = clean_icorpm(df['ICorPM'])

    if 'RemoteWork' in df.columns:
        df['RemoteWork'] = clean_remote_work(df['RemoteWork'])

    if 'Industry' in df.columns:
        df['Industry'] = clean_industries(df['Industry'])

    if 'DevType' in df.columns:
        df['DevType'] = clean_dev_type(df['DevType'])

    if 'PlatformHaveWorkedWith' in df.columns:
        df['PlatformCount'] = count_platformsworkedwith(df['PlatformHaveWorkedWith'])
        df.drop(columns=['PlatformHaveWorkedWith'])
    

    if 'DatabaseHaveWorkedWith' in df.columns:
        df['DatabaseCount'] = count_databasesused(df['DatabaseHaveWorkedWith'])
        df.drop(columns=['DatabaseHaveWorkedWith'])

    if 'LanguageHaveWorkedWith' in df.columns:
        df['has_high_pay_lang'] =has_high_pay_language(df['LanguageHaveWorkedWith'])
        df['LanguageCount'] = count_language(df['LanguageHaveWorkedWith'])
        df = df.drop(columns=['LanguageHaveWorkedWith'])

    if 'Country' in df.columns:
        df['Country'] = group_rare_countries(df['Country'])
    
    if 'ToolCountWork' in df.columns:
        df['ToolCountWork']=pd.to_numeric(df['ToolCountWork'],errors='coerce')

    EMPLOYMENT_KEEP=["Full-time","Freelance/Self-employed"]

    #Step 4-Filter employment(Keep only full time and free lance)
    if 'Employment' in df.columns:
        before =len(df)
        df=df[df['Employment'].isin(EMPLOYMENT_KEEP)]
        df["Employment"]=(df["Employment"]).isin(EMPLOYMENT_KEEP)
        df['Employment']= (df['Employment'] =='Full-time').astype(int)

        print(f"Employment filter:{before}) -> {len(df)} rows,"
              f"We only kept Full-time and freelance")
        

    # Step 5 add interaction features
    df= add_interaction_features(df)

    #step 6  Drop rows where all features are Nan(Handling this edge case)
    df = df.dropna(how = 'all')
    print (f"Clean data shape:{df.shape}")
    print(f"Missing values per column: \n{df.isna().sum().to_string()} \n")

    return df

def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """
    return (categorical columns , numeric columns) from the cleaned DF excluding the target variable 
    """

    non_target =[c for c in df.columns if c != 'log_salary']
    target_enc=[c for c in TARGET_ENC_FEATURES if c in non_target]
    num_cols = df[non_target].select_dtypes(include=['number']).columns.to_list()

    return target_enc, num_cols

















    
