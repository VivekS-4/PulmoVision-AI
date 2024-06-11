

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd

russian_data = {
    "Relationship": [
        "Друг",
        "Мать",
        "Близкий ро",
        "Брат",
        "Отец",
        "Сестра",
        "Дочь",
        "Сын",
        "Дальний ро",
        "Муж",
        "мать",
        "Жена",
        "начальник отдела"
    ]
    }

translation_dict = {
        "Друг": "Friend",
        "Мать": "Mother",
        "Близкий ро": "Close relative",
        "Брат": "Brother",
        "Отец": "Father",
        "Сестра": "Sister",
        "Дочь": "Daughter",
        "Сын": "Son",
        "Дальний ро": "Distant relative",
        "Муж": "Husband",
        "мать": "Mother",
        "Жена": "Wife",
        "начальник отдела":"Friend"
    }

def process_list(json_data):
    # Convert JSON data into a DataFrame
    df = pd.DataFrame(json_data)



    categorical_features = df.select_dtypes(include=['object', 'bool'])
    numerical_features = df.select_dtypes(include=['int64', 'float'])
    print(categorical_features)

    df.drop(columns=['ID'], inplace=True)
    df.drop(columns=['CLNT_JOB_POSITION'], inplace=True)
    df.drop_duplicates(inplace=True)
    columns_to_drop = ['CLNT_SALARY_VALUE', 'LDEAL_YQZ_COM', 'LDEAL_YQZ_CHRG', 'AVG_PCT_MONTH_TO_PCLOSE', 'MAX_PCLOSE_DATE', 'LDEAL_AMT_MONTH', 'AVG_PCT_DEBT_TO_DEAL_AMT', 'LDEAL_YQZ_PC', 'LDEAL_TENOR_MIN', 'LDEAL_TENOR_MAX', 'DEAL_YQZ_IR_MAX', 'LDEAL_DELINQ_PER_MAXYQZ', 'MED_DEBT_PRC_YQZ', 'DEAL_YQZ_IR_MIN', 'LDEAL_USED_AMT_AVG_YQZ', 'CLNT_JOB_POSITION_TYPE', 'APP_CAR', 'APP_TRAVEL_PASS', 'APP_DRIVING_LICENSE', 'APP_KIND_OF_PROP_HABITATION', 'APP_POSITION_TYPE', 'APP_REGISTR_RGN_CODE', 'CNT_TRAN_CLO_TENDENCY1M', 'SUM_TRAN_CLO_TENDENCY1M', 'APP_EMP_TYPE', 'APP_COMP_TYPE', 'APP_EDUCATION', 'APP_MARITAL_STATUS', 'SUM_TRAN_MED_TENDENCY1M', 'CNT_TRAN_MED_TENDENCY1M', 'CLNT_TRUST_RELATION', 'DEAL_GRACE_DAYS_ACC_MAX', 'DEAL_GRACE_DAYS_ACC_AVG', 'DEAL_GRACE_DAYS_ACC_S1X1', 'CNT_TRAN_AUT_TENDENCY1M', 'SUM_TRAN_AUT_TENDENCY1M', 'LDEAL_ACT_DAYS_ACC_PCT_AVG', 'LDEAL_ACT_DAYS_PCT_TR3', 'LDEAL_ACT_DAYS_PCT_CURR', 'LDEAL_ACT_DAYS_PCT_TR', 'LDEAL_ACT_DAYS_PCT_TR4', 'LDEAL_DELINQ_PER_MAXYWZ', 'LDEAL_USED_AMT_AVG_YWZ', 'MED_DEBT_PRC_YWZ', 'DEAL_YWZ_IR_MAX', 'DEAL_YWZ_IR_MIN', 'LDEAL_ACT_DAYS_PCT_AAVG', 'SUM_TRAN_AUT_TENDENCY3M', 'CNT_TRAN_AUT_TENDENCY3M', 'CNT_TRAN_CLO_TENDENCY3M', 'SUM_TRAN_CLO_TENDENCY3M', 'SUM_TRAN_MED_TENDENCY3M', 'CNT_TRAN_MED_TENDENCY3M', 'PRC_ACCEPTS_A_EMAIL_LINK', 'PRC_ACCEPTS_A_MTP', 'CNT_ACCEPTS_TK', 'CNT_ACCEPTS_MTP', 'PRC_ACCEPTS_A_AMOBILE', 'PRC_ACCEPTS_TK', 'PRC_ACCEPTS_A_POS']
    df["CLNT_TRUST_RELATION"] = df["CLNT_TRUST_RELATION"].replace(translation_dict)

    # Function to handle missing values for selected columns
    def handle_missing_values(df,columns_to_drop):
            # Dropping the unnecessary columns with high proportion of missing values
        df.drop(columns=columns_to_drop, inplace=True)

        #filling the missing values of numerical columns with mean
        numerical_columns = df.select_dtypes(include=['number']).columns
        for column in numerical_columns:
            if df[column].isnull().any():
                df[column].fillna(df[column].mean(), inplace=True)

        # filling the missing values of categorical columns with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            if df[column].isnull().any():
                mode_value = df[column].mode()[0]
                df[column].fillna(mode_value, inplace=True)


        return df

    # calling the function to handle missing values.
    processedBankData = handle_missing_values(df,columns_to_drop)

    df = processedBankData


    #df['APP_REGISTR_RGN_CODE'] = df['APP_REGISTR_RGN_CODE'].astype('object')
    df['CR_PROD_CNT_IL'] = df['CR_PROD_CNT_IL'].astype('object')
    df['CR_PROD_CNT_CCFP'] = df['CR_PROD_CNT_CCFP'].astype('object')
    df['CR_PROD_CNT_CC'] = df['CR_PROD_CNT_CC'].astype('object')
    df['CR_PROD_CNT_PIL'] = df['CR_PROD_CNT_PIL'].astype('object')
    df['CR_PROD_CNT_TOVR'] = df['CR_PROD_CNT_TOVR'].astype('object')
    df['CR_PROD_CNT_TOVR'] = df['CR_PROD_CNT_TOVR'].astype('object')
    # df = df.dropna(subset=['APP_REGISTR_RGN_CODE'])
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = df[column].astype('category').cat.codes
        df[column] = df[column].astype('category').cat.codes
    
    # imputer = SimpleImputer(strategy='mean')
    # df = pd.DataFrame(imputer.fit_transform(df))

    df.drop(columns=['PACK'], inplace=True)
    # print(df.columns)

    return df

