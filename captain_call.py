import streamlit as st
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.preprocessing import OneHotEncoder



st.title("Data Processing using Streamlit")
# Define a function to drop null values
st.subheader("An illustrative webpage for those who doesn't love inconsistent units :)")


uploaded_file = st.file_uploader('Select Your data file (CSV)',type="csv")


def return_df(data):
    csv_file = data.to_csv(index=False)
    db = st.sidebar.download_button(label='Download processed CSV',data=csv_file,file_name='processed.csv',mime='text/csv')

def impute_missing_values_with_mean(column_name, data_file):
    mean = data_file[column_name].mean()
    data_file[column_name].fillna(mean, inplace=True)
    return data_file

def impute_missing_values_with_mode(column_name, data_file):
    mode = data_file[column_name].mode().iloc[0]
    data_file[column_name].fillna(mode, inplace=True)
    return data_file

def impute_missing_values_with_median(column_name, data_file):
    median = data_file[column_name].median()
    data_file[column_name].fillna(median, inplace=True)
    return data_file

def detect_outliers_zscore(column, threshold=2):
    z_scores = (column - column.mean()) / column.std()
    outliers = np.abs(z_scores) > threshold
    return outliers    

def scale_numeric_columns(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df   

def identify_outliers(df):
    threshold = 2
    outliers = []
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_columns:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        for i, z in enumerate(z_scores):
            if abs(z) > threshold:
                outliers.append(col)
                break
    return set(outliers)


def encode_features(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categories = []

    for column in categorical_columns:
            categories.append((column, df[column].nunique()))

    # Display the categorical variables and their categories
    if len(categories) > 0:
        st.subheader("Categorical Variables")
        for category in categories:
            st.write(f"{category[0]}: {category[1]} categories")
    else:
        st.subheader("No categorical variables found")

    encoder = OneHotEncoder()
    encoded_array = encoder.fit_transform(df[categorical_columns]).toarray()
    feature_names = []
    for i, column in enumerate(categorical_columns):
        feature_names += [f"{column}_{category}" for category in encoder.categories_[i]]
    encoded_df = pd.DataFrame(encoded_array, columns=feature_names)
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(categorical_columns, axis=1, inplace=True)
    return df






if uploaded_file is not None:
    data_file = pd.read_csv(uploaded_file)




    if st.checkbox("EDA"):
        var = st.selectbox("Show me the",['','shape','column data types','data head','summary'],index=0)

        if var == "data head":
            st.write(data_file.head(5))
        elif var == 'shape':
            st.write(f"There are total {data_file.shape[0]} rows and {data_file.shape[1]} columns") 
        elif var == "column data types":
            st.write(data_file.dtypes)       
        elif var == "summary":
            st.write(data_file.describe(include="all").T)                



    if st.checkbox("Handling Missing Values"):
        st.write("**"+"Missing Values Count"+"**")
        # count the number of missing values and calculate the percentage missing for each column
        missing_count = data_file.isnull().sum()
        missing_pct = data_file.isnull().mean() 


        # combine the count and percentage missing into a new DataFrame
        missing_data = pd.concat([missing_count, missing_pct], axis=1, keys=['Total Missing', 'Percent Missing'])


        # print the result
        st.write(missing_data)


        
        count = 0
        remove_col = [] 
        for i in range(len(missing_pct)):
            if missing_pct[i] > 0.5:
                remove_col.append(data_file.columns.tolist()[i])
                count += 1
        if count >= 1:
            if count == 1:        
                st.write(f"There is only {count} column that have more than 50% missing values.\
                    Therefore it is advisable to remove that column")
            else:
                st.write(f"There are {count} columns that have more than 50% missing values.\
                    Therefore it is advisable to remove these columns")
                    
            del_col = st.multiselect("Delete columns",remove_col)

            for i in del_col:
                data_file.drop([i],axis=1,inplace=True)



        # count the total number of missing values in the DataFrame
        total_missing = data_file.isnull().sum().sum()

        # calculate the total number of values in the DataFrame
        total_values = data_file.size

        #  calculate the percentage of missing values
        missing_pct = (total_missing / total_values) * 100

        if missing_pct <= 5 and missing_pct != 0:
            st.write(f"The DataFrame has only {missing_pct:.2f}% of missing values, therefore it is advisable to remove all missing\
                values from the DataFrame")

        elif missing_pct > 5 and missing_pct != 100:
            st.write(f"The DataFrame has significant {missing_pct:.2f}% of missing values, therefore it is advisable to replace rather than remove missing\
                values from the DataFrame")
                

        if missing_pct != 0:
            var1 = st.selectbox("Based on your subject knowledge and given advise choose either of \
                the two given options.",['','Remove Missing Values','Replace Missing Values'],index=0)

            if var1 == 'Remove Missing Values':
                data_file = data_file.dropna()
                data_file = data_file.reset_index(drop=True)
                st.write("Yah!! All rows that have missing values are removed.")
                

            elif var1 == 'Replace Missing Values':
                columns_with_missing_values = data_file.columns[data_file.isnull().any()].tolist()
                column_info_list = {}

                # Populate the list with dictionaries containing column names and data types
                for column_name in columns_with_missing_values:
                    data_type = data_file[column_name].dtype
                    column_info_list[column_name] = str(data_type)

                mode_count = 0
                median_count = 0
                mean_count = 0 
                mean_col = []
                median_col = []
                mode_col = [] 
                for i in column_info_list:
                    if column_info_list[i] == 'object':
                        impute_missing_values_with_mode(i, data_file)
                        mode_count += 1
                        mode_col.append(i)
                    else:
                        outliers = detect_outliers_zscore(data_file[i])
                        if outliers.any():
                            impute_missing_values_with_median(i, data_file)
                            median_count += 1
                            median_col.append(i)
                        else:
                            impute_missing_values_with_mean(i, data_file)
                            mean_count += 1
                            mean_col.append(i)

                st.write("We did it. Congrats :)") 
                st.write("**"+"What is basically happening?"+"**")
                st.write(f"Missing values of the column ({mode_col} in our DataFrame) with object data type has to be filled thier most frequent object.  \
                    While column ({median_col} in our DataFrame) that contain outliers has to be treated with their corresponding median and as you gussed it column ({mean_col} in our DataFrame)\
                    with no outliers has to be filled with mean") 
                    
        else:
            st.write("Our data doesn't contain missing values. We are good to go ....")        

    if st.checkbox("Feature Scaling"):
        with st.expander('Is it necessary to perform?'):
            st.write("Scaling is not necessary for algorithms like decision trees, which are not distance-based. Distance based models however\
                , must have scaled features without any exception.")

        var2 = st.selectbox("Based on your subject knowledge and given advise choose either of \
                the two given options.",['','Perform Standard Scaling','I do not wish to perform Standard Scaling'],index=0)

        if var2 == "Perform Standard Scaling":
            scale_numeric_columns(data_file)
            st.write("Done. Great we have made so much of progress..")

    if st.checkbox("Outliers Treatment"): 
        with st.expander('Outliers? How to deal with it!!'):
            st.write("Outliers are data points that do not conform with the predominant pattern observed in the data.\
             They can cause disruptions in the predictions by taking the calculations off the actual pattern. Hence it is \
             advisable to remove outliers prior to appyling any machine learning technique.")  

        outlier_cols = identify_outliers(data_file)

        outliers_cols_list = list(outlier_cols)

        if len(outliers_cols_list) >= 1:
            if len(outliers_cols_list) >1:
                st.write(f"{outliers_cols_list} these columns from our dataset contain outliers.")
            else:
                st.write(f"{outliers_cols_list} only this column from our dataset contain outliers.")    

            var3 = st.selectbox("Based on your subject knowledge and given advise choose either of \
                    the two given options.",['','Remove Outliers','Keep Outliers'],index=0)


            if var3 == "Remove Outliers":
                numeric_columns = data_file.select_dtypes(include=[np.number]).columns.tolist()
                z_scores = stats.zscore(data_file[numeric_columns])
                outliers = (abs(z_scores) > 2).any(axis=1)
                data_file = data_file[~outliers]
                st.write("We did it for you!! Congrats")  

        else:
            st.write("Bingo. No column contain outliers. Let's move forward")        
        
    
    if st.checkbox("Feature Encoding"):
        with st.expander('Sounds very hectic. What is it now?'):
            st.write("Sometimes, data is in a format that can’t be processed by machines. For\
            instance, a column with string values, like names, will mean nothing to a model that depends only on numbers. So, we need to process the data to help the model interpret it.")

        categorical_columns = data_file.select_dtypes(include=['object']).columns.tolist()
        categories = []

        for column in categorical_columns:
            categories.append((column, data_file[column].nunique()))

        # Display the categorical variables and their categories
        if len(categories) > 0:
            st.write(f"In the given dataset there are total {len(categorical_columns)} categorical variables. Namely,")
            for category in categories:
                st.write(f"{category[0]} with {category[1]} categories")

            var4 = st.multiselect("Choose columns to perform One Hot Encoding",categorical_columns)

            if len(var4) >= 1:
                encoder = OneHotEncoder()
                encoded_array = encoder.fit_transform(data_file[var4]).toarray()
                feature_names = []

                for i, column in enumerate(var4):
                    feature_names += [f"{column}_{category}" for category in encoder.categories_[i]]

                encoded_df = pd.DataFrame(encoded_array, columns=feature_names)
                data_file = pd.concat([data_file, encoded_df], axis=1)
                data_file.drop(var4, axis=1, inplace=True)

                return_df(data_file)

                st.write("Great!. Processed file has been generated  ")


        else:
            st.write("No categorical variables found in our dataset!!")

            st.write("Processed file has been generated :) ")
            return_df(data_file)

        
        









                

            


 

