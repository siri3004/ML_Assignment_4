# ML_Assignment_4

Video Link: https://drive.google.com/file/d/1bx1Aoou_b6Dwhj2v1jQjze5LOvzmAnX7/view

#Pandas

#Q1 Read the provided CSV file ‘data.csv’.

    import pandas as pd
    df = pd.read_csv('C:/Users/poloj/Desktop/ML_Assignment_4/data.csv')
    
For data manipulation and analysis, import the pandas library.Use the read_csv() function from pandas to read the CSV file named data.csv. The DataFrame object is 

assigned to the variable df, which allows for further manipulation and analysis of the data in the file.

    
#Q2 Show the basic statistical description about the data.

    DataDescription = df.describe()
    print(DataDescription)

To the pandas DataFrame object df, the describe() function is applied. The output of the describe() function is stored in new pandas DataFrame object named DataDescription.

The contents of DataDescription are printed to the console using the print() function.

  
#Q3 Check if the data has null values and Replace the null values with the mean

    # Check for null values
    null_rows = df[df.isnull().any(axis=1)]

    print(null_rows)

    # Replace null values in the Calories column with the mean value

    df['Calories'].fillna(df['Calories'].mean(), inplace=True)

    # Check for null values again
        null_rows = df[df.isnull().any(axis=1)]
        print(null_rows)  
    
•	Checks whether the DataFrame object df contains any null values using the isnull() function, and returns the rows containing null values using the any() function

with axis=1. It replaces the null values in the Calories column with the mean value of the column using the fillna() function with inplace=True.

•	Checks for null values again and prints the rows containing null values to the console.



#Q4 Select at least two columns and aggregate the data using: min, max, count, mean.

    # Select two columns and aggregate the data

        agg_df = df[['Duration', 'Calories']].agg(['min', 'max', 'count', 'mean'])

    # Display the aggregated data

        print(agg_df)

The DataFrame object df is assigned to a new DataFrame object by selected with two columns (Duration and Calories) and the agg() function is used on the new

DataFrame object with four aggregate methods (min, max, count, and mean). The output of the aggregation is stored in a new DataFrame object agg_df.


#Q5 Filter the dataframe to select the rows with calories values between 500 and 1000.

    # Filter the DataFrame to select rows with Calories values between 500 and 1000

        filtered_df = df[(df['Calories'] >= 500) & (df['Calories'] <= 1000)]

    # Display the filtered DataFrame

        print(filtered_df)    
 
Boolean indexing to select rows where the Calories column has a value between 500 and 1000 used to filter the DataFrame object df. The filtered data is assigned to 

a new DataFrame object named filtered_df.



#Q6 Filter the dataframe to select the rows with calories values > 500 and pulse < 100

    # Filter the DataFrame to select rows with Calories values > 500 and Pulse values < 100

        filtered_df = df[(df['Calories'] > 500) & (df['Pulse'] < 100)]

    # Display the filtered DataFrame

        print(filtered_df)
    
Boolean indexing to select rows where the Calories column has a value greater than 500 and the Pulse column has a value less than 100 filters the DataFrame object

df. Stores the filtered data in a new DataFrame object named filtered_df.

#Q7 Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”

    # Create a new DataFrame with all columns except Maxpulse

        df_modified = df.drop('Maxpulse', axis=1)

    # Display the modified DataFrame

        print(df_modified)    

•	From above code creates a new DataFrame object named df_modified which contains all the columns from the DataFrame object df except for the Maxpulse column.

•	Uses the drop() function with axis=1 to drop the Maxpulse column and assigns the modified DataFrame object to the new variable df_modified.

•	It prints the contents of the modified DataFrame object df_modified to the console using the print() function.


#Q8 Delete the “Maxpulse” column from the main df dataframe

    # Delete the Maxpulse column from the main DataFrame

        df.drop('Maxpulse', axis=1, inplace=True)

    # Display the modified DataFrame

    print(df)    
    
To remove the Maxpulse column from the DataFrame object df it uses the drop() function with axis=1 and inplace=True and prints the contents of the modified 

DataFrame object df to the console using the print() function.



#Q9 Convert the datatype of Calories column to int datatype.

    # Convert the data type of the Calories column from float to int
    df['Calories'] = df['Calories'].astype(int)

    # Display the modified DataFrame
    print(df)

•	To convert the data type of the Calories column in the DataFrame object df from float to integer it uses the astype() function and assigns the modified DataFrame

object back to the df variable to store the modified data.It prints the contents of the modified DataFrame object df to the console using the print() function.


#Q10 Using pandas create a scatter plot for the two columns (Duration and Calories).

    import pandas as pd
    import matplotlib.pyplot as plt
    # Load the CSV file
    df = pd.read_csv('C:/Users/poloj/Desktop/ML_Assignment_4/data.csv')

    # Create a scatter plot of the Duration and Calories columns
    df.plot(kind='scatter', x='Duration', y='Calories')

    # Show the plot
    plt.show()
    
•	It imports the pandas library as pd and the matplotlib.pyplot library as plt and uses the read_csv() function to load the CSV file named data.csv and store the contents in a pandas DataFrame object named df.

•	Creates a scatter plot of the Duration and Calories columns in the DataFrame object df using the plot() function with kind='scatter', x='Duration', and y='Calories'.
    
#Titanic

#Q1 Find the correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case inclass.
    # a. Do you think we should keep this feature?

    import pandas as pd

    # Assuming the CSV data is stored in a variable called 'csv_data'
    data = pd.read_csv('C:/Users/poloj/Desktop/ML_Assignment_4/train.csv')

    # Convert 'Sex' column to numerical values
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Calculate the correlation
    correlation = data[['Survived', 'Sex']].corr().iloc[0, 1]
    print("Correlation between 'Survived' and 'Sex':", correlation)
    
•	Imports the pandas library as pd and uses the read_csv() function to read the Titanic dataset CSV file named train.csv and store the contents in a pandas DataFrame object named data.

•	Using the map() function it maps the values in the Sex column to numerical values (0 for male and 1 for female) It calculates the correlation between the Survived and Sex columns using the corr() function and selects the correlation value using the iloc[0, 1] syntax.

•	It prints the calculated correlation value to the console using the print() function.
    
a. Do you think we should keep this feature? 

•	The correlation value between 'Survived' and 'Sex' is 0.5434. That indicates a moderate positive correlation between the two variables which means that as the value of 'Sex' (0 for male, 1 for female) increases, the probability of survival also increases.

•	So, we can conclude that the 'Sex' feature is relevant for predicting survival, where it has a moderate correlation with the target variable and it may be useful to keep this feature in our analysis.
    
#Q2 Do at least two visualizations to describe or show correlations

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load and preprocess data
    data = pd.read_csv('C:/Users/poloj/Desktop/ML_Assignment_4/train.csv')
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Find and handle missing values
    print("Number of missing values in each column:")
    print(data.isna().sum())

    # Fill missing values in 'Age' with the mean
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    # You can choose to fill other missing values or drop the rows with missing values.
    # For example, to drop rows with missing values in the 'Embarked' column:
    data.dropna(subset=['Embarked'], inplace=True)

    # Heatmap of correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()

    # Pairplot for selected features
    selected_features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    sns.pairplot(data[selected_features])
    plt.show()    

•	Using pandas load and preprocess the Titanic dataset which handle missing values in the dataset. Create a heatmap of correlations using seaborn's sns.heatmap() function.

•	Create a pairplot for selected features using seaborn's sns.pairplot() function.

•	Display the visualizations using matplotlib's plt.show() function which creates a pairplot for selected features using the sns.pairplot() function and the list of selected features.



#Q3 Implement Naïve Bayes method using scikit-learn library and report the accuracy. 

    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Handling missing values
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    # Selecting features and target
    X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = data['Survived']

    # Scaling the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Naïve Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Predicting and evaluating the model
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  
    print("Accuracy of Naïve Bayes model:", accuracy)
    
Using StandardScaler(), select features and target variables and scale the features and split the dataset into train and test sets. Using GaussianNB(), train the

Naïve Bayes model and make predictions on the test set and evaluate the model using accuracy_score().

    
    
#Glass

#Q1 Implement Naïve Bayes method using scikit-learn library

    # a. Use the glass dataset available in Link also provided in your assignment.
    
    # b. Use train_test_split to create training and testing part.
    
#Evaluate the model on testing part using score and classification_report(y_true, y_pred)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report

    # Load the dataset
    file_path = "C:/Users/poloj/Desktop/ML_Assignment_4/glass.csv"
    df = pd.read_csv(file_path)

    # Split the dataset into training and testing parts
    X = df.drop("Type", axis=1)
    y = df["Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Naïve Bayes model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", score)
    print("\nClassification Report:\n", report)
    
•	Imports the necessary libraries including pandas, train_test_split, GaussianNB, and classification_report and loads the glass.csv dataset into a pandas DataFrame object df.

•	Using train_test_split() it splits the dataset into training and testing parts and trains the Naïve Bayes model using GaussianNB().

•	It evaluates the model using the score() method to calculate the accuracy and the classification_report() function to generate a classification report.

    

#Q2 Implement linear SVM method using scikit library

    # a. Use the glass dataset available in Link also provided in your assignment.
    
    # b. Use train_test_split to create training and testing part.
    
#Evaluate the model on testing part using score and classification_report(y_true, y_pred)
    
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report

    # Load the dataset
    file_path = "C:/Users/poloj/Desktop/ML_Assignment_4/glass.csv"
    df = pd.read_csv(file_path)

    # Split the dataset into training and testing parts
    X = df.drop("Type", axis=1)
    y = df["Type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the linear SVM model
    model = SVC(kernel="linear", C=1)
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", score)
    print("\nClassification Report:\n", report)
    
•	It imports the necessary libraries including warnings, train_test_split, SVC, and classification_report and loads the glass.csv dataset into a pandas DataFrame object df.

•	Splits the dataset into training and testing parts using train_test_split() and trains the linear SVM model using SVC() with a linear kernel and regularization parameter C=1.

•	It evaluates the model using the score() method to calculate the accuracy and the classification_report() function to generate a classification report.

•	It prints the accuracy and classification report to the console using the print() function.

    
    
#Q3 Do at least two visualizations to describe or show correlations in the Glass Dataset. 

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load the dataset
    file_path = "C:/Users/poloj/Desktop/ML_Assignment_4/glass.csv"
    df = pd.read_csv(file_path)

    # Bar plot of glass type distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x="Type", data=df, palette="bright")
    plt.title("Glass Type Distribution")
    plt.xlabel("Glass Type")
    plt.ylabel("Frequency")
    plt.show()

    # Scatter plot matrix of selected features
    selected_features = ["RI", "Na", "Mg", "Al", "Si", "Type"]
    sns.pairplot(df[selected_features], hue="Type", palette="bright", markers=".", diag_kind="hist")
    plt.title("Scatter Plot Matrix of Selected Features")
    plt.show()
    
•	All the necessary libraries including pandas, seaborn, and matplotlib.pyplot are imported.

•	It loads the glass.csv dataset into a pandas DataFrame object df and creates a bar plot of the glass type distribution using countplot() from seaborn.

•	Using pairplot() from seaborn it creates a scatter plot matrix of selected features and shows the visualizations using show() from matplotlib.pyplot.


    

Which algorithm you got better accuracy? Can you justify why?

  •	Compared to Naïve Bayes model on the Glass dataset, the linear SVM model performs better with an accuracy of 0.6769 compared to 0.3077.
  
  •	This can be attributed to:

    o	SVM's ability to handle complex feature relationships and find an optimal decision boundary.
    
    o	Naïve Bayes' assumption of feature independence, which may not hold for the Glass dataset.
    
    o	The choice of hyperparameters, train-test split, and other factors also affect model performance, but overall, linear SVM is a better choice for this dataset.


    
    
