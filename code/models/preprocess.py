from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
import pandas as pd

def preprocess(train_df, test_df):
    # Normalizing crime_rate column since it has polynomial distribution
    transformed_data, _ = stats.boxcox(train_df["crime_rate"])
    train_df["crime_rate"] = transformed_data
    transformed_data, _ = stats.boxcox(test_df["crime_rate"])
    test_df["crime_rate"] = transformed_data
    
    # Creating new average feature and dropping old ones
    train_df['Average'] = train_df[['dist1', 'dist2', 'dist3', 'dist4']].mean(axis=1)
    train_df = train_df.drop(columns=["dist1", "dist2", "dist3", "dist4"])
    test_df['Average'] = test_df[['dist1', 'dist2', 'dist3', 'dist4']].mean(axis=1)
    test_df = test_df.drop(columns=["dist1", "dist2", "dist3", "dist4"])

    categorical_columns = ['airport', 'waterbody', 'bus_ter']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_columns),
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])
    
    X_train, y_train = train_df.drop(['price'], axis=1), train_df['price']
    X_test, y_test = test_df.drop(['price'], axis=1), test_df['price']
    
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.fit_transform(X_test)
    
    return X_train, y_train, X_test, y_test