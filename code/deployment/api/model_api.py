from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list
    
# def preprocess(features):
#     transformed_data, _ = stats.boxcox(test_df["crime_rate"])
#     test_df["crime_rate"] = transformed_data
#     train_df['Average'] = train_df[['dist1', 'dist2', 'dist3', 'dist4']].mean(axis=1)
#     (OneHotEncoder(drop='first'), categorical_columns)
#     StandardScaler()
#     train_df.drop(['price'], axis=1)

@app.post('/predict')
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    return {'prediction': prediction[0]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=6000)
