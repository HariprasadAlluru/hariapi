from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pandas as pd
 
# Create a FastAPI app instance
app = FastAPI()
 
# Define the shared API key for authentication (this key will be same for all users)
SHARED_API_KEY = "your_shared_api_key"

# HTTPBearer is used to get the token from the Authorization header
security = HTTPBearer()
 
# Function to validate the API key from the Bearer token
def api_key_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SHARED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
 
# Load course data
courses_df = pd.read_csv("interactions.csv", low_memory=False)
interactions_df = pd.read_csv("courses_data_shh.csv", low_memory=False)
 
# API endpoints
@app.get("/api/trending")
def findAllTrendingCourses(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validate the API key
    api_key_auth(credentials)
   
    completed_courses_df = interactions_df[interactions_df['status'] == 'Complete']
    completed_counts = completed_courses_df.groupby('course_title').size().reset_index(name='completions')
    courses_with_completions = pd.merge(courses_df, completed_counts, on='course_title', how='left')
    sorted_courses = courses_with_completions.sort_values(by='completions', ascending=False)
 
    sorted_courses_dict = sorted_courses.head(10).to_dict(orient="records")
    course_ids = [course['course_course_id'] for course in sorted_courses_dict]
 
    return {"Result": course_ids}
 
@app.get("/api/latest")
def findLatestCourses(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validate the API key
    api_key_auth(credentials)
   
    courses_df['course_created_on'] = pd.to_datetime(courses_df['course_created_on'])
    latest_courses_df = courses_df.sort_values(by=['course_created_on', 'course_published'], ascending=False)
    latest_courses_dict = latest_courses_df.head(10).to_dict(orient="records")
 
    course_ids = [course['course_course_id'] for course in latest_courses_dict]
    return {"Result": course_ids}






