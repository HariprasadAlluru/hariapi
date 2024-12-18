from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import re
import faiss
from sentence_transformers import SentenceTransformer
from collections import Counter
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import boto3
 
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
#courses_df = pd.read_csv("courses_data_shh.csv", low_memory=False)
#interactions_df = pd.read_csv("interactions.csv", low_memory=False)

# S3 Bucket and File Paths
BUCKET_NAME = 'getembeddingsone'
USERS_FILE = 'users_data_shh.csv'
COURSES_FILE = 'processed/courses_with_embeddings_20241218_065839.csv'
INTERACTIONS_FILE = 'interactions.csv'
EMBEDDINGS_FILE = 'embeddings/embeddings_matrix_20241218_065839.npy'
ZIP_FILE_KEY = 'interactions.zip'
LOCAL_ZIP_FILE = '/home/ec2-user/file.zip'
LOCAL_UNZIP_DIR = '/home/ec2-user/unzipped/'


s3 = boto3.client('s3')

# Function to download the zipped file from S3
def download_file_from_s3_zip(bucket_name, file_key, local_file):
    try:
        print(f"Downloading {file_key} from S3...")
        s3.download_file(bucket_name, file_key, local_file)
        print(f"Downloaded {file_key} to {local_file}")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")

# Function to unzip the downloaded file
def unzip_file(local_zip_file, unzip_dir):
    try:
        print(f"Unzipping {local_zip_file} to {unzip_dir}...")
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        print(f"Unzipped to {unzip_dir}")
    except Exception as e:
        print(f"Error unzipping file: {e}")

print("downloading interactions zip")
download_file_from_s3_zip(BUCKET_NAME, ZIP_FILE_KEY, LOCAL_ZIP_FILE)
# Ensure the unzip directory exists
os.makedirs(LOCAL_UNZIP_DIR, exist_ok=True)
print("unzipping the interactions")
# Unzip the downloaded file
unzip_file(LOCAL_ZIP_FILE, LOCAL_UNZIP_DIR)
print("reading interactions zip")
interactions_df = pd.read_csv(os.path.join(LOCAL_UNZIP_DIR, 'yourfile.csv'))
print("interactions read done")
print(interactions_df.head(5))
def load_data_from_s3(bucket_name, file_key):
    # Download the file to a local path
    local_file = file_key.split('/')[-1]  # Extract the filename from the key
    s3.download_file(bucket_name, file_key, local_file)
    # Read into a Pandas DataFrame
    return pd.read_csv(local_file)

def load_npy_from_s3(bucket_name, file_key):    
    # Download the file locally
    local_file = file_key.split('/')[-1]  # Extract the filename
    s3.download_file(bucket_name, file_key, local_file)
    
    # Load the .npy file with NumPy
    return np.load(local_file)

print("load courses and users data")
# Load data
users_df = load_data_from_s3(BUCKET_NAME, USERS_FILE)
courses_full_df = load_data_from_s3(BUCKET_NAME, COURSES_FILE)
#interactions_df = load_data_from_s3(BUCKET_NAME, INTERACTIONS_FILE)
print("done users and courses data")
embeddings_matrix = load_npy_from_s3(BUCKET_NAME, EMBEDDINGS_FILE)
print("done embeddings matrix")
interaction_columns = ['username', 'course_id', 'start_date']
# Drop rows with NaN values in the specified columns
user_course_interaction_df = interactions_df.dropna(subset=interaction_columns)

# Drop rows with NaN values in the specified columns
user_course_interaction_df = interactions_df.dropna(subset=interaction_columns)

masked_users_interactions_df = user_course_interaction_df[['username', 'course_id']].apply(lambda row: row.str.contains(html_tag_pattern, na=False)).any(axis=1)

# Drop rows where any of the specified columns contain HTML tags
user_course_interaction_df = user_course_interaction_df[~masked_users_interactions_df]



user_course_interactions_merged = user_course_interaction_df.merge(
    courses_full_df[['course_course_id', 'course_title', 'course_description', 'combined_text', 'course_type', 'course_product_line', 'course_created_on', 'course_updated_on']],
    left_on='course_id',
    right_on='course_course_id',
    how='inner'
)

# API endpoints
@app.get("/api/trending")
def findAllTrendingCourses(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validate the API key
    api_key_auth(credentials)
   
    completed_courses_df = interactions_df[interactions_df['status'] == 'Complete']
    completed_counts = completed_courses_df.groupby('course_title').size().reset_index(name='completions')
    courses_with_completions = pd.merge(courses_full_df, completed_counts, on='course_title', how='left')
    sorted_courses = courses_with_completions.sort_values(by='completions', ascending=False)
 
    sorted_courses_dict = sorted_courses.head(10).to_dict(orient="records")
    course_ids = [course['course_course_id'] for course in sorted_courses_dict]
 
    return {"Result": course_ids}
 
@app.get("/api/latest")
def findLatestCourses(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validate the API key
    api_key_auth(credentials)
   
    courses_full_df['course_created_on'] = pd.to_datetime(courses_full_df['course_created_on'])
    latest_courses_df = courses_full_df.sort_values(by=['course_created_on', 'course_published'], ascending=False)
    latest_courses_dict = latest_courses_df.head(10).to_dict(orient="records")
 
    course_ids = [course['course_course_id'] for course in latest_courses_dict]
    return {"Result": course_ids}

# 2. Get the latest top 3 courses for a user
def get_latest_courses(username, data, top_n=10):
    user_courses = data[data['username'] == username].sort_values('start_date', ascending=False)
    return user_courses.head(top_n)


def setup_faiss_index(user_name, user_course_interactions_merged, model, embeddings_matrix):
  latest_courses = get_latest_courses(user_name, user_course_interactions_merged)
  user_courses_ids = latest_courses['course_course_id'].tolist()

  user_course_embeddings = model.encode(latest_courses['combined_text'].tolist())

  embedding_dimension = user_course_embeddings.shape[1]
  index = faiss.IndexFlatL2(embedding_dimension)
  index.add(embeddings_matrix)

  return index, user_course_embeddings, latest_courses, user_courses_ids

# Step 6: Function to find similar courses
def find_similar_courses(user_embeddings, index, user_courses, all_courses_df, k=1):
    distances, indices = index.search(user_embeddings, k)  # Find top-k nearest neighbors
    similar_courses = []
    for i, course_indices in enumerate(indices):
        course_matches = []
        course_ids = []
        course_types = []
        course_prod_lines = []
        for idx in course_indices:
            # Use iloc to safely retrieve the course details by positional index
            course_matches.append(all_courses_df.iloc[idx]['combined_text'])
            course_ids.append(all_courses_df.iloc[idx]['course_course_id'])
            course_types.append(all_courses_df.iloc[idx]['course_type'])
            course_prod_lines.append(all_courses_df.iloc[idx]['course_product_line'])

        similar_courses.append({
            "user_course": user_courses['course_title'].tolist()[i],
            "similar_courses": course_matches,
            "course_ids": course_ids,
            "course_types": course_types,
            "course_prod_lines": course_prod_lines,
            "index" : course_indices.tolist()
        })
    return similar_courses


def process_results(results, campus_plus_status, user_courses_ids):
  # Initialize the final output and duplicates list
  final_output = []
  duplicates = []
  rec_course_ids = []

  # Get the maximum length of similar_courses in any user_course to ensure we cover all items
  max_len = 10

  # Loop round-robin style through the similar_courses in each list
  for i in range(max_len):
      for course in results:
          user_course = course['user_course']
          # Check if the index i exists in the current similar_courses list
          if i < len(course['similar_courses']):
              similar_course = course['similar_courses'][i]
              course_id = course['course_ids'][i]
              course_type = course['course_types'][i]
              course_prod_line = course['course_prod_lines'][i]

              if campus_plus_status:
                # For Campus Plus users
                if course_type != 'elearning' or course_prod_line == 'CSB / CSI / SyteLine':
                    if course_id not in user_courses_ids:
                        # Add to final_output if it's unique, otherwise add to duplicates
                        if similar_course not in final_output:
                            final_output.append(similar_course)
                            rec_course_ids.append(course_id)
                        else:
                            # Log duplicate course entry
                            duplicates.append({'user_course': user_course, 'duplicate_course': similar_course})
              else:
                  # For non-Campus Plus users
                  if course_id not in user_courses_ids:
                      # Add to final_output if it's unique, otherwise add to duplicates
                      if similar_course not in final_output:
                          final_output.append(similar_course)
                          rec_course_ids.append(course_id)
                      else:
                          # Log duplicate course entry
                          duplicates.append({'user_course': user_course, 'duplicate_course': similar_course})

  return final_output, rec_course_ids, duplicates
# Output the final lists
#print("Final Output (Unique Courses):")
#print(final_output)
#print("\nDuplicates:")
#for duplicate in duplicates:
#    print(duplicate)

@app.get("/recommendations")
def get_recommendations(username: str, credentials: HTTPAuthorizationCredentials = Depends(security)):
  print(username)
  api_key_auth(credentials)
  user_data = users_df[users_df['username'] == username]
  membership_type = user_data['membership_type'].iloc[0]
  if membership_type == 'CAMPUS_PLUS':
    campus_plus_status = True
  else:
    campus_plus_status = False
  print(membership_type)
  index, user_embeddings, user_courses, user_courses_ids= setup_faiss_index(username, user_course_interactions_merged, model, embeddings_matrix)
  print(user_courses)
  results = find_similar_courses(user_embeddings, index, user_courses, courses_full_df, k=11)
  final_output, rec_course_ids, repeated_ids = process_results(results, campus_plus_status, user_courses_ids)
  return {
        "final_output": final_output,
        "recommended_course_ids": rec_course_ids,
        "repeated_course_ids": repeated_ids
    }


'''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
import os

# FastAPI app initialization
app = FastAPI(title="Course Recommendation API")

# Global resources
model = SentenceTransformer('all-minilm-l6-v2')
embeddings_matrix = np.load('./data/embeddings_matrix.npy')

# Load datasets
courses_df = pd.read_csv('./data/courses_data_shh.csv', low_memory=False)
users_df = pd.read_csv('./data/users_data_shh.csv')
interactions_df = pd.read_csv('./data/interactions.csv')

# Preprocess data (mimicking your current cleaning logic)
html_tag_pattern = re.compile(r'<.*?>')
courses_df = courses_df.dropna().query("course_course_id != 'n/a'")
courses_df['combined_text'] = courses_df['course_title'] + " - " + courses_df['course_description']
users_df = users_df.dropna().query("user_id != 'n/a'")
interactions_df = interactions_df.dropna()

# Pydantic request/response models
class RecommendationRequest(BaseModel):
    username: str

class RecommendationResponse(BaseModel):
    recommendations: list
    course_ids: list
    duplicates: list

# Helper functions
def get_latest_courses(username, interactions, top_n=10):
    user_courses = interactions[interactions['username'] == username].sort_values('start_date', ascending=False)
    return user_courses.head(top_n)

def setup_faiss_index(user_embeddings):
    embedding_dimension = user_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(embeddings_matrix)
    return index

def find_similar_courses(user_embeddings, index, all_courses_df, k=1):
    distances, indices = index.search(user_embeddings, k)
    similar_courses = []
    for course_indices in indices:
        course_matches = [all_courses_df.iloc[idx]['combined_text'] for idx in course_indices]
        course_ids = [all_courses_df.iloc[idx]['course_course_id'] for idx in course_indices]
        similar_courses.append({"matches": course_matches, "ids": course_ids})
    return similar_courses

# API Routes
@app.post("/recommend", response_model=RecommendationResponse)
def recommend_courses(request: RecommendationRequest):
    username = request.username
    user_data = users_df[users_df['username'] == username]
    if user_data.empty:
        raise HTTPException(status_code=404, detail="User not found")

    # Membership type
    membership_type = user_data['membership_type'].iloc[0]
    campus_plus_status = membership_type == 'CAMPUS_PLUS'

    # Get user's latest courses
    latest_courses = get_latest_courses(username, interactions_df)

    if latest_courses.empty:
        raise HTTPException(status_code=404, detail="No course interactions found for this user")

    # Generate embeddings for user's latest courses
    user_course_embeddings = model.encode(latest_courses['combined_text'].tolist())

    # Set up FAISS index and find recommendations
    index = setup_faiss_index(user_course_embeddings)
    results = find_similar_courses(user_course_embeddings, index, courses_df, k=10)

    # Process and filter recommendations
    final_output = []
    rec_course_ids = []
    duplicates = []
    for result in results:
        for match, course_id in zip(result['matches'], result['ids']):
            if course_id not in rec_course_ids:
                final_output.append(match)
                rec_course_ids.append(course_id)
            else:
                duplicates.append({"user_course": username, "duplicate_course": match})

    return RecommendationResponse(recommendations=final_output, course_ids=rec_course_ids, duplicates=duplicates)'''
