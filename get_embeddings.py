import pandas as pd
import numpy as np
import boto3
from sentence_transformers import SentenceTransformer

# S3 bucket and file details
bucket_name = "getembeddingsone"
courses_file_key = "courses_data_shh.csv"

aws_access_key_id = 'None'
aws_secret_access_key = 'None'
region_name = 'ap-south-1'

# Initialize S3 client
s3_client = boto3.client('s3',
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key,
                  region_name=region_name)

# Download CSV file from S3
print("Downloading course data...")
s3_client.download_file(bucket_name, courses_file_key, "/tmp/courses_data.csv")

# Load CSV into DataFrame
courses_df = pd.read_csv("/tmp/courses_data.csv")
print("Courses data loaded.") # Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')


## Handle Courses Data ( Pre Processing ) 
courses_full_df = pd.read_csv('/tmp/courses_data.csv',low_memory = False)
course_columns = ['course_course_id', 'course_title', 'course_description', 'course_product_line', 'course_product', 'course_type']
courses_df = courses_full_df[course_columns]

courses_df = courses_df.dropna(subset=course_columns)

courses_df = courses_df[~courses_df[course_columns].apply(lambda row: row.isin(["n/a", "N/A", "N/a", "n/A"]).any(), axis=1)]

html_tag_pattern = re.compile(r'<.*?>')

# Filter out rows with HTML tags in course_description
courses_df = courses_df[~courses_df['course_description'].str.contains(html_tag_pattern, na=False)]

courses_df['combined_text'] = courses_df['course_title'] + " - " + courses_df['course_description']

# Generate embeddings
print("Generating embeddings...")
courses_df['embeddings'] = courses_df['combined_text'].apply(lambda x: model.encode(x))
embeddings_matrix = np.vstack(courses_df['embeddings'].values)
print("Embeddings generated.") # Save embeddings locally
np.save('/tmp/embeddings_matrix.npy', embeddings_matrix)  # Save embeddings matrix
courses_df.drop(columns=['embeddings'], inplace=True)  # Drop embeddings for simplicity
courses_df.to_csv('/tmp/courses_with_embeddings.csv', index=False)  # Save course data

# Upload back to S3
embeddings_file_key = "path/to/embeddings_matrix.npy"
courses_file_key_new = "path/to/courses_with_embeddings.csv"

print("Uploading files to S3...")
s3_client.upload_file('/tmp/embeddings_matrix.npy', bucket_name, embeddings_file_key)
s3_client.upload_file('/tmp/courses_with_embeddings.csv', bucket_name, courses_file_key)
print("Files uploaded to S3.")
