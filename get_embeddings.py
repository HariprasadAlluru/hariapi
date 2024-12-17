import os
import re
import boto3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json

class EmbeddingGenerator:
    def __init__(self, bucket_name, courses_file_key, region_name='ap-south-1'):
        self.bucket_name = bucket_name
        self.courses_file_key = courses_file_key
        self.region_name = region_name
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.last_processed_file = '/tmp/last_processed_courses.json'

    def _get_s3_file_last_modified(self):
        """Get the last modified timestamp of the S3 file."""
        response = self.s3_client.head_object(Bucket=self.bucket_name, Key=self.courses_file_key)
        return response['LastModified']

    def _load_last_processed_timestamp(self):
        """Load the last processed timestamp from a local JSON file."""
        try:
            with open(self.last_processed_file, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data.get('last_processed_timestamp', '1970-01-01'))
        except (FileNotFoundError, json.JSONDecodeError):
            return datetime.min

    def _save_last_processed_timestamp(self, timestamp):
        """Save the current processing timestamp to a local JSON file."""
        with open(self.last_processed_file, 'w') as f:
            json.dump({
                'last_processed_timestamp': timestamp.isoformat(),
                'processed_file_key': self.courses_file_key
            }, f)

    def generate_embeddings(self):
        # Timestamp for current run
        current_run_timestamp = datetime.now()

        # Get S3 file last modified time
        s3_file_last_modified = self._get_s3_file_last_modified()
        
        # Get last processed timestamp
        last_processed_timestamp = self._load_last_processed_timestamp()

        # Check if S3 file is newer than last processed time
        if s3_file_last_modified <= last_processed_timestamp:
            print("No new file to process. Exiting.")
            return False

        # Timestamp for versioning
        timestamp = current_run_timestamp.strftime("%Y%m%d_%H%M%S")

        try:
            # Download CSV file from S3
            print("Downloading course data...")
            local_input_path = f"/tmp/courses_data_{timestamp}.csv"
            self.s3_client.download_file(self.bucket_name, self.courses_file_key, local_input_path)

            # Load CSV into DataFrame
            courses_full_df = pd.read_csv(local_input_path, low_memory=False)
            print("Courses data loaded.")

            # Initialize model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Define course columns
            course_columns = [
                'course_course_id', 'course_title', 'course_description', 
                'course_product_line', 'course_product', 'course_type'
            ]

            # Filter and clean data
            courses_df = courses_full_df[course_columns]
            courses_df = courses_df.dropna(subset=course_columns)

            # Remove rows with 'n/a' or similar values
            courses_df = courses_df[~courses_df[course_columns].apply(
                lambda row: row.isin(["n/a", "N/A", "N/a", "n/A"]).any(), 
                axis=1
            )]

            # Remove HTML tags from description
            html_tag_pattern = re.compile(r'<.*?>')
            courses_df = courses_df[~courses_df['course_description'].str.contains(html_tag_pattern, na=False)]

            # Combine text for embedding
            courses_df['combined_text'] = courses_df['course_title'] + " - " + courses_df['course_description']

            # Generate embeddings
            print("Generating embeddings...")
            courses_df['embeddings'] = courses_df['combined_text'].apply(lambda x: model.encode(x).tolist())

            # Create embeddings matrix
            embeddings_matrix = np.vstack(courses_df['embeddings'].values)

            # Paths for local files with timestamp
            embeddings_matrix_path = f'/tmp/embeddings_matrix_{timestamp}.npy'
            courses_csv_path = f'/tmp/courses_with_embeddings_{timestamp}.csv'

            # Save embeddings matrix
            np.save(embeddings_matrix_path, embeddings_matrix)

            # Prepare and save CSV (without embeddings column)
            courses_df_to_save = courses_df.drop(columns=['embeddings'])
            courses_df_to_save.to_csv(courses_csv_path, index=False)

            # Upload files back to S3
            print("Uploading files to S3...")
            embeddings_file_key = f"embeddings/embeddings_matrix_{timestamp}.npy"
            courses_file_key_new = f"processed/courses_with_embeddings_{timestamp}.csv"

            self.s3_client.upload_file(
                embeddings_matrix_path, 
                self.bucket_name, 
                embeddings_file_key
            )
            self.s3_client.upload_file(
                courses_csv_path, 
                self.bucket_name, 
                courses_file_key_new
            )

            # Save the current timestamp as last processed
            self._save_last_processed_timestamp(current_run_timestamp)

            print("Embedding generation completed successfully!")
            return True

        except Exception as e:
            print(f"Error in embedding generation: {e}")
            return False

def main():
    # S3 bucket and file details
    bucket_name = "getembeddingsone"
    courses_file_key = "courses_data_shh.csv"
    
    # Create generator instance
    generator = EmbeddingGenerator(bucket_name, courses_file_key)
    
    # Run embedding generation
    generator.generate_embeddings()

if __name__ == "__main__":
    main()
