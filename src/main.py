from flask import Flask, render_template, request
from googleapiclient.discovery import build
import re
import os
from dotenv import load_dotenv
import joblib
import pandas as pd

from prepro import preprocess_text 

load_dotenv()

app = Flask(__name__, template_folder='../templates', static_folder='../static')

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def is_valid_youtube_url(url):
    return "youtube.com/watch?v=" in url or "youtu.be/" in url

def get_all_comments(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None

    try:
        while True:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

                reply_count = item['snippet']['totalReplyCount']
                if reply_count > 0:
                    replies = get_replies(youtube, item['id'])
                    comments.extend(replies)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    except Exception as e:
        print(f"Error fetching comments: {e}")

    print(f"Total comments fetched: {len(comments)}")
    return comments

def get_replies(youtube, parent_id):
    replies = []
    next_page_token = None

    try:
        while True:
            response = youtube.comments().list(
                part="snippet",
                parentId=parent_id,
                textFormat="plainText",
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            for item in response.get('items', []):
                reply = item['snippet']['textDisplay']
                replies.append(reply)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

    except Exception as e:
        print(f"Error fetching replies: {e}")

    return replies

@app.route('/', methods=['GET', 'POST'])
def index():
    comments = []
    total_comments = 0
    total_bullying = 0
    total_non_bullying = 0
    predicted_categories = []
    comment_results = []
    message = None

    if request.method == 'POST':
        video_url = request.form.get('video_url')
        api_key = os.getenv('YOUTUBE_API_KEY')

        if not is_valid_youtube_url(video_url):
            message = "URL yang dimasukkan bukan URL YouTube yang valid."
            return render_template('index.html', comments=[], total_comments=0,
                                   total_bullying=0, total_non_bullying=0,
                                   predicted_categories=[], comment_results=[],
                                   message=message)

        video_id = extract_video_id(video_url)

        if video_id:
            comments = get_all_comments(video_id, api_key)
            total_comments = len(comments)

            if not comments:
                message = "Komentar kosong pada video ini."
                return render_template('index.html', comments=[], total_comments=0,
                                       total_bullying=0, total_non_bullying=0,
                                       predicted_categories=[], comment_results=[],
                                       message=message)

            try:
                base_dir = os.path.abspath(os.path.dirname(__file__))
                vectorizer_file_path = os.path.join(base_dir, '..', 'models', 'tfidf_vectorizer.pkl')
                model_file_path = os.path.join(base_dir, '..', 'models', 'svm_model.pkl')

                if not all(os.path.isfile(path) for path in [model_file_path, vectorizer_file_path]):
                    message = "Model atau vectorizer belum tersedia."
                    return render_template('index.html', comments=comments, total_comments=total_comments,
                                           total_bullying=0, total_non_bullying=0,
                                           predicted_categories=[], comment_results=[],
                                           message=message)

                tfidf_vectorizer = joblib.load(vectorizer_file_path)
                model = joblib.load(model_file_path)

                preprocessed_comments = [preprocess_text(comment) for comment in comments]
                processed_comments = tfidf_vectorizer.transform(preprocessed_comments).toarray()

                if processed_comments.shape[0] == 0:
                    raise ValueError("Hasil TF-IDF kosong. Periksa kembali proses preprocessing atau vectorizer.")

                predicted_categories = model.predict(processed_comments)


                total_bullying = (predicted_categories == 'Bullying').sum()
                total_non_bullying = total_comments - total_bullying
                comment_results = list(zip(comments, predicted_categories))

            except Exception as e:
                message = f"Terjadi kesalahan saat memproses data: {str(e)}"

        else:
            message = "URL video tidak valid atau tidak dapat diproses."

    return render_template('index.html',
                           comments=comments,
                           total_comments=total_comments,
                           total_bullying=total_bullying,
                           total_non_bullying=total_non_bullying,
                           predicted_categories=predicted_categories,
                           comment_results=comment_results,
                           message=message)

if __name__ == "__main__":
    app.run(debug=True)
