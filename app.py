from flask import Flask, request, render_template
import os
import pandas as pd
from pdf2image import convert_from_path
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from PIL import Image
import time
import shutil
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Tesseract (Render sets the path via TESSDATA_PREFIX env var)
pytesseract.pytesseract.tesseract_cmd = 'tesseract'

executor = ThreadPoolExecutor(max_workers=4)

def preprocess_image(image):
    return image.convert('L')

def ocr_with_timeout(image):
    future = executor.submit(pytesseract.image_to_string, image)
    try:
        return future.result(timeout=30)
    except TimeoutError:
        logger.error('OCR timed out for a page.')
        return ''

def process_single_pdf(pdf_path, idx, total_pdfs):
    filename = os.path.basename(pdf_path)
    full_text = ''
    try:
        images = convert_from_path(pdf_path, first_page=1, last_page=2, dpi=100)
        for image in images:
            processed_image = preprocess_image(image)
            page_text = ocr_with_timeout(processed_image)
            full_text += page_text[:5000] + ' '
        return (filename, full_text)
    except Exception as e:
        logger.error(f'Error processing {filename}: {e}')
        return (filename, '')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        if not files:
            return render_template('index.html', error='No files uploaded.')

        pdf_paths = []
        for file in files:
            if file and file.filename.endswith('.pdf'):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                pdf_paths.append(file_path)

        if not pdf_paths:
            return render_template('index.html', error='No valid PDF files uploaded.')

        try:
            start_time = time.time()
            data = []
            pdf_texts = []

            total_pdfs = len(pdf_paths)
            futures = [executor.submit(process_single_pdf, pdf_path, idx + 1, total_pdfs)
                       for idx, pdf_path in enumerate(pdf_paths)]

            for future in futures:
                filename, full_text = future.result()
                pdf_texts.append((filename, full_text))
                data.append({
                    'filename': filename,
                    'duplicate_with': '',
                    'similarity_score': 0.0,
                    'malpractice': False
                })

            if len(pdf_texts) > 1:
                texts = [text for _, text in pdf_texts]
                vectorizer = TfidfVectorizer(max_features=5000)
                tfidf_matrix = vectorizer.fit_transform(texts)

                num_clusters = min(10, len(pdf_texts) // 5 + 1)
                kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)

                similarity_threshold = 0.9
                for cluster in range(num_clusters):
                    cluster_indices = [i for i, c in enumerate(clusters) if c == cluster]
                    if len(cluster_indices) < 2:
                        continue
                    cluster_matrix = tfidf_matrix[cluster_indices]
                    similarity_matrix = cosine_similarity(cluster_matrix)
                    for i in range(len(cluster_indices)):
                        for j in range(i + 1, len(cluster_indices)):
                            if similarity_matrix[i][j] >= similarity_threshold:
                                idx_i = cluster_indices[i]
                                idx_j = cluster_indices[j]
                                data[idx_i]['duplicate_with'] = pdf_texts[idx_j][0]
                                data[idx_i]['similarity_score'] = round(similarity_matrix[i][j], 2)
                                data[idx_i]['malpractice'] = True
                                data[idx_j]['duplicate_with'] = pdf_texts[idx_i][0]
                                data[idx_j]['similarity_score'] = round(similarity_matrix[i][j], 2)
                                data[idx_j]['malpractice'] = True

            total_time = time.time() - start_time
            df = pd.DataFrame(data)

            # Clean up uploads
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'])

            return render_template('results.html', tables=[df.to_html(classes='table table-striped', index=False)],
                                 titles=df.columns.values, total_time=round(total_time, 2))

        except Exception as e:
            logger.error(f'Error during PDF processing: {e}')
            return render_template('index.html', error=f'Error processing PDFs: {str(e)}')

    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    logger.error(f'404 error: {e}')
    return render_template('index.html', error='Page not found. Please try uploading again.'), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)