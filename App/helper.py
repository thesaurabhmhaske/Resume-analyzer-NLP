import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text

def calculate_similarity(job_description, resume):
    job_description = clean_text(job_description)
    resume = clean_text(resume)

    text = [resume, job_description]

    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)
    match_percent = cosine_similarity(count_matrix)[0][1] * 100
    match_percent = round(match_percent, 2)
    return match_percent
