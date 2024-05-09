import re
import os


regex_punctuation = re.compile('[\',\.\-/\n]')
regex_alphanum = re.compile('[^a-zA-Z0-9 ]')
regex_num = re.compile('\d[\d ]+')
regex_spaces = re.compile('\s+')

def clean_text(text):
    text = text.lower().strip()

    # remove phi tags
    tags = re.findall('\[\*\*.*?\*\*\]', text)
    for tag in set(tags):
        text = text.replace(tag, ' ')

    text = re.sub(regex_punctuation, ' ', text)
    text = re.sub(regex_alphanum, '', text)
    text = re.sub(regex_num, ' 0 ', text)
    text = re.sub(regex_spaces, ' ', text)
    return text.strip()
    

