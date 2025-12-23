import os
from small_tools import get_files, read_text, write_text, csv_writer, make_dir
import tiktoken
import spacy


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# NLP Tools
nlp = spacy.load("en_core_web_sm")

# load text
root_path = r""

for d, data_set in enumerate(['SemEval2010', 'Nguyen2007']):

    data_path = root_path + r"data/" + data_set + "/docsutf8/"
    save_path = data_path.replace('docsutf8', 'sent_token_stat')
    make_dir(save_path)
    files = get_files(data_path)

    total_sent_num = 0
    for file in files[:]:

        print(file)

        text = read_text(data_path + file)
        doc = nlp(text.replace('\n', ' '))  # .replace('ABSTRACT', '. ABSTRACT'))
        sentences = [sent.text for sent in doc.sents]

        w = csv_writer(save_path + file.replace('.txt', '.csv'), 'w')

        for sent in sentences:
            n = num_tokens_from_string(sent)
            # print(sent, n)
            w.writerow([sent, n])
