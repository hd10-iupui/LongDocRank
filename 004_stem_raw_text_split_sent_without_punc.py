from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from small_tools import read_text, write_text, get_files, remove_head_tail_space, make_dir, read_csv, csv_writer, remove_punc_and_space
from functools import reduce

ps = PorterStemmer()

root_path = r""
data_set = 'SemEval2010'
text_path = root_path + 'data/'+data_set+'/docsutf8/'
sent_path = root_path + r'data/'+data_set+'/sent_token_stat/'

save_path1 = text_path.replace('docsutf8', 'stemmed_docsutf8_no_punc')
make_dir(save_path1)

files1 = get_files(text_path)

for file in files1[:]:
    print('stage 1', file)
    text = read_text(text_path + file)
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
    text = remove_punc_and_space(text)

    words = word_tokenize(text)

    # using reduce to apply stemmer to each word and join them back into a string
    stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")

    stemmed_sentence = remove_head_tail_space(stemmed_sentence)
    stemmed_sentence = stemmed_sentence.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')

    w = write_text(stemmed_sentence, save_path1 + file)


save_path2 = sent_path.replace('sent_token_stat', 'stemmed_docsutf8_by_sent_no_punc')
make_dir(save_path2)

files2 = get_files(sent_path)

for file in files2[:]:

    print('stage 2', file)
    w = csv_writer(save_path2 + file, 'w')

    sentences_token_num_pair = read_csv(sent_path + file)
    sentences = [item for item in sentences_token_num_pair]

    for sent, token_num in sentences:

        sent = remove_punc_and_space(sent)
        words = word_tokenize(sent)

        # using reduce to apply stemmer to each word and join them back into a string
        stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")

        stemmed_sentence = remove_head_tail_space(stemmed_sentence)
        stemmed_sentence = stemmed_sentence.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')

        w.writerow([stemmed_sentence, token_num])
