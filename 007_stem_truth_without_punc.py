from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from small_tools import read_text, write_text, get_files, remove_head_tail_space, make_dir, read_csv, csv_writer, remove_punc_and_space
from functools import reduce

ps = PorterStemmer()

root_path = r""
data_set = 'SemEval2010'

truth_path = root_path + 'data/'+data_set+'/keys/'
save_path3 = truth_path.replace('keys', 'stemmed_keys_no_punc')
make_dir(save_path3)
files3 = get_files(truth_path)

for file in files3[:]:
    print('stage 3', file)
    keys = read_text(truth_path + file)
    keys = keys.split('\n')
    keys = [remove_head_tail_space(item) for item in keys if item]
    keys = [remove_punc_and_space(item) for item in keys if item]

    stemmed_keys = []
    for w in keys:
        words = word_tokenize(w)
        # using reduce to apply stemmer to each word and join them back into a string
        stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")

        stemmed_sentence = remove_head_tail_space(stemmed_sentence)
        stemmed_sentence = stemmed_sentence.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        stemmed_keys.append(stemmed_sentence)

    w = write_text('\n'.join(stemmed_keys), save_path3 + file)
