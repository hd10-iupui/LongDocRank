from small_tools import get_files, read_text, write_text, make_dir, remove_space, remove_punc_and_space, remove_head_tail_space
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce

import re

# stemming tool
ps = PorterStemmer()

# load the stopwords
stopwords = read_text(r'UGIR_stopwords.txt')
stopwords = stopwords.split('\n')


def split_acronym(input_string):
    # Splitting the string using regex to handle the pattern
    import re

    # Extracting the main text and the abbreviation
    match = re.match(r'^(.*) \((.*)\)$', input_string)
    if match:
        part1 = match.group(1)
        part2 = match.group(2)
        return [part1, part2, input_string]
    else:
        return [input_string]


def remove_cite(text_):
    # print(text_)
    left = text_.find('[')
    if text_[left:].find(']') != -1 and text_[left:].replace(',', '').replace(' ', '').replace('[', '').replace(']',
                                                                                                                '').isnumeric():
        text_ = text_[:left]
    return remove_head_tail_space(text_)


# load text
root_path = r""
for llm in ['gpt3']:  # , 'llama3', 'gpt4o'
    print(llm)
    for data in ['SemEval2010', 'Nguyen2007']:
        print(data)
        date = '20250603_may_appear'

        data_path = root_path + r'data\data_'+date+r'/'+data+r'/'+llm+'_round_1\extracted_keyphrase/'
        files = get_files(data_path)

        # save path
        save_path = data_path.replace('extracted_keyphrase', 'extracted_keyphrase_processed' + '/')
        make_dir(save_path)

        # save path -- stemmed
        save_path1 = save_path.replace('extracted_keyphrase_processed', 'stem_extracted_keyphrase_processed_no_acronym_list')
        make_dir(save_path1)

        # process
        for file in files[:]:

            # print(file)

            text = read_text(data_path + file)

            # pre-processing
            text = text.replace('**', '')
            text = text.replace('importance: ', '')
            text = text.replace('Top 50 keyphrases:\n', '')
            text = text.replace('top 50 keyphrases with their importance values:\n', '')

            text = text.split('\n')
            text = [item0 for item0 in text if item0]
            text = [item1 for item1 in text if item1.find('top 50 keyphrase') == -1]
            text = [item1 for item1 in text if item1.find('following keyphrases') == -1]
            text = [item1 for item1 in text if item1.find('importance values') == -1]
            text = [item1 for item1 in text if item1.find('importance scores') == -1]
            text = [item1 for item1 in text if item1[:9] != 'Here are ']
            text = [item1 for item1 in text if item1[:10] != 'Note that ']

            # some results come back within one line
            if len(text) == 1:
                text = text[0].split(' - ')

            if len(text) == 1:  # some results come back within one line
                processed_candidates = text[0].split(', ')
            else:
                # remove line number
                processed_candidates = []
                for item in text:
                    # print(item)
                    # some number ex.: '1.' or '[1]' or '-'
                    line_number = item.split()[0].replace('.', '').replace('[', '').replace(']', '')
                    if line_number.isnumeric() or line_number == '-':
                        item2 = item[item.find(' ') + 1:]
                    else:
                        item2 = item
                    processed_candidates.append(item2)

            # print(processed_candidates[0])
            # print(processed_candidates[-1])

            # remove importance score
            processed_candidates2 = []
            for item2 in processed_candidates:

                if len(item2.split(' - ')) == 2 \
                        and (
                        item2.split(' - ')[1].replace('(', '').replace(')', '').replace('.', '').replace('/', '').isnumeric() or
                        item2.split(' - ')[1] == ''):
                    # float cannot use isnumeric, so need to remove the period
                    processed_candidates2.append(item2.split(' - ')[0])
                elif len(item2.split(' - ')) == 2 and (
                item2.split(' - ')[0].split()[-1].replace('(', '').replace(')', '').replace('.', '').replace('/',
                                                                                                             '').isnumeric()):
                    item3 = item2.split(' - ')[0].split(' ')[:-1]
                    item3 = ' '.join(item3)
                    processed_candidates2.append(item3)
                elif item2.split(' ')[-1].replace('(', '').replace(')', '').replace('.', '').replace('/', '').isnumeric():
                    item4 = item2.split(' ')[:-1]
                    item4 = ' '.join(item4)
                    processed_candidates2.append(item4)
                else:
                    # print(item2)
                    while item2[-1] == '-' or item2[-1] == ' ':
                        item2 = item2[:-1]
                        # print(item2)
                    processed_candidates2.append(item2)

            # print(processed_candidates2)

            # post-processing
            # remove cite
            processed_candidates2 = [remove_cite(item0) for item0 in processed_candidates2]
            processed_candidates2 = [item0 for item0 in processed_candidates2 if item0]
            processed_candidates2 = [item for item in processed_candidates2 if len(item) > 1]
            processed_candidates2 = [item for item in processed_candidates2 if item not in stopwords]
            processed_candidates2 = [item for item in processed_candidates2 if  # keep those non-numeric
                                     item.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace('.',
                                                                                                                      '').isnumeric() == False]

            # split acronym
            processed_candidates22 = []
            for item in processed_candidates2:
                item = split_acronym(item)
                processed_candidates22.append(item)

            # remove the duplicate
            processed_candidates31 = []
            processed_candidates32 = []
            for item in processed_candidates22:
                if item not in processed_candidates31:
                    # print(item)
                    processed_candidates31.append(item)
                    processed_candidates32.append('[' + ', '.join(item) + ']')

            # print(processed_candidates31[0])
            # print(processed_candidates31[-1])

            # write the origin
            write_text('\n'.join(processed_candidates32), save_path + file)

            # so stem directly
            stemmed_candidates0 = []
            for w_list in processed_candidates31:
                stemmed_w_list = []
                for w in w_list:
                    w = w.lower()
                    w = remove_punc_and_space(w)
                    words = word_tokenize(w)
                    # using reduce to apply stemmer to each word and join them back into a string
                    stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")  # print(stemmed_sentence)
                    stemmed_sentence = remove_head_tail_space(stemmed_sentence)
                    stemmed_sentence = stemmed_sentence.replace('  ', ' ').replace('  ', ' ') \
                        .replace('  ', ' ').replace('  ', ' ')
                    stemmed_w_list.append(stemmed_sentence)
                stemmed_candidates0.append(stemmed_w_list)

            stemmed_candidates1 = []
            stemmed_candidates2 = []
            for item in stemmed_candidates0:
                if item not in stemmed_candidates1:
                    stemmed_candidates1.append(item)
                    stemmed_candidates2.append('[' + ', '.join(item) + ']')

            w = write_text('\n'.join(stemmed_candidates2), save_path1 + file)
