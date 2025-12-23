import os
from openai import OpenAI
from small_tools import get_files, read_text, write_text, read_csv, make_dir
import tiktoken


# if text is short, do not need to calculate token limit
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# openai tools

OPENAI_API_KEY = ""

client = OpenAI(api_key=OPENAI_API_KEY)

max_return = 512 + 64

prompt_head = "Extract top 50 keyphrases from the following text. Each keyphrase contains 4 or less grams." \
              "After each phrase, include a numerical value representing its importance. "\
              "Rank the keyphrases by their importance, with the most important phrases listed first. " \
              "For example:" \
              "1. keyphrase - 9" \
              "2. keyphrase - 8 " \
              "3. keyphrase - 8"\
              # "The keyphrases must be exact phrases that appear in the input text. "\

# load text
for d, data_set in enumerate(['SemEval2010', 'Nguyen2007']):

    root_path = r""
    data_path = root_path + r"data/"+data_set+"/sent_token_stat/"
    model = 'gpt3'
    model_version = "gpt-3.5-turbo-0125"  # 16,385
    max_input = 16385  # 8096

    date_mode = '20250603_may_appear'

    files = get_files(data_path)

    # round 1 - 5
    for round in range(1):
        round += 1
        print('----- round', round)
        save_path = root_path + r'data\data_' + date_mode + r'/' + data_set + r'/' + model + '_round_' + str(round) + '/'
        make_dir(save_path+'extracted_keyphrase/')

        print('total files:', len(files))  # 498

        total_sent_num = 0

        for f, file in enumerate(files[:]):
            total_print = ''

            print(f, file)
            total_print += file + '\n'

            sentences_token_num_pair = read_csv(data_path + file)
            sentences = [item[0] for item in sentences_token_num_pair]

            # figure out the input length
            local_thresh = max_input - max_return - 11
            """658 15800 15800
                659 15828 15800
                15811 in the messages"""
            c = 0
            for n, num in enumerate(sentences):
                # print(n + 1, num_tokens_from_string(prompt_head + ' '.join(sentences[:n + 1])), local_thresh)
                if num_tokens_from_string(prompt_head + ' '.join(sentences[:n + 1])) > local_thresh:
                    break
                c = n + 1
            input_text = ' '.join(sentences[:c])

            if len(sentences) != c:
                print('CUT', len(sentences), c)
                total_print += ' '.join(['CUT', str(len(sentences)), str(c), '\n'])

            # response
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": 'system',  # only 4 choice: ['system', 'assistant', 'user', 'function']
                        "content": prompt_head  # "extract the keyphrases",
                    },
                    {
                        "role": 'user',  # only 4 choice: ['system', 'assistant', 'user', 'function']
                        "content":  input_text,
                    }
                ],
                model=model_version,
                temperature=0.0,
                max_tokens=max_return,
                # frequency_penalty=0.0,
                # presence_penalty=0.0,
                # stream=True,
            )

            response = response.choices[0].message.content

            # print(response, '\n')
            total_print += response + '\n\n'

            # save raw output
            write_text(total_print, root_path + r'data\data_' + date_mode + r'/' + data_set + r'/' + model + '.txt', 'a')

            # save extracted keyphrase
            w2 = write_text(response, save_path + r'extracted_keyphrase/' + file.replace('.csv', '.txt'))

