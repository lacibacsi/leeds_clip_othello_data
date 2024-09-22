import datetime
import shutil

from handling import common
from os.path import join

class BookParser():
    """
    Class for parsing, cleaning othello books and prepare them to be used in fine-tuning the llm
    """

    # words to remove before passing in to instruction creation.
    # Not due to copyright violation, but due to having particular - unrelated - words repeated on each page
    # or topics / sentences not related to the game at all
    WORDS_TO_CLEAN = ['copyright', 'Trademarks', '©', 'http', 'www']

    def __init__(self, input_path: str = None, output_path: str = None, bookfile_name: str = None):
        self.input_path = common.DEFAULT_BOOK_PATH if input_path is None else input_path
        self.output_path = common.DEFAULT_OUTPUT_PATH if output_path is None else output_path
        self.book_output_file = common.DEFAULT_BOOK_FILE if bookfile_name is None else bookfile_name

    @property
    def clean_files(self):
        """
        Reads and cleans the text (all) files in the directory.
        Cleaning includes removing too short lines (1-2 characters) and lines with particular words
        :return: name of files saved in output path
        """

        files = common.get_files_from_directory(self.input_path)
        print(f'read {len(files)} files')

        counter = 0
        output_file_names = []

        for file in files:
            # read file content, parse line by line -> TODO: change it to some proper algo
            with open(join(self.input_path, file)) as f:

                temp_output = file + '_parsed' + str(datetime.datetime.now()) + '.txt'
                output_file_names.append(temp_output)
                counter += 1
                print(f'reading file {counter} of {len(files)}')

                with open(join(self.output_path, temp_output), 'w') as fout:

                    for line in f:
                        # checking for length > 0 and < 4 -> those are likely to be noise or parsing error
                        # not removing line breaks though
                        # checking for words -> if found, remove the lines
                        if 0 < len(line) < 5 and str(line) != '\n':
                            continue

                        words = set(line.split(' '))
                        if len(words.intersection(set(self.WORDS_TO_CLEAN))) > 0:
                            continue

                        # write output
                        fout.write(line)

        return output_file_names

    def generate_instructions(self, filenames: []):
        """
        Takes the input files, combines them, generates and saves a question - answer list for instruction tuning
        :param filenames:
        :return: filename of the saved question - answer pairs
        """

        book_path = join(self.output_path, self.book_output_file)

        with open(book_path, 'wb') as f:
            for tempfile in filenames:
                with open(join(self.output_path, tempfile), 'rb') as fd:
                    shutil.copyfileobj(fd, f)

        # for question-answer generation use:
        # https://colab.research.google.com/drive/1XEI4v6kO3EqM6nOgNsK256sHuUWEaOCG#scrollTo=9q0p4lJN-Iht


        # ideas from https://towardsdatascience.com/how-to-generate-instruction-datasets-from-any-documents-for-llm-fine-tuning-abb319a05d91
        # and https://colab.research.google.com/drive/1QEpYZQ0fK22EdB05zFcCUSHO3eRVgQCs?usp=sharing
        # some inspiration from https://dgallitelli95.medium.com/serving-fish-for-dinner-using-boni!pto-v1-on-amazon-sagemaker-to-generate-datasets-for-llm-d8340dee2e85
        # some from https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing#scrollTo=vITh0KVJ10qX

        #dataset = Dataset.from_text(book_path)
        #print(dataset)

        # Initialize the Bonito model
        '''
        bonito = Bonito("BatsResearch/bonito-v1")
        sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=1)
        synthetic_dataset = bonito.generate_tasks(
            dataset,
            context_col="text",
            task_type="qg",
            sampling_params=sampling_params
        )
        '''



