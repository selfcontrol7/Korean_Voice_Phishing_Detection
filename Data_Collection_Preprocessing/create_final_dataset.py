"""
This file generates the final dataset CSV file by merging the genuine voice phishing data and unharmful korean
 conversation. All the raw transcripts have been preprocessed and cleansed before creating the final CSV file. The final
 CSV file  can be used to train any Machine learning or deep learning models. Further, preprocessing steps can be
 performed on the date contains in this CSV if needed.
"""

import os
import pandas as pd
import csv
import re

from tqdm import tqdm


class CreateKorCCVi:
    """
    This class get all the raw data (csv file and text file) from the raw_transcripts folder and create one final CSV
    dataset file containing the voice phishing class and the non-voice phishing class.
    """

    def __init__(self, tfolder):
        self.voice_phishing = None
        self.folder = tfolder
        self.transcribe = None
        self.csv_rowlist = []
        self.dic_data_vishing = {
            'transcript': [],
            'confidence': [],
            'label': []
        }
        self.dic_data_unharmful = {
            'transcript': [],
            'confidence': [],
            'label': []
        }

    def transcript_to_array(self, transcript_file):
        """
        This function read the whole content of the transcribe file, assign the appropriated class then append it in
        the array of rows to create the final csv dataset later.
        :return: the content of the file
        """
        # setting the parameters in case file read is a normal file
        transcript_label = 1
        encoding = "utf8"
        # print("File name >> ", transcript_file)

        if 'non_vishing' in transcript_file:
            transcript_label = 0
            df_non_vishing = pd.read_csv(transcript_file, encoding='utf8')

            list_unharmful_transcripts = df_non_vishing["Spelling_transcript"].to_list()
            for transcript in list_unharmful_transcripts:
                self.dic_data_unharmful['transcript'].append(transcript)
                self.dic_data_unharmful['confidence'].append('')
                self.dic_data_unharmful['label'].append(transcript_label)

                self.csv_rowlist.append([transcript, '', transcript_label])
        else:
            # # Open file
            # f = open(transcribefile, 'r', encoding="utf8")
            # # Feed the file text into findall(); it returns a list of all the found strings
            # strings = re.findall(r'^confidence:\s?\d\.*\d*$', f.read(), re.MULTILINE)
            # print(strings)
            # exit()

            # open the file and read the content
            with open(transcript_file, 'r', encoding=encoding) as file:
                regex_del = r"Confidence:\s?\d+\.\d*"
                regex_confidence = r"\d+\.\d{6,}"
                # confidence = None
                # self.transcribe = file.read().replace('\n', ' ')

                self.transcribe = file.read()
                confidence_value = re.findall(regex_confidence, self.transcribe, re.MULTILINE | re.IGNORECASE)
                # get the confidence value if exist and replace the corresponding line in the transcript
                if confidence_value:
                    subst = ""
                    self.transcribe = re.sub(regex_confidence, subst, self.transcribe, 0, re.MULTILINE | re.IGNORECASE)
                    # print(self.transcribe)

                    # if result:
                    #     # print(result)
                    #     self.csv_rowlist.append([result, '\n'.join(confidence), 0])
                    #
                    #     # bad_words = ['confidence']
                    #     # for line in file:
                    #     #     if not any(bad_word in line for bad_word in bad_words):
                    #     #     # self.csv_rowlist.append([self.transcribe, 0])
                    #
                    #     # self.transcribe = self.transcribe.encode(encoding='UTF-8', errors='strict')
                    #     # self.csv_rowlist.append([self.transcribe, 0])
                    #
                    #     # print("*" * 80)
                    #     # print(self.transcribe)
                    #     # print('\n'.join(confidence))
                    # string = '\n'.join(confidence)
                    # reinitialize the value of confidence
                    # confidence = re.findall(regex_confidence, string, re.MULTILINE | re.IGNORECASE)
                    # print(confidence)

                # append the transcript, confidence and transcript_label in the list
                self.csv_rowlist.append([self.transcribe.lstrip(), '', transcript_label])

                self.dic_data_vishing['transcript'].append(self.transcribe.lstrip())
                self.dic_data_vishing['confidence'].append('')
                self.dic_data_vishing['label'].append(transcript_label)

        # return self.dic_data

    def dic_pd_to_csv(self, dataset_name):
        """
        Save the dictionary pf data to the final CSV dataset file.
        :return:
        """
        df_data_vishing = pd.DataFrame.from_dict(self.dic_data_vishing)
        # print(df_data_vishing.head())
        # print(df_data_vishing.info())
        df_data_vishing.to_csv('df_data_vishing.csv')
        print('*' * 100)

        df_data_unharmful = pd.DataFrame.from_dict(self.dic_data_unharmful)
        # print(df_data_unharmful.head())
        # print(df_data_unharmful.info())
        df_data_unharmful.to_csv('df_data_unharmful.csv')
        print('#' * 100)

        df_dataset = pd.concat([df_data_unharmful, df_data_vishing])
        df_dataset.reset_index(inplace=True)
        df_dataset.rename(columns={"index": "id"}, inplace=True)
        # print(df_dataset.head())
        print(df_dataset.info())
        df_dataset.to_csv('pd_dataset.csv', encoding='utf-8', index=False)
        df_dataset.to_excel('pd_dataset.xlsx', encoding='utf-8', index=False)

    def dic_to_csv(self, dataset_name):
        """

        :return:
        """
        dataset_header = ['transcript', 'confidence', 'label']
        with open(dataset_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dataset_header)
            writer.writeheader()
            writer.writerows(self.dic_data_unharmful)

    def txt_to_df(self):
        """

        :return:
        """
        rls = self.transcript_to_array(transcript_file)

    def txt_to_csv(self, transcript_file):
        """
        read the txt file with pandas and creat the csv
        :return:
        """
        read_transcribe = pd.read_csv(transcript_file, encoding="utf8")
        print(read_transcribe, '\n')
        # read_transcribe.to_csv(os.path.basename(transcript_file) + '.csv', index=None, encoding="utf8")

    def txt_to_csv1(self, dataset_name):
        """
        read the txt file with pandas and creat the csv
        :return:
        """
        df_transcripts = pd.DataFrame(self.csv_rowlist, columns=['transcript', 'confidence', 'label'])
        df_transcripts.reset_index(inplace=True)
        df_transcripts.rename(columns={"index": "id"}, inplace=True)
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(df_transcripts)

        # df_transcripts = df_transcripts.replace('\n', ' ', regex=True)
        df_transcripts.to_csv('list_dataset.csv', encoding='utf-8', index=False)
        # return self.voice_phishing.csv

    def txt_to_csv2(self):
        """
        read the txt file with pandas and creat the csv
        :return:
        """
        col_row = ['Transcript', 'Label']

        with open('voice_phishing.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(col_row)
            writer.writerows(self.csv_rowlist)

    def count_char(self, transcribefile):
        """
        Count the number of character in the transcript file
        :return: Number of characters
        """
        nb_lines = 0
        nb_words = 0
        nb_char = 0
        print("File >> ", transcribefile)

        # open and read the content of the file
        with open(transcribefile, 'r', encoding="utf8") as file:
            # ts = file.read()

            for line in file:
                line = line.strip("\n")  # wont count \n as a character

                words = line.split()
                nb_lines += 1
                nb_words += len(words)
                nb_char += len(line)

        print("lines:", nb_lines, "words:", nb_words, "characters:", nb_char)
        print("*"*80)


if __name__ == "__main__":
    all_transcripts_path = "cleansed_transcripts"
    transcript_file = ""
    dataset_name = "KorCCVi_v2.2.csv"

    c = CreateKorCCVi(all_transcripts_path)
    # # read all files in main folders
    # for transcribe in os.listdir(tfolder):
    #     transcribefile = os.path.join(tfolder, transcribe)
    #     print(transcribefile)
    #
    #     # c.read_file(transcribefile)
    #     # c.txt_to_csv1()
    #     # c.txt_to_csv2()

    # read all files and files in sub-folders
    for subdir, dirs, files in os.walk(all_transcripts_path):
        for file in tqdm(files):
            transcript_file = os.path.join(subdir, file)
            # print(transcript_file)
            c.transcript_to_array(transcript_file)
    # print(c.dic_data)
    c.dic_pd_to_csv(dataset_name)
    c.txt_to_csv1(dataset_name)
    # c.txt_to_csv2()

    # c.count_char(transcribefile)
    # print(c.csv_rowlist)
