"""
Clean the voice phishing and the unharmful (non_voice phishing) transcript files by removing the speakers'
identifications ('피해자 :', '사기범 :' ) and all the transcriptional and De-identification, respectively. Thereafter,
saving the cleaned transcript in a new file in the appropriated folder (cleansed_transcripts/vishing,
cleansed_transcripts/non_vishing).
"""

import os

import pandas as pd
from tqdm import tqdm


def clean_transcripts(raw_transcripts_path):
    """
    Clean the file according to the type and save the new file in appropriated folder
    """
    for subdir, dirs, files in os.walk(raw_transcripts_path):
        for file in tqdm(files, desc='Cleaning transcripts...'):
            full_file_path = os.path.join(subdir, file)
            file_name = os.path.basename(full_file_path)
            file_name_without_extension = os.path.splitext(file_name)[0]

            # check which type of file is being preprocessed top perform teh accurate preprocessing.
            if 'non_vishing' in full_file_path:
                cleansed_file = 'cleansed_transcripts/non_vishing/' + file_name_without_extension + ".csv"
                # read the csv file with pandas and remove all the useless characters
                df_non_vishing = pd.read_csv(full_file_path, encoding='utf8')
                df_non_vishing.replace({'Spelling_transcript': r'({.*?})|(\(.*?\))|(&.*?&)|(name\d?)'},
                                       {'Spelling_transcript': ''},
                                       regex=True, inplace=True)
                df_non_vishing.replace({'Spelling_transcript': r'({.*?})|(\(.*?\))|(&.*?&)|(name\d?)'},
                                       {'Spelling_transcript': ''},
                                       regex=True, inplace=True)
                # print(df_non_vishing['Spelling_transcript'].head(10))
                df_non_vishing.to_csv(cleansed_file)
            else:
                cleansed_file = open('cleansed_transcripts/vishing/' + file_name_without_extension + ".txt", 'a', encoding='utf-8')
                with open(full_file_path, 'r', encoding='utf8') as transcript:
                    Lines = transcript.readlines()
                    # print(Lines)
                    count = 0
                    # Strips the newline character
                    for line in Lines:
                        L = line.strip().split(':')
                        # print("Original line>", line)
                        # print("length", len(L))
                        # print("Line>", (L))
                        if len(L) == 0 or len(L) == 1:
                            # print("List is empty")
                            # print("Line{}: {}".format(count, L))
                            cleansed_file.writelines(L)
                            # cleansed_file.writelines('\n')
                        else:
                            L = line.strip().split(':')[1]
                            L = L.lstrip()
                            # print("Line{}: {}".format(count, L))
                            # writing to file
                            # print("length",len(L))
                            cleansed_file.writelines(L)
                            cleansed_file.writelines('\n')
                cleansed_file.close()


# def preprocess_non_vishing(non_phishing_path):
#     """
#     Clean the unharmful transcript (non_voice phishing) files by removing all the transcriptional and De-identification
#     symbols, then saving the cleaned transcript in a new file in folder cleansed_transcripts/non_vishing
#     """
#     print(non_phishing_path)
#
#     for subdir, dirs, files in os.walk(non_phishing_path):
#         for file in tqdm(files, desc='Cleaning vishing transcripts...'):
#             raw_file = os.path.join(subdir, file)
#             file_name = os.path.basename(raw_file)
#             file_name = os.path.splitext(file_name)[0]
#
#             cleansed_file = open('cleansed_transcripts/non_vishing/' + file_name + ".txt", 'a', encoding='utf-8')
#
#             with open(raw_file, 'r', encoding='utf8') as transcript:
#                 Lines = transcript.readlines()
#                 # print(Lines)
#                 count = 0
#                 # Strips the newline character
#                 for line in Lines:
#                     L = line.strip().split(':')
#                     # print("Original line>", line)
#                     # print("length", len(L))
#                     # print("Line>", (L))
#                     if len(L) == 0 or len(L) == 1:
#                         # print("List is empty")
#                         # print("Line{}: {}".format(count, L))
#                         cleansed_file.writelines(L)
#                         # cleansed_file.writelines('\n')
#                     else:
#                         L = line.strip().split(':')[1]
#                         L = L.lstrip()
#                         # print("Line{}: {}".format(count, L))
#                         # writing to file
#                         # print("length",len(L))
#                         cleansed_file.writelines(L)
#                         cleansed_file.writelines('\n')
#             cleansed_file.close()


if __name__ == "__main__":
    raw_transcripts_path = 'raw_transcripts'
    clean_transcripts(raw_transcripts_path)

    # path = 'cleaning_test'
    # for (root, dirs, files) in os.walk(path):
    #     print(root)
    #     print(dirs)
    #     print(files)
    #     print('--------------------------------')
