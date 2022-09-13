# Python implementation to compute
# number of characters, words, spaces
# and lines in a file

# Function to count number
# of characters, words, spaces
# and lines in a file
import csv
import multiprocessing
import os
import shutil
import time
from datetime import timedelta, datetime
from multiprocessing.pool import Pool
from shutil import copyfile

import pandas as pd
from tqdm import tqdm

plus = 0


def counter(file_name):
    """
    Count the number of characters in a file
    :param fname:
    """
    # file with more that 250 characters
    global plus
    # variable to store total word count
    num_words = 0
    # variable to store total line count
    num_lines = 0
    # variable to store total character count
    num_charc = 0
    # variable to store total space count
    num_spaces = 0

    # opening file using with() method
    # so that file gets closed
    # after completion of work
    with open(file_name, 'r', encoding='cp949') as f:
        # loop to iterate file
        # line by line
        for line in f:
            # incrementing value of
            # num_lines with each
            # iteration of loop to
            # store total line count
            num_lines += 1

            # declaring a variable word
            # and assigning its value as Y
            # because every file is
            # supposed to start with
            # a word or a character
            word = 'Y'

            # loop to iterate every
            # line letter by letter
            for letter in line:
                # condition to check
                # that the encountered character
                # is not white space and a word
                if (letter != ' ' and word == 'Y'):

                    # incrementing the word
                    # count by 1
                    num_words += 1

                    # assigning value N to
                    # variable word because until
                    # space will not encounter
                    # a word can not be completed
                    word = 'N'

                # condition to check
                # that the encountered character
                # is a white space
                elif (letter == ' '):
                    # incrementing the space
                    # count by 1
                    num_spaces += 1

                    # assigning value Y to
                    # variable word because after
                    # white space a word
                    # is supposed to occur
                    word = 'Y'

                # loop to iterate every
                # letter character by
                # character
                for i in letter:
                    # condition to check
                    # that the encountered character
                    # is not white space and not
                    # a newline character
                    if (i != " " and i != "\n"):
                        # incrementing character
                        # count by 1
                        num_charc += 1

    print('Number of characters in text file: ', num_charc)

    # if num_charc >= 330:
    #     plus += 1
    #     print('total = ', plus)
    #
    #     print("File >> ", fname)
    #
    #     # # printing total word count
    #     # print("Number of words in text file: ", num_words)
    #     #
    #     # # printing total line count
    #     # print("Number of lines in text file: ", num_lines)
    #     #
    #     # printing total character count
    #     print('Number of characters in text file: ', num_charc)
    #
    #     # # printing total space count
    #     # print('Number of spaces in text file: ', num_spaces)
    #     #
    #     # os.remove(fname)
    #     print('------------------------------------------------')


def dataframe_to_csv(df, csv_path):
    """
    save the dataframe to the csv path given
    :param df: input dataframe
    :param csv_path: path to teh csv file
    :return:
    """
    header_list = True

    # check if the csv file exist to create a new one with the header
    if os.path.isfile(csv_path):
        header_list = False  # Set csv header to false
    # Save data frame to csv
    df.to_csv(csv_path, mode='a', header=header_list, encoding="utf-8", index=False)


def count_characters(file_name, encoding='cp949'):
    """
    Return the number of characters in a file
    :param encoding:
    :param file_name:
    :return:
    """
    # open file in read mode
    file = open(file_name, "r", encoding=encoding)
    # list of rows
    row_list = []

    # read the content of file
    data = file.read()
    # print(data)
    # get the length of the data
    number_of_characters = len(data)
    # print('Nb. Characters :', number_of_characters)
    # dic = {'filename': [file_name], 'filelenght': [number_of_characters]}
    # csv_path = 'E:\\DATASET\\AIHUB\\한국어음성\\KsponSpeech_lenght_' + datetime.now().strftime("%Y_%m_%d") + '.csv'

    # """save the results in a csv file"""
    # df = [file_name, number_of_characters]
    # row_list.append(df)
    # df = pd.DataFrame(dic)
    # dataframe_to_csv(df, csv_path)  # add the file's length to the dataframe
    return number_of_characters


def filter_unharmful_file(file_name):
    """
    Filter the files according to the keyword predefined
    :return:
    """
    selected_folder = "E:\\DATASET\AIHUB\\한국어음성\\selected_KsponSpeech"
    selected_folder_final = "E:\\DATASET\AIHUB\\한국어음성\\selected_KsponSpeech_final"
    not_selected = "E:\\DATASET\AIHUB\\한국어음성\\not_selected"
    not_selected_final = "E:\\DATASET\AIHUB\\한국어음성\\not_selected_final"
    row_list = []
    file_category = ""

    # define the list of the keywords to search
    keywords_to_search = ['대출', '자금', '금리', '채무', '범죄', '수사', '검찰', '경찰', '은행', '계좌', '통장', '신용',
                          '카드', '납치', '예금', '번호', '조사', '개인', '정보', '금융', '돈', '경제', '이체']
    keywords_in_text = []

    # open file in read mode
    file = open(file_name, "r", encoding='cp949')
    # read the content of file
    data = file.read()

    nb_char = count_characters(file_name)
    if nb_char >= 200:
        for keyword in keywords_to_search:
            if keyword in data:
                keywords_in_text.append(keyword)
        if keywords_in_text:
            file_category = "selected+200"
            # copying the files to the selected_folder_final directory
            shutil.copy2(file_name, selected_folder_final)
        else:
            file_category = "not_selected+200"
            # copying the files to the not_selected directory
            shutil.copy2(file_name, not_selected_final)

        # else:
        #     if nb_char >= 200:
        #         file_category = "not_selected+200"
        #         # copying the files to the not_selected directory
        #         shutil.copy2(file_name, not_selected_final)
        # else:
        #     file_category = "not_selected"
        #     # copying the files to the not_selected directory
        #     shutil.copy2(file_name, not_selected)
        """save the results in a list to create the csv file"""
        row = [file_name, file_category, nb_char, ', '.join(keywords_in_text)]
        row_list.append(row)
        return row_list


def count_fss_lenght(file_name):
    """
    count the number of character a voice phishing document has an return the value
    :return:
    """
    row_list = []
    nb_char = count_characters(file_name, "UTF8")

    """save the results in a list to create the csv file"""
    row = [file_name, nb_char]
    row_list.append(row)
    return row_list


def pool_handler(file_list):
    """
    Run in multiprocessing the text file
    :param csv_path:
    :param file_list: list containing the path to the json files
    """
    # nb = n = max(1, cpu_count() - 1)
    nb_cpus = multiprocessing.cpu_count()
    p = Pool(processes=nb_cpus - 2)
    print('>>> pool_handler - start')
    start = time.time()
    joined_rows = p.imap_unordered(count_fss_lenght, tqdm(file_list, total=len(file_list)))

    # open file and write out all rows from incoming lists of rows
    with open(r'E:\DATASET\FSS\DATASET\All transcripts_new\\FSS_final_lenght_' + datetime.now().strftime(
            "%Y_%m_%d") + '.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "file_length"])
        for row_list in joined_rows:
            if row_list:
                writer.writerows(row_list)

    # # p.map(count_characters, tqdm(file_list, total=len(file_list)))
    # for df in p.imap_unordered(count_characters, tqdm(file_list, total=len(file_list))):
    #     # print(df)
    #     # dataframe_to_csv(df, csv_path)  # add the file's length to the dataframe
    #     df.to_csv(csv_path, mode='w', encoding="utf-8")

    delta_t = time.time() - start
    print("Processing Time :", str(timedelta(seconds=delta_t)))  # provide time in hour
    p.close()
    p.join()


if __name__ == '__main__':
    # tfolder = "E:\\DATASET\\AIHUB\\한국어음성\\All"
    tfolder = "E:\\DATASET\\FSS\\DATASET\\All transcripts_new\\by_folder"
    file_list = []

    # for transcribe in os.listdir(tfolder):
    #     try:
    #         transcribefile = os.path.join(tfolder, transcribe)
    #         counter(transcribefile)
    #     except:
    #         print('File not found')

    for subdir, dirs, files in tqdm(os.walk(tfolder)):
        for file in files:
            file_path = os.path.join(subdir, file)
            file_list.append(file_path)

    print('List of {} text files entries created.'.format(len(file_list)))
    pool_handler(file_list)
