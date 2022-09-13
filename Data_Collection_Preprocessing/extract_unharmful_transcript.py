"""
This file contains the code to extract the unharmful conversation from the json file and create a csv file
The unharmful files come from the public dataset called 일상 대화 말뭉치 2020 (NIKL_DIALOGUE_2020_v1.2) and
available online at https://corpus.korean.go.kr/ after registration and dataset usage application.
"""
import csv
import io
import os
import re
import time
from datetime import timedelta, datetime
from itertools import zip_longest
from collections import Counter

import multiprocessing
from multiprocessing import set_start_method
from multiprocessing import Pool, Process, Queue
from functools import partial

import requests
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


def json_extract(obj, key):
    """
    Recursively fetch values from nested JSON.
    Extract nested values from a JSON tree.
    source: https://hackersandslackers.com/extract-data-from-complex-json-python/
    :param obj:
    :param key:
    :return:
    """

    arr = []

    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values


def extract_values(obj, key):
    """Recursively pull values of specified key from nested JSON."""
    arr = []

    def extract(obj, arr, key):
        """Return all matching values in an object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    if k == key:
                        arr.append(v)
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    results = extract(obj, arr, key)
    to_return = []
    for result in results:
        if type(result) == list:
            for item in result:
                to_return.append(item)
        else:
            to_return.append(result)
    return to_return


"""function using multiprocessing and filename only as input data"""
def json_to_csv(json_file):
    """
    Extract the needed data from a json and return a dataframe or save into CSV file
    :param json_file:
    :return:
    """
    # define the list of the keywords to search
    keywords_to_search = ['대출', '자금', '금리', '채무', '범죄', '수사', '검찰', '경찰', '은행', '계좌', '통장', '신용',
                          '카드', '납치', '예금', '번호', '조사', '개인', '정보', '금융', '돈', '경제', '이체']
    keywords_in_text = []

    file = open(json_file, encoding="utf8")  # Opening JSON file
    data = json.load(file)  # returns JSON object as a dictionary
    row_list = []

    # db_columns = ['Topic', 'Spelling_transcript', 'Pronunciation_transcript', 'ST_length', 'File_name']
    # df = pd.DataFrame(columns=db_columns)
    #
    # """extract the data from the json using the defined columns names"""
    # df['Topic'] = pd.Series(extract_values(data, 'topic'), dtype=pd.StringDtype())
    # df['Spelling_transcript'] = ' '.join(extract_values(data, 'form'))
    # df['Pronunciation_transcript'] = ' '.join(extract_values(data, 'original_form'))
    # df['ST_length'] = len(extract_values(data, 'form'))
    # df['File_name'] = json_file
    #
    # csv_path = 'csv/NIKL_DIALOGUE_2020_v1.2_' + datetime.now().strftime("%Y_%m_%d") + '.csv'
    # dataframe_to_csv(df, csv_path)

    """save the results in a list to create the csv file"""
    topic = ''.join(extract_values(data, 'topic'))
    spelling_transcript = ' '.join(extract_values(data, 'form'))
    # clean the transcript from useless words.
    regex = r"({.*?})|(\(.*?\))|(&.*?&)|(name\d?)"
    subst = ""
    cleansed_spelling_transcript = re.sub(regex, subst, spelling_transcript, 0, re.MULTILINE)

    pronunciation_transcript = ' '.join(extract_values(data, 'original_form'))

    for keyword in keywords_to_search:
        if keyword in spelling_transcript:
            keywords_in_text.append(keyword)

    row = [topic, cleansed_spelling_transcript, pronunciation_transcript, len(cleansed_spelling_transcript), ', '.join(keywords_in_text), json_file]
    row_list.append(row)
    return row_list


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


def pool_handler(json_file_list):
    """
    Run in multiprocessing the json data extraction
    :param json_file_list: list containing the path to the json files
    """
    nb_cpus = multiprocessing.cpu_count()
    p = Pool(processes=nb_cpus - 2)
    print('>>> pool_handler - start')
    start = time.time()
    joined_rows = p.imap_unordered(json_to_csv, tqdm(json_file_list, total=len(json_file_list)))

    # open file and write out all rows from incoming lists of rows
    with open(r'raw_transcripts/non_vishing/NIKL_DIALOGUE_2020_v1.2__' + datetime.now().strftime("%Y_%m_%d") + '.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Topic', 'Spelling_transcript', 'Pronunciation_transcript', 'ST_length', 'keywords', 'File_name'])
        for row_list in joined_rows:
            if row_list:
                writer.writerows(row_list)

    delta_t = time.time() - start
    print("Processing Time :", str(timedelta(seconds=delta_t)))  # provide time in hour
    print('>>> pool_handler - end')
    p.close()
    p.join()


if __name__ == '__main__':
    save_path = "E:\\DATASET\\국립국어원 일상 대화 말뭉치 2020(버전 1.2)"
    json_file_list = []

    for (root, dirs, files) in os.walk(save_path):
        """count the number of file inside the folder and perform extract"""
        for transcript_file in files:
            json_file_path = os.path.join(root, transcript_file)
            json_file_list.append(json_file_path)  # create a list of the json path
    print('List of {} json files entries created.'.format(len(json_file_list)))
    pool_handler(json_file_list)
