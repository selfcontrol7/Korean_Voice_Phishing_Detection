import glob
import os

path = glob.glob('dataset_transcripts/phishing')

for i in path:
    name = i.split('/')[1]
    file2 = open('C:/AAA/phishing_cleaned/' + name, 'a')
    file1 = open(i, 'r')
    print("path", path)
    exit()
    Lines = file1.readlines()

    count = 0
    # Strips the newline character
    for line in Lines:
        L = line.strip().split(':')
        print("length", len(L))
        print("length", (L))
        if len(L) == 0 or len(L) == 1:
            print("List is empty")

            print("Line{}: {}".format(count, L))
            file2.writelines('\n')
        else:
            L = line.strip().split(':')[1]
            print("Line{}: {}".format(count, L))
            # writing to file
            # print("length",len(L))
            file2.writelines(L)

file1.close()
