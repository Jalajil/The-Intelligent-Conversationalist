import os
from natsort import natsorted

listOfFiles=os.listdir("C:\\Users\\admin\\Documents\\IslamInterestTracker1\\en\\Accepted_Islam")
listOfFiles1=os.listdir("C:\\Users\\admin\Documents\IslamInterestTracker1\\en\\Interested_in_Islam")
listOfFiles2=os.listdir("C:\\Users\\admin\\Documents\\IslamInterestTracker1\\en\\Wants_to_Convert")

listOfFiles = natsorted(listOfFiles)
listOfFiles1 = natsorted(listOfFiles1)
listOfFiles2 = natsorted(listOfFiles2)


def rename(folder,type,num=1):
    for file in folder:
        os.rename(f'C:\\Users\\admin\\Documents\\IslamInterestTracker1\\en\\{type}\\{file}',f'C:\\Users\\admin\\Documents\\IslamInterestTracker1\\en\\{type}\\{type}{num}.json')
        num+=1

rename(listOfFiles,'Accepted_Islam')
rename(listOfFiles1,'Interested_in_Islam')
rename(listOfFiles2,'Wants_to_Convert')

