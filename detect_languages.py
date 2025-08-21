# from langdetect import detect
# from natsort import natsorted
# import json
# import os
#
# dict = {}
# root = "C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\latest_dataset - Copy"
# language = "en"
# #
# # ar = []
# # count = 1
# # counI = 1
# # counW = 1
# #
# # listOfFiles = os.listdir(f"{root}\\Accepted_Islam")
# # listOfFiles1 = os.listdir(f"{root}\\Interested_in_Islam")
# # listOfFiles2 = os.listdir(f"{root}\\Wants_to_Convert")
# #
# # listOfFiles = natsorted(listOfFiles)
# # listOfFiles1 = natsorted(listOfFiles1)
# # listOfFiles2 = natsorted(listOfFiles2)
# #
# # print(len(listOfFiles))
# # print(len(listOfFiles1))
# # print(len(listOfFiles2))
# #
# # while listOfFiles:
# #     with open(file=f"{root}\\Accepted_Islam\\{listOfFiles.pop(0)}", mode="r") as file:
# #         data = file.read()
# #     js = json.loads(data)
# #     text = data
# #     language = detect(text)
# #     if language not in ar:
# #         ar.append(language)
# #         dict[f"{language}"] = 1
# #     else:
# #         dict[f"{language}"] += 1
# #
# #     try:
# #         os.mkdir(f"{root}\\{language}")
# #         print("folder created")
# #     except FileExistsError:
# #         pass
# #     try:
# #         os.mkdir(f"{root}\\{language}\\Accepted_Islam")
# #         print("folder created")
# #     except FileExistsError:
# #         pass
# #
# #     with open(file=f"{root}\\{language}\\Accepted_Islam\\Accepted_Islam{count}.json", mode="w") as f:
# #         json.dump(js, f)
# #         count += 1
# #
# # while listOfFiles1:
# #     with open(file=f"{root}\\Interested_in_Islam\\{listOfFiles1.pop(0)}", mode="r") as file:
# #         data = file.read()
# #     js = json.loads(data)
# #     text = data
# #     language = detect(text)
# #     if language not in ar:
# #         ar.append(language)
# #         dict[f"{language}"] = 1
# #     else:
# #         dict[f"{language}"] += 1
# #
# #     try:
# #         os.mkdir(f"{root}\\{language}")
# #         print("folder created")
# #     except FileExistsError:
# #         pass
# #     try:
# #         os.mkdir(f"{root}\\{language}\\Interested_in_Islam")
# #         print("folder created")
# #     except FileExistsError:
# #         pass
# #
# #     with open(file=f"{root}\\{language}\\Interested_in_Islam\\Interested_in_Islam{counI}.json", mode="w") as f1:
# #         json.dump(js, f1)
# #         counI += 1
# #
# # while listOfFiles2:
# #     with open(file=f"{root}\\Wants_to_Convert\\{listOfFiles2.pop(0)}", mode="r") as file:
# #         data = file.read()
# #     js = json.loads(data)
# #     text = data
# #     language = detect(text)
# #     if language not in ar:
# #         ar.append(language)
# #         dict[f"{language}"] = 1
# #     else:
# #         dict[f"{language}"] += 1
# #
# #     try:
# #         os.mkdir(f"{root}\\{language}")
# #         print("folder created")
# #     except FileExistsError:
# #         pass
# #     try:
# #         os.mkdir(f"{root}\\{language}\\Wants_to_Convert")
# #         print("folder created")
# #     except FileExistsError:
# #         pass
# #
# #     with open(file=f"{root}\\{language}\\Wants_to_Convert\\Wants_to_Convert{counW}.json", mode="w") as f2:
# #         json.dump(js, f2)
# #         counW += 1
# #
# # for key, value in dict.items():
# #     print(f"{key}:{value}")
# #
# #     import os
# #     from natsort import natsorted
# #
# #
# # def rename_files_in_folder(folder_path, prefix):
# #     files = os.listdir(folder_path)
# #     files = natsorted(files)
# #
# #     for index, filename in enumerate(files, start=1):
# #         old_file = os.path.join(folder_path, filename)
# #         new_file = os.path.join(folder_path, f"{prefix}{index}.json")
# #         os.rename(old_file, new_file)
# #         print(f"Renamed: {old_file} to {new_file}")
# #
#
#
#
# folders = {
#     "Accepted_Islam - Copy": "Accepted_Islam",
#     "Interested_in_Islam - Copy": "Interested_in_Islam",
#     "Wants_to_Convert - Copy": "Wants_to_Convert"
# }
#
# # for folder, prefix in folders.items():
# #     folder_path = os.path.join(root, language, folder)
# #     if os.path.exists(folder_path):
# #         rename_files_in_folder(folder_path, prefix)
#
#
#
# import re
#
#
# def numerical_sort(value):
#     numbers = re.findall(r'\d+', value)
#     return int(numbers[0]) if numbers else 0
#
#
# def check_ascending_order(folder_path, prefix):
#     files = os.listdir(folder_path)
#     files = [file for file in files if file.startswith(prefix)]
#     sorted_files = sorted(files, key=numerical_sort)
#
#     for i in range(1, len(sorted_files)):
#         current_number = numerical_sort(sorted_files[i])
#         previous_number = numerical_sort(sorted_files[i - 1])
#         if current_number != previous_number + 1:
#             return False, sorted_files[i - 1], sorted_files[i]
#
#     return True, None, None
#
# for folder, prefix in folders.items():
#     folder_path = os.path.join(root, language, folder)
#     if os.path.exists(folder_path):
#         is_arranged, previous_file, current_file = check_ascending_order(folder_path, prefix)
#         if is_arranged:
#             print(f"Folder '{folder}' is fully arranged.")
#         else:
#             print(f"Folder '{folder}' is not arranged. Issue between files: {previous_file} and {current_file}")
#     else:
#         print(f"Folder '{folder}' does not exist.")

# Here should the detect language start. The above code is messy.
from langdetect import detect
from natsort import natsorted
import json
import os

dict = {}
root = "C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\latest_dataset - Copy"
language = "en"

ar = []
count = 1
counI = 1
counW = 1

listOfFiles = os.listdir(f"{root}\\Accepted_Islam")
listOfFiles1 = os.listdir(f"{root}\\Interested_in_Islam")
listOfFiles2 = os.listdir(f"{root}\\Wants_to_Convert")

listOfFiles = natsorted(listOfFiles)
listOfFiles1 = natsorted(listOfFiles1)
listOfFiles2 = natsorted(listOfFiles2)

while listOfFiles:
    with open(file=f"{root}\\Accepted_Islam\\{listOfFiles.pop(0)}", mode="r") as file:
        data = file.read()
    js = json.loads(data)
    text = data
    language = detect(text)
    if language not in ar:
        ar.append(language)
        dict[f"{language}"] = 1
    else:
        dict[f"{language}"] += 1

    try:
        os.mkdir(f"{root}\\{language}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"{root}\\{language}\\Accepted_Islam")
    except FileExistsError:
        pass

    with open(file=f"{root}\\{language}\\Accepted_Islam\\Accepted_Islam{count}.json", mode="w") as f:
        json.dump(js, f)
        count += 1

while listOfFiles1:
    with open(file=f"{root}\\Interested_in_Islam\\{listOfFiles1.pop(0)}", mode="r") as file:
        data = file.read()
    js = json.loads(data)
    text = data
    language = detect(text)
    if language not in ar:
        ar.append(language)
        dict[f"{language}"] = 1
    else:
        dict[f"{language}"] += 1

    try:
        os.mkdir(f"{root}\\{language}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"{root}\\{language}\\Interested_in_Islam")
    except FileExistsError:
        pass

    with open(file=f"{root}\\{language}\\Interested_in_Islam\\Interested_in_Islam{counI}.json", mode="w") as f1:
        json.dump(js, f1)
        counI += 1

while listOfFiles2:
    with open(file=f"{root}\\Wants_to_Convert\\{listOfFiles2.pop(0)}", mode="r") as file:
        data = file.read()
    js = json.loads(data)
    text = data
    language = detect(text)
    if language not in ar:
        ar.append(language)
        dict[f"{language}"] = 1
    else:
        dict[f"{language}"] += 1

    try:
        os.mkdir(f"{root}\\{language}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"{root}\\{language}\\Wants_to_Convert")
    except FileExistsError:
        pass

    with open(file=f"{root}\\{language}\\Wants_to_Convert\\Wants_to_Convert{counW}.json", mode="w") as f2:
        json.dump(js, f2)
        counW += 1

for key, value in dict.items():
    print(f"{key}:{value}")
