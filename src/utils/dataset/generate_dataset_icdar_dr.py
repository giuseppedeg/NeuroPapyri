import os
import shutil
import csv
from tqdm import tqdm
import pickle
import config as C

ICDAR_Dataset = os.path.join(C.ICDAR_IMAGES_PATH, "homer2")
icdar_dataset_ci = C.ICDAR_DATASET01_PATH
icdar_dataset_dr = C.ICDAR_DATASET02_PATH


id_documents = {}
last_id_doc = 0

id_writer = {}
last_id_writ = 0


#letters = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']
#letters = ['α','ε','ι','ν','ο','σ','τ','ρ','γ','η']
#letters = ['α','ε','ι','ν','ο']
#letters = ['α','ε','ο']
letters = ['ε']

letters_position = {
    'α':0,
    'β':1,
    'γ':2,
    'δ':3,
    'ε':4,
    'ζ':5,
    'η':6,
    'θ':7,
    'ι':8,
    'κ':9,
    'λ':10,
    'μ':11,
    'ν':12,
    'ξ':13,
    'ο':14,
    'π':15,
    'ρ':16,
    'σ':17,
    'τ':18,
    'υ':19,
    'φ':20,
    'χ':21,
    'ψ':22,
    'ω':23
}

valide_letter_position = []
for letter in letters:
    valide_letter_position.append(letters_position[letter])


def read_csv_file(csv_path, type="train"):
    global last_id_doc
    global last_id_writ
    global id_documents
    global id_writer

    #os.mkdir(os.path.join(icdar_dataset_dr, type))

    lines_towrite = set()

    all_targets = {}

    with open(csv_path) as train_file:
        train_csv = csv.reader(train_file, delimiter=",")

        with open(os.path.join(icdar_dataset_dr,type+".csv"), "w") as newtrain_file:
            for id_row, row in tqdm(enumerate(train_csv), desc=type):
                if id_row == 0:
                    newtrain_file.write("author,p_id,auth_id,dir,target\n")
                else:
                    img = row[0]
                    filename = os.path.splitext(os.path.basename(row[-1]))[0]

                    if filename not in id_documents:
                        id_documents[filename] = last_id_doc
                        last_id_doc += 1

                    if filename not in id_writer:
                        id_writer[filename] = last_id_writ
                        last_id_writ += 1


                    if not os.path.exists(os.path.join(icdar_dataset_dr, filename+"_"+type)):
                        os.mkdir(os.path.join(icdar_dataset_dr, filename+"_"+type))
                    
                    shutil.copy(src=os.path.join(icdar_dataset_ci, type, img),
                                dst=os.path.join(icdar_dataset_dr, filename+"_"+type, img))
                    
                    curr_target = row[1]
                    curr_target = curr_target.replace(" ", "")
                    curr_target = curr_target.replace("[", "")
                    curr_target = curr_target.replace("]", "")
                    curr_target = curr_target.split(',')
                    curr_target = list(map(int, curr_target))

                    target = []
                    for lett_pos in valide_letter_position:
                        target.append(curr_target[lett_pos])

                    #lines_towrite.add(f"{filename},{id_documents[filename]},{id_documents[filename]},{filename+'_'+type},{target}\n")
                    lines_towrite.add(f"{filename},{id_documents[filename]},{id_documents[filename]},{filename+'_'+type}\n")
                    


                    all_targets[os.path.join(filename+'_'+type,row[0])] = {'target':target, 'transcript':row[2]}


            for line in lines_towrite:
                newtrain_file.write(line)

    with open(os.path.join(icdar_dataset_dr,type+"_target"), "wb") as target_file:
        pickle.dump(all_targets, target_file)





if os.path.exists(icdar_dataset_dr):
    shutil.rmtree(icdar_dataset_dr)
os.makedirs(icdar_dataset_dr)


for fragment_folder in os.listdir(ICDAR_Dataset):
    for img_name in os.listdir(os.path.join(ICDAR_Dataset, fragment_folder)):
        filename = os.path.splitext(img_name)[0]
        id_documents[filename] = last_id_doc
    last_id_doc += 1

read_csv_file(os.path.join(icdar_dataset_ci,"train/datas.csv"))
read_csv_file(os.path.join(icdar_dataset_ci,"test/datas.csv"), type="test")

    

           
