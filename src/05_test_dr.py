import os
from datetime import datetime
import torch
from torchvision.transforms import Resize, Normalize
import numpy as np
from dataset import Dataset_dr
from utils.utils import resnet18, targets
from models import EmbeddingNetwork, FreezeEmbeddingNetwork
from tqdm import tqdm
from utils.utils_triplet_loss import dist_between_embs, cosine_distance
from sklearn import metrics as skl_metrics
from utils import metrics
import numpy as np
import argparse
import config as C

TABLE_SEP = ","

parser = argparse.ArgumentParser(
                    prog='Test_dr',
                    description='Test model for document retrieval')

parser.add_argument('-d_path', '--data_path', type=str, default=C.ICDAR_DATASET02_PATH)
parser.add_argument('-o', '--output_save', type=str, default='results/dr/')
parser.add_argument('-m','--models', nargs='+', help='List of model to test', required=True)
parser.add_argument('-tt', '--target_type', type=str, default="document", choices=["document", "author"])
parser.add_argument('-r_h', '--resize_height', type=int, default=246)
parser.add_argument('-r_w', '--resize_width', type=int, default=616)
parser.add_argument('-bs', '--batch_size', type=int, default=32)

args = parser.parse_args()


data_path = args.data_path
output_save_base = args.output_save
selected_models = args.models
target_type = args.target_type
resize_height = args.resize_height
resize_width = args.resize_width
batch_size = args.batch_size

k = 10 
num_workers = 1
pin_memory = True
#dist_fn = torch.nn.PairwiseDistance(p=2)
dist_fn = cosine_distance

epoch_to_test = []
for model in selected_models:
    epoch_to_test.append(int(os.path.basename(model).split(".")[0].split("_")[-1]))

#targets = ['α','β','γ','δ','ε','ζ','η','θ','ι','κ','λ','μ','ν','ξ','ο','π','ρ','σ','τ','υ','φ','χ','ψ','ω']
#targets = ['α','ε','ι','ν','ο','σ','τ','ρ','γ','η']
#targets = ['α','ε','ι','ν','ο']
#targets = ['α','ε','ο']
targets = ['ε']

data_mean = C.ICDAR_DATA_MEAN
data_std = C.ICDAR_DATA_STD




def get_space(model, dataloader):
    model.eval()

    log_interval = int(len(dataloader)/C.TQDM_DISP_FR)

    true_lab_dict = {}

    classes = None
    all_embs = None
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(dataloader), desc='Buld space'):
            img_tensor, author_id, papyrus_id, img_path = data

            if target_type == "document":
                target = papyrus_id if len(papyrus_id) > 0 else None
            else:
                target = author_id if len(author_id) > 0 else None

            for img_p, tar in zip(img_path, target):
                true_lab_dict[img_p.split(os.path.sep)[-2].replace("_train", "")] = tar.item()

            img_tensor = img_tensor.to(device)
            
            embs = model(img_tensor) 

            if all_embs is None:
                all_embs = embs.to('cpu')
            else:
                all_embs =  torch.cat((all_embs, embs.to('cpu')), 0)

            if classes is None:
                classes = target
            else:
                classes =  torch.cat((classes, target), 0)
                    
            message = 'Buld space: [{:.0f}/{} ({:.0f}%)]'.format(
                    batch_idx * len(data[0]), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader))

            if batch_idx % log_interval == 0:
                tqdm.write(message)
            
    return (classes, all_embs), true_lab_dict


def get_list_imgs(model, dataloader, training_space, k):
    model.eval()

    log_interval = int(len(dataloader)/C.TQDM_DISP_FR)

    with torch.no_grad():
        whole_target_actual = []
        whole_neighbors = []
        res_dict = {}
        
        for batch_idx, data in tqdm(enumerate(dataloader), desc=f'Test k {k}'):
            img_tensor, author_id, papyrus_id, img_path = data
            
            if target_type == "document":
                target = papyrus_id if len(papyrus_id) > 0 else None
            else:
                target = author_id if len(author_id) > 0 else None

            img_tensor = img_tensor.to(device)
            
            embs = model(img_tensor)

            dist = get_distance_matrix(training_space, embs.to('cpu'), dist_fn)
            dist = dist.numpy()
            sorted_indx = np.argsort(dist, axis=0)

            all_neighbors = []
            for i in range(sorted_indx.shape[1]):
                neighbors_dist=[] 
                neighbors = training_space[0][sorted_indx[:k,i]]
                for ind in sorted_indx[:k,i]:
                    neighbors_dist.append(dist[ind][i])
                
                all_neighbors.append(np.array(neighbors))
            
            whole_target_actual.extend(target.numpy())
            whole_neighbors.extend(all_neighbors)

            for ind, path in enumerate(img_path):
                res_dict[path] = {"TrueLabel":target[ind].item(),"PredictedLabel":all_neighbors[ind][0]}

            message = 'Test k {}: [{:.0f}/{} ({:.0f}%)]'.format(
                        k, batch_idx * len(data[0]), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader))

            if batch_idx+1 % log_interval == 0:
                tqdm.write(message)

    whole_target_actual = np.array(whole_target_actual)
    whole_neighbors = np.array(whole_neighbors)
           
    return res_dict, whole_target_actual, whole_neighbors


def get_distance_matrix(space, embs, dist_fn):
        
    dist = torch.zeros(len(space[1]), len(embs))

    for i in range(len(space[1])):
        for j in range(len(embs)):
            dist[i][j] = dist_between_embs((space[1][i], embs[j]), dist_fn)
    return dist


def res_table_builder(folder_path, out_filename="table_result.txt", cif_round=4):

    if os.path.isfile(os.path.join(folder_path,out_filename)):
        os.remove(os.path.join(folder_path,out_filename))

    res_prec = {} 
    res_rec = {} 
    res_f1 = {} 
    res_top1a = {}
    res_top2a = {}
    res_top3a = {}
    res_top5a = {}
    res_top10a = {}
    res_map1 = {}
    res_map10 = {}

    for res_folder in os.listdir(folder_path):
        curr_epoch = int(res_folder.split("_")[-2])

        for res_filename in os.listdir(os.path.join(folder_path,res_folder)):
            curr_k = int(res_filename.split(".")[0].split("_")[-1].replace("k",""))
                
            with open(os.path.join(folder_path,res_folder,res_filename), "r") as res_file:
                lines = res_file.readlines()

                res_prec[curr_epoch] = round(float(lines[4].split(" ")[1]), cif_round)
                res_rec[curr_epoch] = round(float(lines[5].split(" ")[1]), cif_round)
                res_f1[curr_epoch] = round(float(lines[6].split(" ")[1]), cif_round)
                res_top1a[curr_epoch] = round(float(lines[7].split(" ")[1]), cif_round)
                res_top2a[curr_epoch] = round(float(lines[8].split(" ")[1]), cif_round)
                res_top3a[curr_epoch] = round(float(lines[9].split(" ")[1]), cif_round)
                res_top5a[curr_epoch] = round(float(lines[10].split(" ")[1]), cif_round)
                res_top10a[curr_epoch] = round(float(lines[11].split(" ")[1]), cif_round)
                res_map1[curr_epoch] = round(float(lines[12].split(" ")[1]), cif_round)
                res_map10[curr_epoch] = round(float(lines[13].split(" ")[1]), cif_round)

        
    res_prec = dict(sorted(res_prec.items()))
    res_rec = dict(sorted(res_rec.items()))
    res_f1 = dict(sorted(res_f1.items()))
    res_top1a = dict(sorted(res_top1a.items()))
    res_top2a = dict(sorted(res_top2a.items()))
    res_top3a = dict(sorted(res_top3a.items()))
    res_top5a = dict(sorted(res_top5a.items()))
    res_top10a = dict(sorted(res_top10a.items()))
    res_map1 = dict(sorted(res_map1.items()))
    res_map10 = dict(sorted(res_map10.items()))

    with open(os.path.join(folder_path,out_filename), "w") as table_file:
        table_file.write("Results:\n\n")

        table_file.write(f"Epochs{TABLE_SEP}Precision{TABLE_SEP}Recall{TABLE_SEP}F1\n")
        for epoch, prec in res_prec.items():
            if epoch in epoch_to_test:
                line = f"{epoch}{TABLE_SEP}{prec}{TABLE_SEP}{res_rec[epoch]}{TABLE_SEP}{res_f1[epoch]}\n"
                table_file.write(line)

        table_file.write("\n\nAccuracy:\n")
        table_file.write(f"Epochs{TABLE_SEP}Top1-A{TABLE_SEP}Top2-2{TABLE_SEP}Top3-A{TABLE_SEP}Top5-A{TABLE_SEP}Top10-A\n")
        for epoch, t1a in res_top1a.items():
            if epoch in epoch_to_test:
                line = f"{epoch}{TABLE_SEP}{t1a}{TABLE_SEP}{res_top2a[epoch]}{TABLE_SEP}{res_top3a[epoch]}{TABLE_SEP}{res_top5a[epoch]}{TABLE_SEP}{res_top10a[epoch]}\n"
                table_file.write(line)
        
        table_file.write("\n\nmean Avarage Precision (mAP):\n")
        table_file.write(f"Epochs{TABLE_SEP}mAP_1{TABLE_SEP}mAP_10\n")
        for epoch, map_1 in res_map1.items():
            if epoch in epoch_to_test:
                line = f"{epoch}{TABLE_SEP}{map_1}{TABLE_SEP}{res_map10[epoch]}\n"
                table_file.write(line)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")


# DATA ---------------------------------------------------------------------------------------------------------
#transforms = [Resize((resize_height, resize_width))]
transforms = [
    Resize((resize_height, resize_width)), 
    Normalize(mean=data_mean, std=data_std)
    ]

training_set = Dataset_dr(basepath=data_path, transformer=transforms, label=target_type)
test_set = Dataset_dr(basepath=data_path, transformer=transforms, dtype='test', label=target_type)

kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, **kwargs)

print(f"Len Training:{len(training_set)}   len test:{len(test_set)}")

# MODEL -------------------------------------------------------------------------------------------------------
backbone = resnet18()
model_pre = EmbeddingNetwork(backbone, targets, 512, batch_size, device)
model = FreezeEmbeddingNetwork(model_pre, 512, freeze=False)



for selected_model in selected_models:
    output_save = os.path.join(output_save_base, f'{start_time}_{os.path.splitext(os.path.basename(selected_model))[0]}_test')
    os.makedirs(output_save)

    if selected_model != None:
        checkpoint = torch.load(selected_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model Loaded: {selected_model}")

    model.to(device)


    ## Test -----------------------------------------------------------------------------------------------------

    # build training reppresentation
    training_space = get_space(model, train_loader) 

    res_dict, targets_actual, neighbors = get_list_imgs(model, test_loader, training_space[0], k=k)
    
    target_class = neighbors[:,0]

    prec = skl_metrics.precision_score(y_true=targets_actual, y_pred=target_class, average='weighted') #micro
    rec = skl_metrics.recall_score(y_true=targets_actual, y_pred=target_class, average='weighted') #micro

    F1 = 2 * (prec * rec) / (prec + rec)

    t1_accuracy = metrics.topAccuracy(actual=[[x] for x in targets_actual],predicted=neighbors, k=1)
    t2_accuracy = metrics.topAccuracy(actual=[[x] for x in targets_actual],predicted=neighbors, k=2)
    t3_accuracy = metrics.topAccuracy(actual=[[x] for x in targets_actual],predicted=neighbors, k=3)
    t5_accuracy = metrics.topAccuracy(actual=[[x] for x in targets_actual],predicted=neighbors, k=5)
    t10_accuracy = metrics.topAccuracy(actual=[[x] for x in targets_actual],predicted=neighbors, k=10)

    map_1 = metrics.mapk(actual=[[x] for x in targets_actual],predicted=neighbors, k=1)
    map_10 = metrics.mapk(actual=[[x] for x in targets_actual],predicted=neighbors, k=10)

    print(f"Results k {k}: pecision:{prec}, recal:{rec}, F1:{F1}\n\nTop1-Accuracy: {t1_accuracy}\nTop2-Accuracy: {t2_accuracy}\nTop3-Accuracy: {t3_accuracy}\nTop5-Accuracy: {t5_accuracy}\nTop10-Accuracy: {t10_accuracy}\n\nmAP: {map_10}\n")

    with open(os.path.join(output_save, f"results_k{k}.txt"), "w") as out_file:
            out_file.write(f"Results\n")
            out_file.write(f"Model: {selected_model}\n")
            out_file.write(f"Len Retrieved List k:{k}\n")
            out_file.write(f"Distance Function: {dist_fn}\n")
            out_file.write(f"Precision: {prec}\n")
            out_file.write(f"Recal: {rec}\n")
            out_file.write(f"F1: {F1}\n")    
            out_file.write(f"Top1-Accuracy: {t1_accuracy}\n")    
            out_file.write(f"Top2-Accuracy: {t2_accuracy}\n")    
            out_file.write(f"Top3-Accuracy: {t3_accuracy}\n")    
            out_file.write(f"Top5-Accuracy: {t5_accuracy}\n")    
            out_file.write(f"Top10-Accuracy: {t10_accuracy}\n")    
            out_file.write(f"mAP_K1: {map_1}\n")    
            out_file.write(f"mAP_K10: {map_10}\n")    
    
    res_table_builder(output_save_base)
