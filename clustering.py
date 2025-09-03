import os
import yaml
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score,fowlkes_mallows_score,pairwise_distances
from data_utils import Datasets
from data_utils import split, get_assets_org_id
from models.ConOA import ConOA
from models.MeOA import MeOA

def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0,1", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="WOI_a", type=str, help="which dataset to use, options: WOI_a, WOI_b")
    parser.add_argument("-e", "--embedding_model", default="ConOA", type=str, help="which embedding model to use, options: ConOA, MeOA")
    # parser.add_argument("-n", "--embedding_model_name", default="", type=str)
    args = parser.parse_args()

    return args

def pairwise_f1(true_labels, pred_labels):
    TP = FP = FN = 0
    n = len(true_labels)
    for i in range(n):
        for j in range(i + 1, n):
            same_true_label = true_labels[i] == true_labels[j]
            same_pred_label = pred_labels[i] == pred_labels[j]
            
            if same_true_label and same_pred_label:
                TP += 1
            elif not same_true_label and same_pred_label:
                FP += 1
            elif same_true_label and not same_pred_label:
                FN += 1
    
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return f1_score

def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")
    args = get_cmd()
    dataset_name = args.dataset
    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[-1]]
    else:
        conf = conf[dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = args.embedding_model
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["model"] = device
    
    embedding_model_names = ['allrm_ES_6ac_2aocd_2oc_ave_dropout_rewrite_8_768_128_1e-05_1e-07_2_1_0.1_9775_0.04_0.99']
    
    for embedding_model_name in embedding_model_names:
        #### Dataset #### 
        conf["mode"] = 'html' if 'html' in embedding_model_name else 'rewrite'
        
        with open(os.path.join(conf["model_path"], args.embedding_model, "conf", embedding_model_name), 'r', encoding='utf-8') as f:
            model_conf = json.loads(f.read())
        model_conf["device"] = device
        tokenizer = BertTokenizer.from_pretrained(model_conf["bert_path"])
        dataset = Datasets(conf, tokenizer)
        
        #### Embedding model ####
        if args.embedding_model == "ConOA":
            emb_model = ConOA(model_conf).to(device)
        elif args.embedding_model == "MeOA":
            emb_model = MeOA(model_conf).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(args.embedding_model))
        model_path = os.path.join(conf["model_path"], args.embedding_model, "model", embedding_model_name)
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(embedding_model_name))
            checkpoint = torch.load(model_path, map_location=device)
            emb_model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(embedding_model_name))
        else:
            print("=> no checkpoint found at '{}'".format(embedding_model_name))
        
        #### Get embedding ####
        test_assets_info = dataset.test_raw_data[1]
        test_emb = torch.tensor([]).to(device)
        test_text_list = [test_assets_info[key]['text'] for key in test_assets_info]
        emb_model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(test_text_list)), desc='Getting test assets\' embedding'):
                sentences = split(tokenizer, test_text_list[i], conf['max_token_len']-2)
                if len(sentences) > 40:
                    sentences = sentences[:40]
                if args.embedding_model == "ConOA":
                    embedding = emb_model.encoder_q(tokenizer(sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                    test_emb = torch.cat((test_emb, torch.mean(embedding, keepdim=True, dim=0)), 0)
                else:
                    embedding = emb_model.embedding(tokenizer(sentences, max_length=conf['max_token_len'], truncation=True, padding='max_length', return_tensors="pt").to(device))
                    test_emb = torch.cat((test_emb, torch.mean(embedding, dim=0)), 0)
            # test_emb---[test_num, emb_size]
        test_emb = test_emb.cpu().numpy()
        
        #### Get label ####
        true_label = get_assets_org_id(conf["data_path"], "test").numpy()
        
        f = open('./output/clustering/res-%s-%s.txt' % (dataset_name, embedding_model_name), 'a', encoding = 'utf-8')
        #### Clustering ####
        HAC_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8)
        result = HAC_model.fit_predict(test_emb)
        
        n_clusters_ = len(set(result)) - (1 if -1 in result else 0)
        pred_label = np.zeros_like(result)
        for i in range(n_clusters_):
            mask = (result == i)
            true_labels_in_cluster = true_label[mask]
            most_common_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
            pred_label[mask] = most_common_label

        purity = np.sum(pred_label == true_label) / len(true_label)
        print(f"Purity: {purity}")
        f.write(f"Purity: {purity}\n")

        ari = adjusted_rand_score(true_label, result)
        print(f"ARI: {ari}")
        f.write(f"ARI: {ari}\n")

        fmi = fowlkes_mallows_score(true_label, result)
        print(f"FMI: {fmi}")
        f.write(f"FMI: {fmi}\n")

        f1 = pairwise_f1(true_label, result)
        print(f"F1: {f1}")
        f.write(f"F1: {f1}\n")

        f.close()


if __name__ == "__main__":
    main()

