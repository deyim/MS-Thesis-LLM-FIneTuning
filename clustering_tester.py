from transformers import BertTokenizer
import numpy as np

import argparse
import csv
import logging
import os
import random
import sys

# Evaluation
from sklearn.metrics import f1_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

# def vectorize_documents(corpus)
#     tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
#     features = []
#     for idx, document in enumerate(corpus):
#         tokens = tokenizer.tokenize(document) 
#         avg_vector = np.average(tokens) #check data
#         features.append((idx, avg_vector))
  

def clustering(args, corpus):
    centroids = initialize_centroids()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    results = []

    for iteration in range(args.max_iter):
        random_docs = features.random(args.mini_batch_size)
        d = None
        for b in range(args.mini_batch_size):
            tokens = tokenizer.tokenize(random_docs[b]) 
            avg_vector = np.average(tokens) #check data
        
        # s, p = clustering model(d, initial centroids)
        # p, y <- 1 - NN(s, kc)
        # update parameters in clustering model using NLL
    
    return results
        
    
    
def evaluation(results, corpus):

    # F-measure
    f1 = f1_score(results, corpus, average='micro')

    # ARI measure
    ARI = adjusted_rand_score()


    # AMI measure
    AMI = adjusted_mutual_info_score()


def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--cluster_num",
                        default=5,
                        type=int,
                        required=True,
                        help="The number of clusters")
    parser.add_argument("--max_iter",
                        default=10,
                        type=int,
                        required=True,
                        help="The number of clusters")
    parser.add_argument("--bert_model")
    parser.add_argument("--clustering_model")
    parser.add_argument("--mini_batch_size",
                        default=5)
    parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")

    retuers = nltk.#
    # features = vectorize_documents(args, corpus)
    cluster_results = clustering(args, corpus)
    eval_results = evaluation(cluster_results)


    args = parser.parse_args()


if __name__ == "__main__":
    main()