# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, PreTrainedModel, BertModel, BertConfig
import numpy as np
from sklearn import metrics
from time import time
from sklearn.utils.linear_assignment_ import linear_assignment
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from dec.keras_dec import DeepEmbeddingClustering
from nltk.corpus import stopwords
import nltk
import string, re, csv
from sklearn.metrics import f1_score
torch.set_grad_enabled(False)

# Raw text to embeddings 
def TextsToEmbedding(texts, model, tokenizer):
  embeddings = []
  print("Converting to Embeddings...")
  i = 0
  for text in texts:
    i += 1
    if i % 1000 == 0:
      print(str(i) + ' is being completed... ')
    if len(text) > 512:
      text = text[:512]
    tokens = tokenizer.encode_plus(text, return_tensors="pt")
    embedding = model(**tokens)[0]
    embedding = np.array(embedding).reshape(-1,512).mean(0)
    embeddings.append(embedding)
  
  return np.asarray(embeddings)

def FileToEmbedding(file, model, tokenizer):
  embeddings = []
  with open(filename, 'r') as f:
    for line in f:
      if len(line) > 512:
        line = line[:512]
      tokens = tokenizer.encode_plus(text, return_tensors="pt")
      embedding = model(**tokens)[0]
      embedding = np.array(embedding).reshpae(-1,512).mean(0)
      embeddings.append(embedding)
  return np.asarray(embeddings)

def cluster_acc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
  
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind])*1.0/y_pred.size, w

def cluster_and_eval(embeddings, true_labels, model):
  print("running clustering...")
  result = model.fit_predict(embeddings)

  ARI_result = metrics.adjusted_rand_score(true_labels, result)
  NMI_result = metrics.adjusted_mutual_info_score(true_labels, result)  
  # HOMO_result = metrics.homogeneity_score(true_labels, result)
  ACC_result = cluster_acc(true_labels, result)
  F_result = f1_score(true_labels, result, average='micro')
  print("ARI_result: {:.5f}, ACC_result: {:.5f} NMI_result: {:.5f}, F_result: {:.5f}"\
        .format(ACC_result[0], ARI_result, NMI_result, F_result))
  return ACC_result[0], ARI_result, NMI_result, F_result

def clusteringEval(embeddings, target, num_clusters, mini_batch=None):
  evaluations = []
  if not mini_batch: mini_batch = 1000
  models = [
            KMeans(n_clusters=num_clusters),
            GaussianMixture(n_components=num_clusters),
            Birch(n_clusters=num_clusters)]
  model_names = ["Kmeans","Gaussian","Birch"]
  for model, name in zip(models, model_names):
    print("clustering model: ", name)
    evaluations.append(cluster_and_eval(embeddings, target, model))
    print(evaluations)
  
  model = DeepEmbeddingClustering(n_clusters=num_clusters, input_dim=512)
  model.initialize(embeddings, finetune_iters=1000, layerwise_pretrain_iters=1000)
  model.cluster(embeddings, y=target, iter_max=10000)
  result = model.y_pred
  ARI_result = metrics.adjusted_rand_score(target, result)
  NMI_result = metrics.adjusted_mutual_info_score(target, result) 
  ACC_result = cluster_acc(target, result)
  F_result = f1_score(target, result, average='micro')
  print("ARI_result: {:.5f}, ACC_result: {:.5f} NMI_result: {:.5f}, F_result: {:.5f}"\
        .format(ACC_result[0], ARI_result, NMI_result, F_result))
  evaluations.append((ACC_result[0], ARI_result, NMI_result, F_result))

  return evaluations

def one_text_preprocess(text):
  # stopwords_set = set(stopwords.words('english'))
  lines = text.split('\n')
  new_text = []
  for line in lines:
    if ':' not in line:
      new_text.append(line)
  new_text =  ' '.join(new_text)
  new_text = re.sub(r'[^a-zA-Z]+', ' ', new_text)
  new_text = new_text.split()
  if len(new_text) < 25:
    return None
  new_text = [w.lower() for w in new_text]
  new_text = ' '.join(new_text)
  return new_text

def evaluate20NewsGroup(models, names, tokenizers, evaluation_results):
  newsgroups_train = fetch_20newsgroups(subset='train')
  newsgroups_test = fetch_20newsgroups(subset='test')
  # Build Corpus
  dataset = []
  target = []
  for text, label in zip(newsgroups_train.data, newsgroups_train.target):
    text = one_text_preprocess(text)
    if text:
      dataset.append(text)
      target.append(label)
  for text, label in zip(newsgroups_test.data, newsgroups_test.target):
    text = one_text_preprocess(text)
    if text:
      dataset.append(text)
      target.append(label)
  num_clusters = 20
  dataset = np.asarray(dataset)
  target = np.asarray(target)

  print("RUNNING ON 20NewsGroup DATASET")

  for model, tokenizer, name in zip(models, tokenizers, names):
    print("\nrunning on ", name, " model")
    embeddings = TextsToEmbedding(dataset, model, tokenizer)
    np.save(root + "embeddings/" + name + "_20NEWSGROUP_full", embeddings)
    # embeddings = np.load(root + "embeddings/" + name + "_20NEWSGROUP.npy")
    evaluations = ClusteringEval(embeddings, target, num_clusters)
    evaluation_results["20_News_Group"] = evaluations

def evaluateReuters(models, names, tokenizers, evaluation_results):
  target = []
  dataset = []
  num_clusters = 6
  data_route = "test_data/reuters.csv"

  with open(root+data_route, 'r', encoding='utf-8') as f:
      rdr = csv.reader(f, delimiter=',', quotechar='"')
      for row in rdr:
        target.append(int(row[0]))
        dataset.append(row[1])

  dataset = np.asarray(dataset)
  target = np.asarray(target)

  print("RUNNING ON REUTERS DATASET")

  for model, tokenizer, name in zip(models, tokenizers, names):
    print("\nrunning on ", name, " model")
    embeddings = TextsToEmbedding(dataset, model, tokenizer)
    np.save(root + "embeddings/" + name + "_REUTERS_full", embeddings)
    # embeddings = np.load(root + "embeddings/" + name + "_REUTERS.npy", embeddings)
    evaluations = ClusteringEval(embeddings, target, num_clusters)
    evaluation_results["Reuters"] = evaluations

def evaluateAGNews(models, names, tokenizers, evaluation_results):
  data_route = 'test_data/ag_news_full_shortened.csv'

  dataset = []
  target = []
  with open(root+data_route, 'r', encoding='utf-8') as f:
      rdr = csv.reader(f, delimiter=',', quotechar='"')
      for row in rdr:
        target.append(int(row[0])-1)
        text = row[1] + row[2]
        text = re.sub(r'[^a-zA-Z]+', ' ', text)
        text = text.split()
        text = [word.lower()  for word in text]
        text = ' '.join(text)
        dataset.append(text)

  dataset = np.asarray(dataset)
  target = np.asarray(target)
  num_clusters = 4

  for model, tokenizer, name in zip(models, tokenizers, names):
    print("running on ", name)
    embeddings = TextsToEmbedding(dataset, model, tokenizer)
    np.save(root + "embeddings/" + name + "_AG_NEWS_full", embeddings)
    evaluations = ClusteringEval(embeddings, target, num_clusters)
    evaluation_results["AG_News"] = evaluations

def evaluateBBCNews(models, names, tokenizers, evaluation_results):
  data_route = 'test_data/bbc_test.csv'

  dataset = []
  target = []
  with open(root+data_route, 'r', encoding='utf-8') as f:
      rdr = csv.reader(f, delimiter=',', quotechar='"')
      for row in rdr:
        target.append(int(row[0]))
        dataset.append(row[1])

  dataset = np.asarray(dataset)
  target = np.asarray(target)
  num_clusters = 5

  for model, tokenizer, name in zip(models, tokenizers, names):
    print("running on ", namne)
    embeddings = TextsToEmbedding(dataset, model, tokenizer)
    np.save(root + "embeddings/" + name + "_BBC_NEWS_full", embeddings)
    evaluations = ClusteringEval(embeddings, target, num_clusters)
    evaluation_results["BBC_News"] = evaluations

def evaluateYahooAnswers(models, names, tokenizers, evaluation_results):
  data_route = 'test_data/yahoo_train_full_shortened.csv'

  dataset = []
  target = []
  with open(root+data_route, 'r', encoding='utf-8') as f:
    rdr = csv.reader(f, delimiter=',', quotechar='"')
    for row in rdr:
      target.append(int(row[0]))
      dataset.append(row[1])

  dataset = np.asarray(dataset)
  target = np.asarray(target)
  num_clusters = 10

  for model, tokenizer, name in zip(models, tokenizers, names):
    print("running on ", name)
    embeddings = TextsToEmbedding(dataset, model, tokenizer)
    np.save(root + "embeddings/" + name + "_YAHOO_ANSWERS", embeddings)
    evaluations = ClusteringEval(embeddings, target, num_clusters)
    evaluation_results["YAHOO_ANSWERS"] = evaluations

if __name__ == "main":

  root = '/content/drive/My Drive/defense/'

  #Model A - Pre-trained BERT
  bert_route = root + '/base/'
  bert_tokenizer = BertTokenizer.from_pretrained(bert_route + 'vocab.txt')
  bert_config = BertConfig.from_json_file(bert_route + 'bert_config.json')
  bert_model = BertModel.from_pretrained(bert_route + 'bert_model_pytorch.bin', config=bert_config)

  #Model B - RDF Pre-trained BERT
  rdf_route = root + '/rdf_bert/'
  rdf_tokenizer = BertTokenizer.from_pretrained(bert_route + 'vocab.txt')
  rdf_config = BertConfig.from_json_file(bert_route + 'bert_config.json')
  rdf_model = BertModel.from_pretrained(rdf_route + 'rdf_bert_model_pytorch.bin', config=rdf_config)

  #Model C - RDF only trained BERT
  rdf_only_route = root + '/rdf_only_bert/'
  rdf_only_tokenizer = BertTokenizer.from_pretrained(bert_route + 'vocab.txt')
  rdf_only_config = BertConfig.from_json_file(bert_route + 'bert_config.json')
  rdf_only_model = BertModel.from_pretrained(rdf_only_route + 'rdf_only_bert_model_pytorch.bin', config=rdf_only_config)


  names = ["bert_model", "rdf_bert_model", "rdf_only_model"]
  models = [bert_model, rdf_model, rdf_only_model]
  tokenizers = [bert_tokenizer, rdf_tokenizer, rdf_only_tokenizer]
  evaluation_results = {}

  # 20NewsGroup    
  evaluate20NewsGroup(models, names, tokenizers, evaluation_results)

  # REUTERS 
  evaluate20NewsGroup(models, names, tokenizers, evaluation_results)

  # AG_NEWS
  evaluateAGNews(models, names, tokenizers, evaluation_results)  

  # BBC
  evaluateBBCNews(models, names, tokenizers, evaluation_results)
  
  # YAHOO
  evaluateYahooAnswerss(models, names, tokenizers, evaluation_results)
 
