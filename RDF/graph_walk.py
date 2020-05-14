import numpy as np
import random
import copy
import re
from graph import *
from tqdm import tqdm
instances = set()

def process_entity(s):
  if '^^' in s:
    s = s.split('^^')[0]
  s = s[1:-1]
  s = s.split('/')[-1]
  if "Category:" in s:
    s = s[len('Category:'):]    
  s = re.sub('[^0-9a-zA-Z]+', ' ', s).lower()
  s.strip()
  if re.match("^[0-9 ]+$", string=s):
    s = ''
  return s

def process_predicate(s):
  if 'rdf-schema#seeAlso' in s:
    return ''
  if 'www.w3.org' in s:
    return ''
  if 'http://purl.org/dc/terms/subject' in s:
    return ''
  s = s[1:-1]
  s = s.split('/')[-1]
  s = re.sub('[^0-9a-zA-Z]+', ' ', s)
  s = re.sub("([a-z])([A-Z])","\g<1> \g<2>", s)
  s = s.lower()
  return s

def merge_kg(kg, filename):
  with open(filename, 'r') as f:
    for line_ in f:
      line = line_.split()
      s, p, o = line[0], line[1], line[2]
      s = process_entity(s)
      o = process_entity(o)
      p = process_predicate(p)
      if s == '': continue    
      s_v = None
      o_v = None
      if s:
        s_v = kg.add_vertex(s)
      if o:
        o_v = kg.add_vertex(o)
      kg.add_edge(s_v, p, o_v)
      # kg.add_edge(o_v, p, s_v)
  if None in kg._vertices:
    kg._vertices.remove(None)
  return kg

def BFS(kg, output_file):
  vertices = kg._vertices.copy()
  root = random.choice(tuple(vertices))
  visited = set()
  q = [root]
  f = open(output_file, "a+")
  print("start node num: ", len(vertices))
  prob = 0
  s = 0
  n = 0
  while q:
    node = q.pop(0)
    if node in vertices:
      vertices.remove(node)
    else:
      prob += 1
      if not q and vertices:
        new_root = random.choice(tuple(vertices))
        q.append(new_root)
      continue
    visited.add(node)
    n += 1
    neighbors = kg.get_neighbors(node)
    for o,p in neighbors:
      sentence = node.name
      if p != None:
        sentence += ' ' + p.name
      if o != None:
        sentence += ' ' + o.name
      if sentence != node.name:
        s += 1
        f.write(sentence)
        f.write('\n')
        
      if o and o not in visited: 
        q.append(o)
    
    if not q and vertices:
      new_root = random.choice(tuple(vertices))
      q.append(new_root)
      
  f.close()
  print("Error: ", prob, "sentences: ", s, "node: ", n, "left vertices: ", len(vertices), q)
  return

kg = None
for file in files:
  print(file)
  start = time.time()
  kg = KnowledgeGraph()
  merge_kg(kg, file)
  print(len(kg._vertices))
  BFS(kg, "./output_" + file + ".txt")
  print(time.time() - start)

from google.colab import drive
drive.mount('/content/drive')