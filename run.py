# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import json
from pytorch_transformers.modeling_xlnet import XLNetForMultipleChoice, XLNetConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_xlnet import XLNetTokenizer
from itertools import cycle
from pprint import pprint
import re

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
MODEL_CLASSES = {
    'xlnet': (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
}
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in ( XLNetConfig,)), ())

logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 idx,
                 context_sentence,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 ending_4,
                 nodes,
                 adj_matrix,
                 label = None):
        self.idx = idx
        self.context_sentence = context_sentence
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
            ending_4,
        ]
        self.label = label
        self.nodes=nodes
        self.adj_matrixs=adj_matrix

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            "id: {}".format(self.idx),
            "context_sentence: {}".format(self.context_sentence),
            "ending_0: {}".format(self.endings[0]),
            "ending_1: {}".format(self.endings[1]),
            "ending_2: {}".format(self.endings[2]),
            "ending_3: {}".format(self.endings[3]),
            "ending_4: {}".format(self.endings[4]),
        ]

        if self.label is not None:
            l.append("label: {}".format(self.label))

        return "\n".join(l)


    

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'node_ids': nodes_ids,
                'adj_mask':adj_mask,
            }
            for _, input_ids, input_mask, segment_ids,nodes_ids,adj_mask in choices_features
        ]
        self.label = label

from collections import defaultdict
class Graph:
    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.directed = directed
    def addEdge(self, frm, to):
        self.graph[frm].append(to)
        if self.directed is False:
            if frm == to:
                self.graph[to] = []
        else:
            self.graph[to] = self.graph[to]
    def topoSortvisit(self, s, visited, sortlist):
        visited[s] = True
        for i in self.graph[s]:
            if not visited[i]:
                self.topoSortvisit(i, visited, sortlist)
        sortlist.insert(0, s)
    def topoSort(self):
        visited = {i: False for i in self.graph}
        sortlist = []
        for key in self.graph:
            self.graph[key]=sorted(self.graph[key])
        keys=list(self.graph)
        for v in sorted(keys):
            if not visited[v]:
                self.topoSortvisit(v, visited, sortlist)
        return sortlist 
    
def read_examples(input_file, is_training):
    cont=0
    examples=[]
    with open(input_file+'.concept') as f1,open(input_file+'_NL') as f2 , open(input_file+'.wikigraphnew') as f3:
        for line,line2,line3 in zip(f1,f2,f3):
            # select 10 examples for test
            #if cont>0:
            #    break
            if cont%1000==0:
                logger.info('read cont:{}'.format(cont))

            js=json.loads(line.strip())
            js2=json.loads(line2.strip())
            js3=json.loads(line3.strip())
            qa_list=[]
            context=js['question']['stem']
            node_lists=[]
            adj_matrix_lists=[]

            # temp: .concept
            # temp2: add one _NL
            # temp3: wikipedia

            for temp,temp2,temp3 in zip(js['question']['choices'],js2['question']['choices'],js3['question']['choices']):
                ###################################################################
                # max 20 nodes in Concept-Graph
                k=20
                g = Graph(directed=True)
                temp["node"]=temp["node"][:k]
                # add edges according to the edges provided by Jingjing
                for r in temp['relation']:
                    if r[0]<k and r[1]<k:
                        g.addEdge(r[0],r[1])
                # add one edge pointing to itself
                for i in range(len(temp["node"])):
                    g.addEdge(i,i)
                # get the sequence according to topology sort algorithm
                topsort_seq = g.topoSort()
                sorted_node=[]
                # topsort_seq contains the idx of nodes, only containing numbers
                for i in topsort_seq:
                    # temp['node'][i]: get the evidence corresponding to node i
                    sorted_node.append(temp['node'][i])

                # sorted_node contains the sorted ConceptNet evidence

                sorted_adj_matrix=[]    
                adj_matrix_list=[]
                for tmp in temp['evidence_edges']:
                    # adj_matrix_list: the list of edges. each 1 or 0 represents two nodes are connected or not.
                    adj_matrix_list.append(tmp)
                    # sorted_adj_matrix: each line represents the nodes linked to the node i
                    sorted_adj_matrix.append([0 for i in range(len(tmp))])

                for i in range(min(k,len(adj_matrix_list))):
                    for j in range(min(k,len(adj_matrix_list[i]))):
                        # sorted_adj_matrix[i][j]=1 means node i is connected to j, 0 means not.
                        sorted_adj_matrix[i][j]=adj_matrix_list[topsort_seq[i]][topsort_seq[j]]

                # This is the end of ConceptNet graph construction process.
                ###################################################################
                # Wiki-Graph has at most 10 sentences.
                k=10
                wiki_nodes=[]
                # temp3['node'] means srl results.
                for n in temp3['node']:
                    wiki_nodes.append(''.join(n.split()))
                wiki_evidences=[]
                # wiki_evidences contains the origin wikipedia evidence
                for n in temp3['searched_evidence']['basic']:
                    wiki_evidences.append(''.join(n['text'].split()))

                # each node belongs to which evidence
                node2evidences={}

                for idx,n in enumerate(wiki_nodes):
                    for idx1,e in enumerate(wiki_evidences):
                        # the condition is not influential by ''.join()
                        if n in e:
                            if idx not in node2evidences: 
                                node2evidences[idx]=[]
                            node2evidences[idx].append(idx1)

                            
                g = Graph(directed=True)                
                wiki_adj=[]
                sorted_wiki_adj=[]
                # wiki_adj and sorted_wiki_adj: both k*k, 10*10 matrix
                for i in range(min(k,len(wiki_evidences))):
                    wiki_adj.append([0 for j in range(len(wiki_evidences))])
                    sorted_wiki_adj.append([0 for j in range(len(wiki_evidences))])

                # add edges between two evidences if the components of them are connected...
                for r in temp3['relation']:
                    for i in node2evidences[r[0]]:
                        for j in node2evidences[r[1]]:
                            if i<k and j<k:
                                # construct a directed graph.
                                g.addEdge(i,j)
                                # what does wiki_adj mean?
                                wiki_adj[i][j]=1
                                wiki_adj[j][i]=1
                wiki_evidences=[]
                for tmp in temp3['searched_evidence']['basic'][:k]:
                    wiki_evidences.append(tmp['text'])                                
                for i in range(len(wiki_evidences)):
                    g.addEdge(i,i)
                topsort_seq = g.topoSort()
                sorted_evidences=[]
                for i in topsort_seq:
                    sorted_evidences.append(wiki_evidences[i])
                sorted_adj_matrix=[]
                # the re-ordered wikipedia evidence
                for i in range(min(k,len(wiki_adj))):
                    for j in range(min(k,len(wiki_adj[i]))):
                        sorted_wiki_adj[i][j]=wiki_adj[topsort_seq[i]][topsort_seq[j]]

                ###################################################################
                t=sorted_evidences # only wiki evidence
                if len(t)==0:
                    t=["None"]     
                t1=sorted_node  # only Concept node
                if len(t1)==0:
                    t1=["None"]


                # Add Wiki-Graph triple nodes
                # sentence_number
                # 10个句子，每个句子包含哪些三元组
                srl_triples_index = [[] for i in range(len(wiki_evidences))]

                index = 0
                evidence2arguments = {}
                for srl in temp3['searched_evidence']['basic']:
                    srl_triple_index = []
                    srl_triples = srl['srl_triple']
                    # print('srl_triples:{}'.format(srl_triples))
                    srl_verbs = []
                    if len(srl['srl']['verbs']) > 0:
                        srl_verbs = srl['srl']['verbs']
                    else:
                        srl_verbs = [srl['srl']['verbs']]

                    # for triple, verb, evidence in zip(srl_triples, srl_verbs, wiki_evidences):
                    for verb in srl_verbs:
                        # print(verb)
                        if len(verb) == 0:
                            continue
                        text = verb['description']
                        res = re.findall(r"\[(.*?)\]", text)
                        for temp_res in res:
                            if len(temp_res.split(':'))>=2:
                                temp_res2 = temp_res.split(':')[1].strip()
                                text = text.replace('[' + temp_res + ']', temp_res2)
                        if len(text.split(' ')) != len(verb['tags']):
                            continue

                        # add nodes and edges
                        if text not in evidence2arguments:
                            evidence2arguments[text] = []
                        # print(text)
                        # print(verb['tags'])
                        # res = re.findall(r"\[ARG0:(.*?)\]", verb['description'])
                        # print(res)

                        tags = verb['tags']
                        if not ('B-ARG0' in tags and 'B-V' in tags and 'B-ARG1' in tags):
                            continue

                        # for ARG0
                        start = tags.index('B-ARG0')
                        end = start
                        for temp_i in range(start+1, len(tags)):
                            if tags[temp_i] == 'I-ARG0':
                                end += 1
                            else:
                                break
                        text = text.split(' ')
                        # print('ARG0 {} {} {}'.format(text[start:end+1], start, end))
                        evidence2arguments[' '.join(text)].append([' '.join(text[start:end+1]), start, end])

                        # for verb
                        start = tags.index('B-V')
                        end = start
                        for temp_i in range(start+1, len(tags)):
                            if tags[temp_i] == 'I-V':
                                end += 1
                            else:
                                break
                        # print('VERB {} {}'.format(text[start:end+1],start, end))
                        evidence2arguments[' '.join(text)].append([' '.join(text[start:end + 1]), start, end])

                        # for ARG1
                        start = tags.index('B-ARG1')
                        end = start
                        for temp_i in range(start+1, len(tags)):
                            if tags[temp_i] == 'I-ARG1':
                                end += 1
                            else:
                                break
                        # print('ARG1 {} {}'.format(text[start:end+1],start, end))
                        evidence2arguments[' '.join(text)].append([' '.join(text[start:end + 1]), start, end])
                    # if ' '.join(text) in evidence2arguments:
                    #     print(evidence2arguments[' '.join(text)])
                qa_list.append(([temp['text'],'##'.join(t),'##'.join([temp2['text']] + t1),sorted_wiki_adj,evidence2arguments]))

                # node_lists contains the nodes in ConceptNet
                # adj_matrix_lists contains the adjacent matrix of Wiki-Graph
                node_lists.append(sorted_node)
                # adj_matrix_lists.append(sorted_adj_matrix) Emmm... This is a bug... DOES NOT transfer the matrix at all....
                adj_matrix_lists.append(sorted_wiki_adj)

            cont+=1
            examples.append(
                Example(
                        idx = cont,
                        context_sentence = context,
                        ending_0 = qa_list[0],
                        ending_1 = qa_list[1],
                        ending_2 = qa_list[2],
                        ending_3 = qa_list[3],
                        ending_4 = qa_list[4],
                        nodes=node_lists,
                        adj_matrix=adj_matrix_lists,
                        label = ord(js['answerKey'])-ord('A') if is_training else None
                        ) 
            )


    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        choices_features = []
        if example_index % 1000 == 0 and example_index > 0:
            logger.info('convert example to feature:{}'.format(example_index))
        # change example:
        for ending_index, (ending,node,adj_matrix) in enumerate(zip(example.endings,example.nodes,example.adj_matrixs)):


            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens

            # Question + Answer
            ending_tokens = tokenizer.tokenize(example.context_sentence) + tokenizer.tokenize(
                'The answer is') + tokenizer.tokenize(ending[0])

            evidence2arguments = ending[-1]
            # conceptnet evidence
            concept_context_tokens_choice = tokenizer.tokenize(ending[2])
            _truncate_seq_pair(concept_context_tokens_choice, [], 128)
            concept_context_tokens_choice.append("<sep>")

            tokens_choice = concept_context_tokens_choice
            tokens_choice.extend(tokenizer.tokenize('# Wikipedia #'))
            wiki_nodes_choice = []
            for key in evidence2arguments:
                # one key, one evidence
                words2tokens = {}
                evidence_tokens = []
                key = key.split(' ')
                for temp_i in range(len(key)):
                    tokens = tokenizer.tokenize(key[temp_i])
                    words2tokens[temp_i] = (len(evidence_tokens), len(tokens) + len(evidence_tokens))
                    evidence_tokens.extend(tokens)

                if len(tokens_choice) + len(evidence_tokens) > 256 - len(ending_tokens) - 3 - 2: # 2 for ##
                    break

                # print('tokens choice:{}'.format(len(tokens_choice)))
                origin_length = len(tokens_choice)
                tokens_choice.extend(evidence_tokens)
                tokens_choice.extend(tokenizer.tokenize('##'))

                # 更新原先evidence2arguments中每个argument存储的start与end
                for item in evidence2arguments[' '.join(key)]:
                    start = words2tokens[item[1]][0] + origin_length
                    end = words2tokens[item[2]][1] + origin_length
                    wiki_nodes_choice.append([item[0], start, end])

            tokens_choice.append("<sep>")
            tokens_choice.extend(ending_tokens)
            tokens_choice.append('<cls>')

            tokens = tokens_choice
            segment_ids = [0] * (len(tokens) - len(ending_tokens) - 2) + [1] * (len(ending_tokens) + 1)+[2]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            #padding补在前面，需要对位置进行移动更改
            padding_length = max_seq_length - len(input_ids)
            for node in wiki_nodes_choice:
                node[1] += padding_length
                node[2] += padding_length

            temp=' '.join(str(x) for x in input_ids)
            temp=temp.split('17 7967 20631 17 7967')[0]
            temp=temp.split('7967 7967')
            temp=[len(x.split()) for x in temp]
            input_mask = [1] * len(input_ids)
            padding_length = max_seq_length - len(input_ids)

            input_ids = ([0] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            segment_ids = ([4] * padding_length) + segment_ids

            skip=padding_length+temp[0]+2
            node=[]
            for t in temp[1:]:
                vector=np.zeros(max_seq_length)
                vector[skip:t+skip]=1
                skip+=t+2
                node.append(vector[None,:])
            node=node[:50]
            for i in range(50-len(node)):
                vector=np.zeros(max_seq_length)
                node.append(vector[None,:])


            node_size=len(temp)-1
            matrix=np.zeros((150,150))
            for i,val in enumerate(adj_matrix):
                for j,v in enumerate(adj_matrix[i]):
                    if v==1 and i<50 and j<50 and i<node_size and j<node_size:
                        matrix[i,j]=1
       
            ############################################################################

            temp=' '.join(str(x) for x in input_ids)
            temp=temp.split('17 7967 20631 17 7967')
            skip=len(temp[0].split())+5
            temp = temp[1]

            # temp=temp.split(' 4 ')[0] # 4: <sep>
            # temp=' '.join(temp).split('7967 7967')
            # temp=[len(x.split()) for x in temp]

            for item in wiki_nodes_choice:
                vector = np.zeros(max_seq_length)
                for temp in range(item[1], item[2]):
                    vector[temp] = 1
                node.append(vector[None,:])

            node = node[:150]
            for i in range(150 - len(node)):
                vector = np.zeros(max_seq_length)
                node.append(vector[None, :])
            node = np.concatenate(node, 0)

            node_size = len(wiki_nodes_choice) - 1  # 0.. len(wiki_nodes_choices)-1

            for idx1, node1 in enumerate(wiki_nodes_choice):
                for idx2, node2 in enumerate(wiki_nodes_choice):
                    if node1[0].lower() == node2[0].lower() and idx1<100 and idx2<100 and idx1<node_size and idx2<node_size :
                        matrix[idx1+50, idx2+50] = 1
                        # Directed or Undirected

            # for t in temp:
            #     vector=np.zeros(max_seq_length)
            #     vector[skip:t+skip]=1
            #     skip+=t+2
            #     node.append(vector[None,:])
            # node=node[:100]
            # for i in range(100-len(node)):
            #     vector=np.zeros(max_seq_length)
            #     node.append(vector[None,:])
            # node=np.concatenate(node,0)
            # node_size=len(temp)-1
            # for i,val in enumerate(ending[-1]):
            #     for j,v in enumerate(ending[-1][i]):
            #         if v==1 and i<50 and j<50 and i<node_size and j<node_size:
            #             matrix[i+50,j+50]=1
            # print(len(input_ids))


            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids,node,matrix))


        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example.idx))
            for choice_idx, (tokens, input_ids, input_mask, segment_ids,_,_) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581','_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id = example.idx,
                choices_features = choices_features,
                label = label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--meta_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--report_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()
    
    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    
    # Set seed
    set_seed(args)


    try:
        os.makedirs(args.output_dir)
    except:
        pass

    tokenizer = XLNetTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    config = XLNetConfig.from_pretrained(args.model_name_or_path, num_labels=5)
    # Prepare model
    
    model = XLNetForMultipleChoice.from_pretrained(args.model_name_or_path,args,config=config)


        
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.do_train:

        # Prepare data loader

        train_examples = read_examples(os.path.join(args.data_dir, 'train.jsonl'), is_training = True)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, True)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_node_ids = torch.tensor(select_field(train_features, 'node_ids'), dtype=torch.long)
        all_adj_mask = torch.tensor(select_field(train_features, 'adj_mask'), dtype=torch.long) 
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,all_node_ids,all_adj_mask)
        logger.info(all_input_ids.size())
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)

        num_train_optimization_steps =  args.train_steps


        # Prepare optimizer

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=10000)
        
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        best_acc=0
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0        
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader)
        

        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids,node_ids,adj_mask = batch
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,node_mask=node_ids,adj_mask=adj_mask)
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if args.fp16:
                optimizer.backward(loss)
            else:

                loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1


            if (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0 
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))


            if args.do_eval and (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                for file in ['dev.jsonl']:
                    eval_examples = read_examples(os.path.join(args.data_dir, file), is_training = True)
                    inference_labels=[]
                    gold_labels=[]
                    eval_examples = read_examples(os.path.join(args.data_dir, file), is_training = True)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,False)
                    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)

                    all_node_ids = torch.tensor(select_field(eval_features, 'node_ids'), dtype=torch.long)
                    all_adj_mask = torch.tensor(select_field(eval_features, 'adj_mask'), dtype=torch.long)                     

                    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,all_node_ids,all_adj_mask)
                        
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)  
                        
                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids,node_ids,adj_mask in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)
                        node_ids = node_ids.to(device)
                        adj_mask = adj_mask.to(device) 

                        with torch.no_grad():
                            tmp_eval_loss= model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids,node_mask=node_ids,adj_mask=adj_mask)
                            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,node_mask=node_ids,adj_mask=adj_mask)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        tmp_eval_accuracy = accuracy(logits, label_ids)
                        #if nb_eval_steps==0:
                        #    print(logits)
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
                        eval_loss += tmp_eval_loss.mean().item()
                        eval_accuracy += tmp_eval_accuracy

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = eval_accuracy / nb_eval_examples

                    result = {'eval_loss': eval_loss,
                              'eval_accuracy': eval_accuracy,
                              'global_step': global_step+1,
                              'loss': train_loss}

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*'*80)
                        writer.write('\n')
                    if eval_accuracy>best_acc and 'dev' in file:
                        print("="*80)
                        print("Best Acc",eval_accuracy)
                        print("Saving Model......")
                        best_acc=eval_accuracy
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("="*80)
                        inference_labels=np.concatenate(inference_labels,0)
                        gold_labels=np.concatenate(gold_labels,0)
                        with open(os.path.join(args.output_dir, "error_output.txt"),'w') as f:
                            for i in range(len(eval_examples)):
                                if inference_labels[i]!=gold_labels[i]:
                                    try:
                                        f.write(str(repr(eval_examples[i]))+'\n')
                                        f.write(str(inference_labels[i])+'\n')
                                        f.write("="*80+'\n')
                                    except:
                                        pass
                    else:
                        print("="*80) 
    if args.do_test:
        for file,flag in [('dev.jsonl','dev'),('test.jsonl','test')]:
            inference_labels=[]
            eval_examples = read_examples(os.path.join(args.data_dir, file), is_training = False)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,False)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)

            all_node_ids = torch.tensor(select_field(eval_features, 'node_ids'), dtype=torch.long)
            all_adj_mask = torch.tensor(select_field(eval_features, 'adj_mask'), dtype=torch.long)                     


            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_node_ids,all_adj_mask)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, node_ids,adj_mask in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                node_ids = node_ids.to(device)
                adj_mask = adj_mask.to(device) 
                with torch.no_grad():
                    logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,node_mask=node_ids,adj_mask=adj_mask).detach().cpu().numpy()
                inference_labels.append(np.argmax(logits, axis=1))
            inference_labels=list(np.concatenate(inference_labels,0))
            dic={0:'A',1:'B',2:'C',3:'D',4:'E'}
            with open('data/commonsenseQA/{}.jsonl'.format(flag)) as f1,open(os.path.join(args.output_dir, 'predictions_{}.csv'.format(flag)),'w') as f2:
                for line1,line2 in zip(f1,inference_labels):
                    js=json.loads(line1.strip())
                    predict=dic[line2]
                    f2.write(js['id']+','+predict+'\n')            


if __name__ == "__main__":
    main()
