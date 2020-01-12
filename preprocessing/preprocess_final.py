import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os,sys
from gtfparse import read_gtf
from networkx.readwrite import json_graph
import networkx as nx
import re
from collections import defaultdict
from scipy import sparse 
import scipy.stats as stats
from node2vec import Node2Vec
from tqdm import tqdm


def build_expression_network(expression, thr=0.5):
    foo = lambda x: pd.Series([i for i in reversed(x.split(','))])
    dream_data = pd.DataFrame(expression['log_expValuesTumor'].apply(foo))
    
    # calculate correlation
    dream_data=dream_data.T
    dream_data=dream_data.astype(float)
    cor_matrix,pval = stats.spearmanr(dream_data,nan_policy='omit')
    
    expression_list = defaultdict(list)

    for i in range(cor_matrix.shape[0]):
        for j in range(i+1, cor_matrix.shape[1]):
            if cor_matrix[i,j] >= thr or cor_matrix[i,j] <= -thr:
                inputs = expression['geneSymbol'][i] 
                outputs = expression['geneSymbol'][j] 
                expression_list[inputs].append((outputs, cor_matrix[i,j]))
                expression_list[outputs].append((inputs, cor_matrix[i,j]))
    
    return expression_list

def preprocess(pvalue_thr=1e-200, cancer_type='BRCA'):
    ##############################################################
    ####### Select pseudogene and coding genes
    ##############################################################
    print("Select pseudogene and coding genes")
    gencode = read_gtf("../data/raw_data/gencode.v29.annotation.gtf")
    gencode = gencode[gencode['feature'] == 'gene']

    # select pseudogenes
    pseudogene = gencode[gencode['gene_type'].isin(["transcribed_unprocessed_pseudogene","transcribed_processed_pseudogene","translated_processed_pseudogene"])]
    pseudogene = pseudogene.drop(["score",
                                  "strand","frame","level","tag","exon_number", 
                 "exon_id","ont","protein_id","ccdsid","transcript_support_level",
                "havana_transcript","havana_gene","source","transcript_id",
                                 "transcript_name","transcript_type"],axis=1)
    pseudogene.drop_duplicates(subset=['gene_name'], keep='first', inplace=True)
    
    print("Pseudogene number: ", pseudogene.shape[0])

    # select coding genes
    coding = gencode[gencode['gene_type'] == 'protein_coding']
    coding = coding.drop(["score",
                          "strand","frame","level","tag","exon_number",                         "exon_id","ont","protein_id","ccdsid","transcript_support_level",                    "havana_transcript","havana_gene","source","transcript_id",
                                 "transcript_name","transcript_type"],axis=1)
    coding.drop_duplicates(subset=['gene_name'], keep='first', inplace=True)
    
    print("coding gene number: ", coding.shape[0])
    
    
    
    ##############################################################
    ####### generate genome sequence
    ##############################################################
    print("generate genome sequence")
    with open("../data/raw_data/GRCh38.primary_assembly.genome.fa") as f:
        data = f.readlines()

    chr_seq_map = dict()
    i = 0
    while i < len(data):
        if data[i][0] == ">":
            key = data[i].split(" ")[0]
            j = 1
            temp = []
            while (i+j) < len(data):
                if data[i+j][0] != ">":
                    temp.append(data[i+j][:-1])
                    j = j+1
                else:
                    break
            value = "".join(temp)
            chr_seq_map[key] = value
            i = i + j         

    chr_seq_mapping = dict()
    for key,value in chr_seq_map.items():
        if key[:4] == '>chr':
            chr_seq_mapping[ key[1:] ] = value

    def func(x):
        temp = chr_seq_mapping[x['seqname']]
        return temp[x['start']:x['end']]
    pseudogene['sequence'] = pseudogene.apply(func, axis=1)
    coding['sequence'] = coding.apply(func, axis=1)
    all_genes = pseudogene.append(coding, ignore_index=True, sort=False)
    all_genes.drop_duplicates(subset=['gene_name'], keep='first', inplace=True)
    
    
    
    ##################################################################
    ######## choose final pseudogene and coding candidates
    
    pseudo_list = all_genes[all_genes['gene_type'] != 'protein_coding']['gene_name'].values
    coding_list = all_genes[all_genes['gene_type'] == 'protein_coding']['gene_name'].values

    # build similarity network
    print("filtering blast results")

    similarity_res = pd.read_csv("../data/raw_data/blast_similarity.csv", names=['query','target','evalue'])
    similarity_res = similarity_res[similarity_res['evalue'] < pvalue_thr]

    # delete self-self pairs
    similarity_res = similarity_res[(similarity_res['query'] != similarity_res['target'])]
    # Only select pseudogene as the query
    similarity_select = similarity_res[similarity_res['query'].isin(pseudo_list)]
    # select corresonding coding genes
    similarity_candidate = set(similarity_select['target'].unique()) | set(pseudo_list)
    # filter by the candidates
    similarity_final = similarity_res[similarity_res['query'].isin(similarity_candidate)]
    similarity_final = similarity_final[similarity_final['target'].isin(similarity_final['query'].unique())]
    final_similarity_candidate = np.array(list(set(similarity_final['query'].unique()) | set(similarity_final['target'].unique())))

    # this is all the genes we will use in our model
    final_all_genes = all_genes[all_genes['gene_name'].isin(final_similarity_candidate)]
    final_all_genes.index = range(len(final_all_genes))
    all_genes_mapping = dict(zip(final_all_genes['gene_name'], range(len(final_all_genes))))
    print("Dataset size: ", final_all_genes.shape[0])
    
    
    
    ##################################################################
    ###################### Build Networks ############################
    
    print("build similarity adj matrix")
    similarity_final['query_id'] = similarity_final['query'].apply(lambda x:all_genes_mapping[x])
    similarity_final['target_id'] = similarity_final['target'].apply(lambda x:all_genes_mapping[x])
    adj_simi = np.zeros((len(all_genes_mapping),len(all_genes_mapping)))
    for i,row in tqdm(similarity_final.iterrows()):
        adj_simi[ row['query_id'] ][ row['target_id'] ] = 1
    sAdj_simi = sparse.csr_matrix(adj_simi)
    sparse.save_npz("../data/final_input/adj_simi.npz",sAdj_simi)
    
    
    print("build TCGA co-expression network")
    df = pd.read_csv('../data/raw_data/TCGA_'+ cancer_type + '.csv', names =["index","id","name","geneSymbol","MedianExpValueTumor",
                         "MedianExpValueNormal","log_aveExpValueTumor",
                         "log_aveExpValueNormal","expValuesTumor","expValuesNormal",
                         "log_expValuesTumor","log_expValuesNormal","paired"],
                             index_col=["index"],skipinitialspace=True)
    expression_new=df[['geneSymbol', 'log_expValuesTumor']]
    # only select genes that are included in the pseudogene and coding gene list
    all_genes_name = final_all_genes['gene_name'].values
    expression_selected = expression_new[ expression_new['geneSymbol'].isin(all_genes_name) ]
    expression_selected = expression_selected.reset_index(drop=True)
    
    # build expression networks 
    expression_pairs = build_expression_network(expression_selected)
    final_co_expression = defaultdict(list)
    for key,value in tqdm(expression_pairs.items()):
        temp = [(all_genes_mapping[x[0]],x[1]) for x in value]
        final_co_expression[ all_genes_mapping[key] ] = temp
    adj_co_expression = np.zeros((len(all_genes_mapping),len(all_genes_mapping)))
    for key,value in tqdm(final_co_expression.items()):
        for x in value:
            adj_co_expression[key][ x[0] ] = 1
    sAdj_co = sparse.csr_matrix(adj_co_expression)
    sparse.save_npz("../data/final_input/adj_TCGA_"+cancer_type +".npz",sAdj_co)
    
    print("generate node2vec embeddings for co-expression network")
    G_coexp = nx.from_scipy_sparse_matrix(sAdj_co)
    node2vec_coexp = Node2Vec(G_coexp, dimensions=256, walk_length=15, num_walks=150, workers=28)
    model_coexp = node2vec_coexp.fit(window=10, min_count=1, batch_words=4)  
    model_coexp.wv.save_word2vec_format("../data/final_input/node2vec_TCGA_"+cancer_type+".txt")

    
    
    print("build ppi and genetic interaction network")
    biogrid = pd.read_table("../data/raw_data/BIOGRID-ALL-3.5.173.tab2.txt")
    biogrid = biogrid[(biogrid['Organism Interactor A'] == 9606) & (biogrid['Organism Interactor B'] == 9606)]
    biogrid = biogrid[['#BioGRID Interaction ID','Entrez Gene Interactor A','Entrez Gene Interactor B', 'Official Symbol Interactor A','Official Symbol Interactor B',
                      'Experimental System','Experimental System Type']]
    biogrid = biogrid[(biogrid['Official Symbol Interactor A'].isin(final_all_genes['gene_name'].unique())) & 
                     (biogrid['Official Symbol Interactor B'].isin(final_all_genes['gene_name'].unique()))]
    biogrid['query_id'] = biogrid['Official Symbol Interactor A'].apply(lambda x:all_genes_mapping[x])
    biogrid['target_id'] = biogrid['Official Symbol Interactor B'].apply(lambda x:all_genes_mapping[x])

    adj_ppi = np.zeros((len(all_genes_mapping),len(all_genes_mapping)))
    for i,row in tqdm(biogrid.iterrows()):
        adj_ppi[ row['query_id'] ][ row['target_id'] ] = 1
    sAdj_ppi = sparse.csr_matrix(adj_ppi)
    sparse.save_npz("../data/final_input/adj_ppi.npz",sAdj_ppi)
    
    print("generate node2vec embeddings for PPI and genetic interaction network")
    G_ppi = nx.from_scipy_sparse_matrix(sAdj_ppi)
    node2vec_ppi = Node2Vec(G_ppi, dimensions=256, walk_length=15, num_walks=150, workers=28)
    model_ppi = node2vec_ppi.fit(window=10, min_count=1, batch_words=4)  
    model_ppi.wv.save_word2vec_format("../data/final_input/node2vec_ppi.txt")
    
    
    ##########################################################
    ############ Generate feature dataframe ##################
    
    print("Get GO labels for both pseudogenes and coding genes")
    goa = pd.read_csv("../data/raw_data/goa_human.gaf",sep="\t", 
                      skiprows=31,names=['DB','DB Object ID','DB Object Symbol','Qualifier','GO','reference','Evidence',
                                          'with form','Aspect','DB Object Name','Synonym','type','Taxon','Date','Assigned by',
                                          'extension','Gene product form ID'])
    goa = goa[goa['DB Object Symbol'].isin(final_all_genes['gene_name'])]
    goa_F = goa[goa['Aspect']=='F']
    goa_P = goa[goa['Aspect']=='P']
    goa_C = goa[goa['Aspect']=='C']
    final_all_genes['MF'] = final_all_genes['gene_name'].apply(lambda x: list(goa_F[goa_F['DB Object Symbol'] == x]['GO']))
    final_all_genes['BP'] = final_all_genes['gene_name'].apply(lambda x: list(goa_P[goa_P['DB Object Symbol'] == x]['GO']))
    final_all_genes['CC'] = final_all_genes['gene_name'].apply(lambda x: list(goa_C[goa_C['DB Object Symbol'] == x]['GO']))

    from go_anchestor import get_gene_ontology, get_anchestors

    go = get_gene_ontology()
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'

    new_cc = []
    new_mf = []
    new_bp = []

    for i, row in final_all_genes.iterrows():
        labels = row['CC']
        temp = set([])
        for x in labels:
            temp = temp | get_anchestors(go, x)
        temp.discard(CELLULAR_COMPONENT)
        new_cc.append(list(temp))

        labels = row['MF']
        temp = set([])
        for x in labels:
            temp = temp | get_anchestors(go, x)
        temp.discard(MOLECULAR_FUNCTION)
        new_mf.append(list(temp))

        labels = row['BP']
        temp = set([])
        for x in labels:
            temp = temp | get_anchestors(go, x)
        temp.discard(BIOLOGICAL_PROCESS)
        new_bp.append(list(temp))

    final_all_genes['cc'] = new_cc
    final_all_genes['mf'] = new_mf
    final_all_genes['bp'] = new_bp

    mf_items = [item for sublist in final_all_genes['mf'] for item in sublist]
    mf_unique_elements, mf_counts_elements = np.unique(mf_items, return_counts=True)

    bp_items = [item for sublist in final_all_genes['bp'] for item in sublist]
    bp_unique_elements, bp_counts_elements = np.unique(bp_items, return_counts=True)

    cc_items = [item for sublist in final_all_genes['cc'] for item in sublist]
    cc_unique_elements, cc_counts_elements = np.unique(cc_items, return_counts=True)

    mf_list = mf_unique_elements[np.where(mf_counts_elements > 25)]
    cc_list = cc_unique_elements[np.where(cc_counts_elements > 25)]
    bp_list = bp_unique_elements[np.where(bp_counts_elements > 250)]
    
    print("CC:", len(cc_list))
    print("MF:", len(mf_list))
    print("BP:", len(bp_list))

    temp_mf = final_all_genes['mf'].apply(lambda x: list(set(x) & set(mf_list)))
    final_all_genes['temp_mf'] = temp_mf
    temp_cc = final_all_genes['cc'].apply(lambda x: list(set(x) & set(cc_list)))
    final_all_genes['temp_cc'] = temp_cc
    temp_bp = final_all_genes['bp'].apply(lambda x: list(set(x) & set(bp_list)))
    final_all_genes['temp_bp'] = temp_bp

    mf_dict = dict(zip(list(mf_list),range(len(mf_list))))
    cc_dict = dict(zip(list(cc_list),range(len(cc_list))))
    bp_dict = dict(zip(list(bp_list),range(len(bp_list))))
    mf_encoding = [[0]*len(mf_dict) for i in range(len(final_all_genes))]
    cc_encoding = [[0]*len(cc_dict) for i in range(len(final_all_genes))]
    bp_encoding = [[0]*len(bp_dict) for i in range(len(final_all_genes))]

    for i,row in final_all_genes.iterrows():
        for x in row['temp_mf']:
            mf_encoding[i][ mf_dict[x] ] = 1
        for x in row['temp_cc']:
            cc_encoding[i][ cc_dict[x] ] = 1
        for x in row['temp_bp']:
            bp_encoding[i][ bp_dict[x] ] = 1

    final_all_genes['cc_label'] = cc_encoding
    final_all_genes['mf_label'] = mf_encoding
    final_all_genes['bp_label'] = bp_encoding

    final_all_genes.rename(columns={"temp_mf":"filter_mf","temp_bp":"filter_bp","temp_cc":"filter_cc"},inplace=True)
    final_all_genes.drop(columns=['MF','CC','BP','mf','cc','bp'],inplace=True)

    with open("../data/final_input/mf_list.txt","w") as f:
        for x in list(mf_list):
            f.write(x+"\n")

    with open("../data/final_input/cc_list.txt","w") as f:
        for x in list(cc_list):
            f.write(x+"\n")

    with open("../data/final_input/bp_list.txt","w") as f:
        for x in list(bp_list):
            f.write(x+"\n")
    
    print("Add microRNA interactions as features")
    miRNA = pd.read_excel("../data/raw_data/miRNA.xlsx")
    miRNA = miRNA[miRNA['Target Gene'].isin(final_all_genes['gene_name'].unique())]
    selected_miRNA = miRNA['miRNA'].value_counts().index[(miRNA['miRNA'].value_counts() > 250)]
    miRNA = miRNA[miRNA['miRNA'].isin(selected_miRNA)]

    micro_mapping = dict(zip(list(selected_miRNA), range(len(selected_miRNA))))
    micro_encoding = []
    for i,row in tqdm(final_all_genes.iterrows()):
        cur_mir = miRNA[miRNA['Target Gene'] == row['gene_name']]['miRNA']
        temp_encoding = [0]*len(selected_miRNA)
        for x in cur_mir:
            temp_encoding[ micro_mapping[ x ] ] = 1
        micro_encoding.append(temp_encoding)

    final_all_genes['microRNA_250'] = micro_encoding

    with open("../data/final_input/microRNA_list.txt","w") as f:
        for x in list(selected_miRNA):
            f.write(x+"\n")
    
    
    print("Add GTEx median expression profiles")
    GTEx = pd.read_csv("../data/raw_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct",
                  skiprows=2, sep='\t')
    final_all_genes['gene_id'] = final_all_genes['gene_id'].apply(lambda x:x.split(".")[0])
    GTEx['Name'] = GTEx['Name'].apply(lambda x:x.split(".")[0])
    GTEx_new = pd.DataFrame({'gene_id':GTEx['Name'], 
                             'expression':GTEx.iloc[:,2:].values.tolist()})
    GTEx_new.drop_duplicates(subset=['gene_id'], keep='first', inplace=True)
    final_all_genes = pd.merge(final_all_genes, GTEx_new, on='gene_id',how='left')
    final_all_genes['expression'] = final_all_genes['expression'].apply(lambda d: d if isinstance(d, list) else [0.0]*54)
    
    def reshape(features):
        return np.hstack(features).reshape((len(features),len(features[0])))
    
    print("generate GTEx node2vec features...")
    expression = reshape(final_all_genes['expression'].values).T
    cor_matrix, pval = spearmanr(expression, nan_policy='omit')
    cor_matrix = np.nan_to_num(cor_matrix,0)
    
    thr = 0.9
    adj_coexp_GTEx = np.zeros((final_all_genes.shape[0],final_all_genes.shape[0]))
    adj_coexp_GTEx[ cor_matrix > thr ] = 1
    adj_coexp_GTEx = sparse.csr_matrix(adj_coexp_GTEx)
    sparse.save_npz("../data/final_input/adj_GTEx.npz",adj_coexp_GTEx)
    
    print("generate node2vec embeddings for GTEx co-expression network")
    G_coexp = nx.from_scipy_sparse_matrix(adj_coexp_GTEx)
    node2vec_coexp = Node2Vec(G_coexp, dimensions=256, walk_length=15, num_walks=150, workers=28)
    model_coexp = node2vec_coexp.fit(window=10, min_count=1, batch_words=4)  
    model_coexp.wv.save_word2vec_format("../data/final_input/node2vec_GTEx.txt")

    
    print("Saving feature dataframe")
    final_all_genes.to_pickle("../data/final_input/features_all.pkl")
    features_input = final_all_genes.loc[:,['gene_id','gene_name','gene_type',
                                           'cc_label','mf_label','bp_label',
                                           'microRNA_250','expression']]
    features_input.to_pickle("../data/final_input/features.pkl")
    
    
preprocess(pvalue_thr=1e-200, cancer_type='BRCA')

