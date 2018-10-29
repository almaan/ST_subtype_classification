#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:21:04 2018

@author: alma
"""

import pandas as pd
import numpy as np
import os.path as osp
from inspect import getsourcefile

#Path Variables
DIR = '/'.join(osp.realpath(getsourcefile(lambda:0)).split('/')[:-2]) 
LIBRARY = osp.join(DIR,'lib')

#Path to mapping reference
ensgpth = osp.join(LIBRARY,'ensgtohgncmap.reference.tsv')
#Path to genelist, for proper order and extraction
genepathmod = osp.join(LIBRARY,'genelist.sclf')

with open(genepathmod,'r+') as fopen:
            genelist = fopen.readlines()
    
genelist = list(map(lambda x: x.replace('\n',''),genelist))

def ensgtohgnc(data):
    """
    Map From ENSEMBL Id's to HGNC gene symbols. All identifiers where mapping is
    unsuccessfull will be set to 0.0 for all samples.
    
    Arguments:
        
        - data : count-matrix (pandas dataframe)
        
    Output:
        
        - out_df : dataframe with all genes sorted in correct order for usage
                    as input to classifier.
                    
        - stats : report of outcome. Number of successfully mapped genes. 
    
    """
    
    ensg_df = pd.read_csv(ensgpth,sep='\t')
    ensg_df = ensg_df.dropna(axis = 0)
    genelist = np.ravel(pd.read_csv(genepathmod,sep='\t',header=None).values).astype(str)
    
    matched = []
    mapped = []

    colnames = data.columns
    
    #map ENSEMBL Identifiers to HGNC Gene Symbols
    #if multiple matches are found. First is chosen.
    for raw in colnames:
        pos = (ensg_df.iloc[:,0] == raw)
        if pos.sum() > 0:
            pos = np.argmax(pos.values)
            geneid = ensg_df.iloc[pos,1]
            if geneid not in mapped and geneid in genelist:
                matched.append(raw)
                mapped.append(geneid)
    
    #extract matched genes         
    new_df = data[matched]
    new_df.columns = mapped
    
    #genereate 0-filled new data-frame to be filled with extracted values
    out_df = pd.DataFrame(0.0,index = data.iloc[:,0].values,columns = genelist)
    out_df[mapped] = new_df[mapped].values
    
    #stats for logging
    stats = dict(n_matches = len(mapped), n_total = out_df.shape[1])
    
    return out_df, stats

def extractfromhgnc(data):
    """
    Extract genes to be used in classifier and sort these in correct order.
    
    Arguments:
        
         - data : count-matrix (pandas dataframe)
    
    Output:
        
        - out_df : dataframe with all genes sorted in correct order for usage
                    as input to classifier.
                    
        - stats : report of outcome. Number of successfully mapped genes. 
    
    """
    #genereate 0-filled new data-frame to be filled with extracted values
    out_df = pd.DataFrame(0.0,index = data.index, columns = genelist)
    #find which genes from the genelist that are present in sample
    intersect = list(filter( lambda x: x in data.columns, genelist))
    try:
        #extract matched genes to new data frame
        out_df[intersect] = data[intersect].values
        error = False
    except Exception as e:
        #record error
        error = e
    
    stats = dict(n_matches = len(intersect), n_total = out_df.shape[1], error = error)    
    return out_df, stats


