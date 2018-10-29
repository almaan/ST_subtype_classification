#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Alma Andersson 2018-10-29

"""
from utils import MRFCluster, MapIds
from utils import SmoothViz as sv
from sklearn.preprocessing import LabelEncoder

from datetime import datetime as dt

from scipy.stats import entropy as ent

import tensorflow as tf
from keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from inspect import getsourcefile
import argparse as arp
import os.path as osp
import logging
import warnings
import sys
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore',
                        category=DeprecationWarning)   
warnings.filterwarnings('ignore',
                        category=RuntimeWarning)

DIR = '/'.join(osp.realpath(getsourcefile(lambda:0)).split('/')[:-1]) 
LIBRARY = osp.join(DIR,'lib')

def FilterCoordinates(transcrd,
                      sptcrd,
                      pixcrd):
    
    x = transcrd.tolist()
    y = sptcrd.tolist()
    w = pixcrd.tolist()
    
    idx = [ii  for ii in range(len(x)) if x[ii] in y]
    
    x = [x[ii] for ii in idx]
    z = [y[y.index(jj)] for jj in x]
    p = [w[y.index(jj)] for jj in x]
    
    return np.array(z), np.array(p), idx

def Main(data_pth,
         spots_pth,
         img_pth,
         n_lda_iterations,
         theta,transpose,
         output, 
         saveimg,
         noswap, 
         logname,
         verbose,
         ensembl,
         ):
    
    subtypes = np.array(['luma',
                         'lumb',
                         'her2nonlum',
                         'her2lum',
                         'tnbc'])
    
    colors = dict(luma = '#49D8A6',
                  lumb = '#0C8A99', 
                  her2lum = '#FF7C94',
                  tnbc='#7AAC49',
                  her2nonlum = '#FFCA4F',
                  nontumor='#424242'
                  )
                  
    trained_model_path = osp.join(LIBRARY,'subtype_classifier_model.h5')
    
    ext_data = data_pth.split('.')[-1]
    ext_spots = spots_pth.split('.')[-1]
    
    
    # =============================================================================
    # ----------------------------Read count-matrix--------------------------------    
    # =============================================================================
    
    if ext_data == 'tsv':
        logger.info(f'Identified count-matrix as tsv-file')
        data_df = pd.read_csv(data_pth,sep='\t',header = 0, index_col = 0)
        if transpose:
            data_df = data_df.T
        coords = np.array(list(map(lambda x: x.split('x'),data_df.index)),dtype=int)
        if coords.shape[1] != 2:
            logger.error(f'count-matrix is not correctly formated. row-index should be [x_coordinate]x[y_coordinate]')
            logger.error(f'Exiting')
            sys.exit(0)

    elif ext_data == 'csv':
        logger.info(f'Identified count-matrix as csv-file')
        data_df = pd.read_csv(data_pth,sep=',',header= 0,index_col = 0)
        if transpose:
           logger.info(f'Transposing count-matrix')
           data_df = data_df.T
        coords = np.array(list(map(lambda x: x.split('x'),data_df.index)),dtype=int)
        if coords.shape[1] != 2:
            logger.error(f'count-matrix is not correctly formated. row-index should be [x_coordinate]x[y_coordinate]')
            logger.error(f'Exiting')
            sys.exit(0)

    else:
        logger.error(f'count-matrix is not correctly formatted. Exiting.')
        sys.exit(0)
    
    
    # =============================================================================
    # ----------------------------Read spot-file-----------------------------------    
    # =============================================================================
    
    skiptest = True
    skip = None
    
    if len(spots_pth) > 0:
        if ext_spots == 'tsv':
            logger.info(f'spot file identified as tsv-format')
            while skiptest:
                spots_df = pd.read_csv(spots_pth,sep='\t',
                                       skiprows = skip,
                                       index_col=None)
                
                if 'x' not in spots_df.columns or 'y' not in spots_df.columns:
                    if skip == 1:
                        logger.error(f'spot-file is not properly formatted. Exiting.')
                        sys.exit(0)
                    skip = 1
                    
                elif 'x' in spots_df.columns and 'y' in spots_df.columns:
                    skiptest = False
            
        elif ext_spots == 'csv':
            logger.info(f'spot file identified as csv-file')
            while skiptest:
                spots_df = pd.read_csv(spots_pth,sep=',',
                                       skiprows = skip,
                                       index_col=None)
                
                if 'x' not in spots_df.columns or 'y' not in spots_df.columns:
                    if skip == 1:
                        logger.error(f'spot-file is not properly formatted. Exiting.')
                        sys.exit(0)
                    skip = 1
                    
                elif 'x' in spots_df.columns and 'y' in spots_df.columns:
                    logger.info(f'spot-file successfully read. Number of spots : {spots_df.shape[0]:d}')
                    skiptest = False 
                
        elif ext_spots == 'txt':
            logger.info(f'spot-file identified as txt-file')
            
            spots_df = pd.read_csv(spots_pth,sep='\t',header=0,index_col=0)
            spots_df['x'] = np.round(spots_df['x'].values/100,0).astype(int)
            spots_df['y'] = np.round(spots_df['y'].values/100,0).astype(int)
            spots_df = spots_df.rename(columns={'X':'pixel_x', 'Y':'pixel_y'})

        else:
            logger.error('spot-file is not properly formatted. Exiting')
            sys.exit(0)
    
    # =============================================================================
    # ----------------------------Load gene-list-----------------------------------        
    # =============================================================================

    if ensembl:
        logger.info(f'Mapping from ENSEMBL Identifier to HGNC Gene Symbols')
        data_df, stats = MapIds.ensgtohgnc(data_df)
        logger.info(f"Identified genes : {stats['n_matches']:d}/{stats['n_total']:d}")
    else:
        logger.info(f'Extract genes directly from input Matrix')
        data_df,stats = MapIds.extractfromhgnc(data_df)
        
        if stats['error']:
            logger.error(f"Extraction was unsuccessfull: {stats['error']:s}")
        else:
            logger.info(f"Identified genes : {stats['n_matches']:d}/{stats['n_total']:d}")
    
    
    # =============================================================================
    #-----------------------------Classification-----------------------------------     
    # =============================================================================
 
    if len(img_pth) > 0 and len(spots_pth) > 0:
        sptcrd = spots_df[['x','y']].values
        pixcrd = spots_df[['pixel_x','pixel_y']].values
        coords, pixcrd, pos = FilterCoordinates(coords,sptcrd,pixcrd)
    
        data_df = data_df.iloc[pos,:]
    
    
    clf = load_model(trained_model_path)
    predicted_category = clf.predict(np.log2(data_df.values.astype(float) + 1.0))
    predicted_entropy = ent(predicted_category.T)
   
    l1 = LabelEncoder()
    l1.fit(subtypes)
    highest_prob_category_num = np.argmax(predicted_category,axis=1)
    
    highest_prob_category_str =  l1.inverse_transform(highest_prob_category_num)
    subtype_names_inverted = l1.inverse_transform(np.arange(len(subtypes)))
    highest_prob_category_str_backup = np.copy(highest_prob_category_str)
    
    maxvals = np.array([np.max(predicted_category[x,:]) for x in range(predicted_category.shape[0])],dtype=np.float)
    maxvals_backup = np.copy(maxvals)
       
    # =============================================================================
    # ---------------------------MRF-clustering------------------------------------
    # =============================================================================
    
    graph = MRFCluster.Grapher(coords,data_df.values,
                               theta=theta,
                               entropy = predicted_entropy, 
                               logname = logname,
                               verbose = verbose,
                               )
    
    if len(img_pth) > 0 and len(spots_pth) > 0:
        graph.Load_Image_Information(spots = spots_df, img_pth=img_pth)
    
    graph.Cluster(n_iterations=n_lda_iterations)
    idx = graph.idx
    
    highest_prob_category_str[idx == 1] = 'nontumor' 
    maxvals[idx == 1] = 1.0
    
    # =============================================================================
    # ---------------------------Visualization-------------------------------------    
    # =============================================================================
    
    try_labels = True
    
    while try_labels:  
        if len(img_pth) > 0 and len(spots_pth) > 0:
            entropy_limit = 1.1
            simg = sv.GenerateImage(pixcrd,
                                    img_pth,
                                    highest_prob_category_str,
                                    predicted_category,
                                    graph.spot_radius,
                                    )
            
            fig = plt.figure(figsize = (28.00,13.00))
            
            subp1 = plt.subplot(132)
            plt.title('subtype classification',fontsize = 25)
            plt.imshow(plt.imread(img_pth))
            plt.imshow(simg,interpolation='gaussian',alpha = 0.4)
            
            plt.scatter(pixcrd[predicted_entropy < entropy_limit,0],
                        pixcrd[predicted_entropy < entropy_limit,1],
                        s=graph.spot_radius*4.0,
                        alpha = 0.2,
                        c='black',
                        )
            
            plt.scatter(pixcrd[predicted_entropy >= entropy_limit,0],
                        pixcrd[predicted_entropy >= entropy_limit,1],
                        s=graph.spot_radius*4.0,
                        alpha = 0.2,
                        c='black',
                        edgecolors='red')
            
            ax = fig.get_axes()[0]
            ax.set_xticks([])
            ax.set_yticks([])
            
            
            subp2 = plt.subplot(133)        
            plt.title('subtype distribution',fontsize = 25)
            s,c = np.unique(highest_prob_category_str[highest_prob_category_str != 'nontumor'],return_counts=True)
            
            bar = sns.barplot(x=s,y=c,palette=colors)
            bar.tick_params(labelsize=40)
            plt.setp(bar.get_xticklabels(), rotation=45)
     
            for side in ['top','bottom','right','left']:
                subp1.spines[side].set_visible(False)
                if side not in ['left','bottom']:
                    subp2.spines[side].set_visible(False)
    
            subp3 = plt.subplot(131)
            
            plt.imshow(plt.imread(img_pth))
            plt.title('Entropy',fontsize = 25)
            
            entropy_size = ent(np.ones(len(subtypes))*1./len(subtypes)) - predicted_entropy
            entropy_size = entropy_size/np.max(entropy_size)
            
            plt.scatter(pixcrd[:,0],
                        pixcrd[:,1],
                        s=entropy_size*graph.spot_radius*4.0,
                        alpha = 0.5,
                        c='red')
            
            ax = fig.get_axes()[0]
            ax.set_xticks([])
            ax.set_yticks([])
            
            fig.tight_layout()
            
            # =============================================================================
            # -----------------------------Save Image--------------------------------------            
            # =============================================================================
            
            if len(saveimg) > 0:
                #if extension is valid use. Otherwise save as png file with same basename as provided.
                ext = saveimg.split('.')[-1]
                if ext not in ['png','jpg','gif','bmp']:
                    ext = 'png'
                
                save_pth = '.'.join(saveimg.split('.')[0:-1]) + '.' + ext
                fig.savefig(save_pth)
                logger.info(f'saving image From MRF-Clustering at :  {save_pth:s}')

            
            # =============================================================================
            # -------------Prompt for swapping decision if not blocked---------------------            
            # =============================================================================
            if noswap:
                try_labels = False
            else:
                plt.show()
                ans = input('swap tumor and non-tumor regions (y/n) >> ')
                if ans.lower() == 'y' or ans.lower() == 'yes' or ans == '1':
                    logger.infor(f'swapping classification labels')
                    
                    #reset classification for new interpolation
                    highest_prob_category_str = np.copy(highest_prob_category_str_backup)
                    maxvals = np.copy(maxvals_backup)
                    idx = 1 - idx
                    highest_prob_category_str[idx == 1] = 'nontumor' 
                    maxvals[idx == 1] = 1.0
                    #do not ask for second swap
                    noswap = True
                    
                else:
                    try_labels = False
    # =============================================================================
    #-------------------------------Save Results-----------------------------------
    # =============================================================================
    
    if len(output) > 0:
        tumor_label = np.array(['nontumor'] * len(idx))
        tumor_label[idx == 0] = 'tumor'
        #generate dict to be saved
        data_out = dict(x = coords[:,0], 
                        y = coords[:,1],
                        tumor =tumor_label,
                        pred_subtype = highest_prob_category_str)
        
        #add probability of each subtupe
        predicted_category[tumor_label == 'nontumor',:] = 0.0
        for num,sub in enumerate(subtype_names_inverted):
            data_out.update({'p_' + sub: predicted_category[:,num]})
        #transform to dataframe
        out_df = pd.DataFrame(data = data_out, index = data_df.index)
        out_df.to_csv(output,sep = '\t',header = True, index = True)
        
        logger.info(f'saving results From MRF-Clustering at :  {output:s}')

if __name__ == '__main__':
    
    prs = arp.ArgumentParser()
    prs.add_argument('-c', '--countmat',
                     required=True,
                     type=str,
                     help = f' '.join(['count-matrix in either .tsv or .csv format.',
                                       'gene-names should be given as column headers,',
                                       'whilst spot coordinates should be given as rownames',
                                       'in the format [x_coord]x[y_coord].',
                                       ])
                    )
    
    prs.add_argument('-i','--img',
                     required=False,
                     type=str,
                     default='',
                     help=f' '.join(['path to image-file compatible with the provided spot-file.',
                                     'note how large image-files might require large memory allocations',
                                     ])
                    )
    
    prs.add_argument('-s','--spots',
                     required = False,
                     type=str,
                     default='',
                     help = ' '.join(['spot-file compatible with provided count-matrix and image-file.',
                                      'should have all the fields pixel_x,pixel_y, x, y, new_x and new_y.',
                                      'if first line is information of spot-file based on old ST-pipeline output',
                                      'the first row will be skipped.',
                                      ])
                    )
    
    prs.add_argument('-T','--transpose',
                     required=False,
                     default= False,
                     action = 'store_true',
                     help = ' '.join(['include if count-matrix have genes as rows and',
                                      'samples as columns.',
                                     ])
                    )
    
    
    prs.add_argument('-t','--theta',
                     required =False,
                     default= 0.1,
                     type = float,
                     help = ' '.join(['influence of neigboring spots on classification of a spot.',
                                      'an increased value will give a "smoother" clustering.',
                                      'choose values within the interval [0,1].',
                                      'default is set to 0.1',
                                     ])
                     )
    
    prs.add_argument('-n','--iter',
                     required = False, 
                     default = 75,
                     type = int,
                     help = ' '.join(['number of iterations to be used during LDA.',
                                      'the parameter does influence the outcome to some extent',
                                      'thus it is recommended to test multiple parameters.',
                                      'default is set to 75.',
                                      ])
                     )
    
    prs.add_argument('-o','--output',
                     required = False,
                     default = '',
                     type = str,
                     help = ' '.join(['filename of output.',
                                      'the output will be a tsv file where each spot',
                                      'is classified as any of the subtypes or non-tumorous',
                                      'if no filename is specified the result will not be saved.',
                                      ])
                     )
    prs.add_argument('-si', '--saveimage',
                     required = False, 
                     default = '',
                     type = str,
                     help = ' '.join(['filename of saved image.'
                                      'if no name is given the image is not saved',
                                      'output consists of three images.',
                                      '(1) visualization of subtypes by overlaid on the HE-image,',
                                      '(2) distribution of subtypes visualized with a bar plot and',
                                      '(3) entropy of each point.',
                                     ])
                     )
    
    prs.add_argument('-ns','--noswap',
                     required = False, 
                     default = False,
                     action = 'store_true',
                     help = ' '.join(['include if no alternative to swap cluster labels should be given.',
                                      'if flag is used lowest entropy cluster is taken as tumor cluster',
                                      ])
                     )

    prs.add_argument('-e','--ensembl',
                     required = False,
                     default = False,
                     action = 'store_true',
                     help = ' '.join(['include if gene identifiers are of ENSEMBL type.',
                                      'genes will be mapped to hgnc gene-names and extracted.',
                                      ])
                     )
    prs.add_argument('-l','--log',
                     required = False,
                     default = False,
                     help = ' '.join(['name of log-file.',
                                      'if none is provided the log file will be named based on date and time.',
                                      'log-file is deposited in working directory.',
                                      ])
                     )
    prs.add_argument('-v','--verbose',
                     required = False,
                     default = True,
                     action = 'store_false',
                     help= ' '.join(['include flag in order to silence output.',
                                     'logged messages will by default be passed to stdout if',
                                     'flag is not passed',
                                     ])
                     )
    
    arg = prs.parse_args()

    if isinstance(arg.log,bool):
    
        logname = str(dt.now())
        logname = '.'.join([logname.replace(' ','-'),'log'])
    
        logging.basicConfig(filename=logname,
                        level=logging.INFO)
    
        logger = logging.getLogger('Main')
    else:
        logname = arg.log
        
    if arg.verbose:
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    try: 
        arguments = dict(transpose = arg.transpose,
                         n_lda_iterations = int(arg.iter),
                         theta = arg.theta,
                         output = arg.output,
                         saveimg = arg.saveimage,
                         noswap = arg.noswap,
                         data_pth = arg.countmat,
                         spots_pth = arg.spots,
                         img_pth = arg.img,
                         logname = logname,
                         verbose = arg.verbose,
                         ensembl = arg.ensembl,
                         )
        
    except SystemExit:
        logger.error(f' '.join(['Failed to perform subtype classification.',
                     'Look into log-file for further information.']))
    
    Main(**arguments)
    
    sys.stderr.close()
    sys.stderr = sys.__stderr__