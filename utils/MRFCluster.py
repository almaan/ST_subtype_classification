#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 09:55:38 2018

@author: Alma Andersson

"""
from sklearn.decomposition import LatentDirichletAllocation as LDA
from utils import EstimateRadius
from scipy.spatial.distance import cityblock

import numpy as np
import pandas as pd
import sys

import graph_tool.all as gt


import cv2

import logging
from multiprocessing import cpu_count

import time

class Grapher:
    """
    Object used to construct a graph based on the count matrix and spot data. 
    
    Arguments : 
        
        - original_coordinates - Coordinates of spots (unadjusted)
        - values - count matrix values (array)
    
    """
    def __init__(self,
                 original_coordinates,
                 values,
                 theta=0.1,
                 entropy=np.array([]),
                 logname = False,
                 verbose = True,
                 ):
        
        np.random.seed(int(time.time()))
        
        if isinstance(logname,str):
            self.logname = logname
        elif isinstance(logname,bool):
            self.logname = 'MRF.cluster.log'
            logging.basicConfig(filename=self.logname,
                            level=logging.INFO,
                            )
            
        self.logger = logging.getLogger('MRFClustering')
        
        if verbose:
            ch = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.ocoords = original_coordinates
        self.ncoords = self.ocoords - np.min(self.ocoords,axis=0)
        
        self.entropy = entropy
        self.theta = theta
        
        if isinstance(values,np.ndarray):
            self.vals = values
        else:
            self.logger.error(f'Provided Count Matrix is not numpy array. Exiting')
            sys.exit(0)
            
        
        
        self.loaded_image = False
        self.n_clusters = 2
        

        self.g = gt.Graph()
        self.g.add_vertex(n=self.ncoords.shape[0]+2)
        
        for v1 in range(1,self.ncoords.shape[0]+1):
            for v2 in range(1,self.ncoords.shape[0]+1):
                if v1 > v2:
                    if cityblock(self.ncoords[v1-1,:],self.ncoords[v2-1,:]) < 2:
                        self.g.add_edge(self.g.vertex(v1),
                                        self.g.vertex(v2))
                        
                        self.g.add_edge(self.g.vertex(v2),
                                        self.g.vertex(v1))
        
        self.src = 0
        self.tgt = self.g.num_vertices()-1

        elist_s = [ (x,y) for x,y in zip(np.zeros(self.ncoords.shape[0]),np.arange(1,self.tgt))]
        self.g.add_edge_list(elist_s)
        
        elist_t = [ (x,y) for x,y in zip(np.arange(1,self.tgt),np.ones(self.ncoords.shape[0])*self.tgt)]
        self.g.add_edge_list(elist_t)

        pos = self.g.new_vertex_property('vector<double>')
        
        for x in range(1,self.tgt-1):
            pos[x] = (self.ncoords[x,0],self.ncoords[x,1])
            
        pos[0] = (0,0)
        pos[self.tgt] = (np.max(self.ncoords,axis=None)+1,np.max(self.ncoords,axis=None)+1)
        self.g.vertex_properties['pos'] = pos

            
    def Load_Image_Information(self,
                               spots,
                               img_pth,
                               ):
        
        """
        Gives each vertex an intensity index based on the intensity of the pixels found within 
        spot_radius pixels. If no spots are in the neighborhood then we set the importance weight to theta
        
        Arguments:
            - img_pth : path to HE-image file
            - spt_pth : path to spot file. 
        
        """
        self.logger.info('Loading Image and reading Intensity Values')
                 
        self.loaded_image = True
        img = cv2.imread(img_pth)

        try:
            #Exctract "Hematoxylin" part of image
            img = extract_hematoxylin(img)
        except ValueError:
            #Extract "Hematoxylin" part of image for subsets of image due to memory limitations
            
            self.logger.info('Image too large. Partition into smaller segments for HE-transform')
            rlen = img.shape[0]
            clen = img.shape[1]
            l1 =  np.linspace(0,rlen,5,dtype=int)
            l2 =  np.linspace(0,clen,5,dtype=int)
            l1[-1] = rlen-1
            l2[-1] = clen -1
            
            for jj in range(l2.shape[0]-1):
                for ii in range(l1.shape[0]-1):
                    img[ii:ii+1,jj:jj+1,:] = extract_hematoxylin(img[ii:ii+1,jj:jj+1,:])
            
        
        
        spot_radius = int(EstimateRadius.EstimateRadius(spt=spots))
        self.spot_radius = spot_radius
        self.logger.info(' '.join(['Spot Radius Estimated to :',str(spot_radius)]))
        
        n_spots = self.ocoords.shape[0]
        intensity = np.ones(n_spots)*self.theta   
        
        #generate new vertex property for intensity values of image
        ivals = self.g.new_vertex_property('float')
        
        #image attributes
        std = np.std(img)
        mean_intensity = np.nanmean(img) / std
        
        for ii in range(n_spots):
            row = spots[(spots['x']==self.ocoords[ii,0]) & (spots['y']==self.ocoords[ii,1])]
            if row.values.any():
                x = row['pixel_x'].values
                y = row['pixel_y'].values
                
                if row.shape[0] > 1:
                    x = x[0]
                    y = y[0]
                    self.logger.error(f'Multiple spots assigned to coordinates : ({x:d},{y:d})')

                #Square area covering spot, centered on spot-center. 
                #Distance to each side is equal to radius of spot
                x_low = np.max((0,x-spot_radius))
                y_low = np.max((0,y-spot_radius))
                x_high = np.min((img.shape[0]-1,x+spot_radius)) + 1
                y_high = np.min((img.shape[1]-1,y+spot_radius)) + 1
                 
                try:
                    #Get average intensity over spot
                    intensity[ii] = np.nanmean(img[int(x_low):int(x_high),int(y_low):int(y_high)])
                    ivals[ii+1] = intensity[ii]/std
                    self.logger.info(f"Weighted intensity for vertex ({ii+1:d}) : {ivals[ii+1]:f}")
                except RuntimeWarning:
                    #set to mean if failure to compute average intensity
                    ivals[ii + 1] = mean_intensity
                    self.logger.error(f"Weighted intensity for vertex ({ii+1:d}) : {mean_intensity:f} ")
            else:
                ivals[ii+1] = mean_intensity
                self.logger.error(f"Weighted intensity for vertex ({ii+1:d}) : {mean_intensity:f}")

        self.g.vertex_properties['ivals'] = ivals
        self.logger.info('Completed loading image and computation intensity values')
        
                  
    def Initialize(self,
                   n_iterations = 25,
                   n_jobs = -1):
        
        """
        Initialize the MRFClustering procedure by generating a temporary classification
        based on the Latent Dirichlet Allocation algorithm.
        
        Arguments:
            
            - n_iterations : number of max-iterations to perform. Default 25.
            - n_jobs : number of processors to use. Default set to maximum available.
        
        """
        
        self.logger.info(' '.join(['Initialize LDA using', str(n_iterations), 'iterations']))
       
        n_cores = cpu_count()
        
        if n_jobs > n_cores:
            n_jobs = n_cores
            self.logger.info(f' '.join(['Specified number of cores exceed those available.',
                                        'Cores to be used are set to {:d}.'.format(n_cores)]))
        
        #LDA object instantiation. using default parameters except for n_jobs and max_iter.
        lda = LDA(n_components = self.n_clusters,
                  learning_method = 'online',
                  doc_topic_prior = 0.1, 
                  topic_word_prior = 0.1,
                  batch_size = 100,
                  max_iter = n_iterations,
                  n_jobs = n_jobs,
                  )
        
        try:        
            res = lda.fit(self.vals)
            self.dist = res.transform(self.vals)
            self.idx = np.argmax(self.dist,axis=1)
            logging.info('LDA completed Successfully')
        except Exception as e:
            logging.error(f' LDA failed : {e:s}')
        
    def Weight_Graph(self,):
        
        """
        Weigh the edges of the constructed graph based on intensity values if
        image is provided, otherwise static value theta is given to neighboring
        nodes.
        
        """
        
        logging.info('Generating Graph Capacities')
        
        cap = self.g.new_ep("float")
        self.g.edge_properties['cap'] = cap    
        self.g.reindex_edges()
    
        for node in range(1,self.tgt):
            for neigh in self.g.vertex(node).out_neighbors():
                if neigh < self.tgt and neigh > self.src:
                    
                    if self.idx[int(node)-1] == self.idx[int(neigh)-1]:
                        #Neighbor does not have same classification as node
                        cap[self.g.edge(int(node),int(neigh))] = 0.0
                    else:
                        #Neighbor has same classification as node
                        if self.loaded_image:
                            #Use RBF to adjust influence based on intensity if image is provided
                            weight = np.exp(-((self.g.vertex_properties['ivals'][int(node)]-self.g.vertex_properties['ivals'][int(neigh)]))**2)
                            if weight == np.nan:
                                #give zero weight to spots where ivals were not assigned
                                weight = 0.0                            

                            cap[self.g.edge(int(node),int(neigh))] = np.min((weight,1.0))*self.theta
                        else:
                            #use static influence if no image is provided
                            cap[self.g.edge(int(node),int(neigh))] = self.theta
                        
        for node in range(1,self.tgt):
            #Set unary potential to probability of belonging to each class               
            cap[self.g.edge(self.src,node)] = self.dist[node-1,0]
            cap[self.g.edge(node,self.tgt)] = self.dist[node-1,1]
            
        self.logger.info('Graph was successfuly weighted')
        
    
    def Cut_Graph(self,):
        """
        Performs a minumim s-t cut of the weighted graph using Boyokov Kolmogorov's algorithm.
        The two subgraphs will be given the label 0 or 1 representing tumor and non-tumor.
        If an entropy vector is provided, the set of spots with lowest entropy is set
        as the tumor class. If only an image is provided, darkest region will be
        taken as the tumor-class.
        
        """
        #generate max_flow
        res = gt.boykov_kolmogorov_max_flow(self.g,
                                            self.src, 
                                            self.tgt, 
                                            self.g.edge_properties['cap'])
        #cut graph
        part = gt.min_st_cut(self.g, 
                             self.src, 
                             self.g.edge_properties['cap'],
                             res)
        
        self.idx = np.array(part.a[1:-1])
        
        if self.loaded_image and self.entropy.sum() == 0.0:
        #if image but no entropy vector is provided, set darkes.
            mu = np.zeros(2)
            for ii in range(2):
                try:
                    mu[ii] = np.nanmean((np.array(self.g.vertex_properties['ivals'].a)[1:-1])[self.idx == ii])
                except RuntimeWarning:
                    mu = np.array([0,1])
                    continue
                
            pos = self.idx == np.argmax(mu)
            self.idx[pos] = 0
            self.idx[pos == False] = 1
            
        else:
            #if entropy vector is provided
            av_entropy = np.array([np.mean(self.entropy[self.idx == 0]),np.mean(self.entropy[self.idx==1])])
            pos = self.idx == np.argmin(av_entropy)
            self.idx[pos] = 0
            self.idx [pos == False] = 1
            
    def Cluster(self,n_iterations=100):
        self.Initialize(n_iterations)
        self.Weight_Graph()
        self.Cut_Graph()
        
        
def extract_hematoxylin(img):
    """
    Extracts the "Hematoxylin" part of an image, related to nucleus density.
    Largely based on the scripts provided found in the module HistomicsTK. 
    
    Arguments:
        
        - img : an array with height and width in first two dimensions and RGB color channels in third
    
    """
    def rgb_to_sda(img_rgb):
        """
        Convert from RGB to SDA space
        """
        img_rgb = img_rgb.astype(float) + 1
        img_rgb = np.minimum(img_rgb, bg)
        img_sda = -np.log(img_rgb/(bg)) * 255.0/np.log(bg)
        return img_sda

    def sda_to_rgb(img_sda,):
        """
        Convert from SDA to RGB space
        """
        img_rgb = bg ** (1 - img_sda / 255.)
        return img_rgb - bg

    #background
    bg = 255.0 
    #stain vectors
    he_mat = np.array([[ 1.48852734, -0.16263078,  0.51267147],
                       [-1.0808497 ,  1.1196527 , -0.29176186],
                       [-0.52656453, -0.1198253 ,  1.49085422]])

        
    oldshape = img.shape[0:2]
    #deconvolution in order to obtain hematoxylin and eosin concentrations
    img = img.reshape((-1, img.shape[-1])).T
    sda_fwd = rgb_to_sda(img)
    sda_deconv = np.dot(he_mat, sda_fwd)
    
    #extract contributions of each stain in RGB-space
    sda_inv = sda_to_rgb(sda_deconv)    
    img = sda_inv.T.reshape(oldshape[0],oldshape[1],3)
    img = img.astype(np.uint8)[:,:,0]
    
    return img
                        
