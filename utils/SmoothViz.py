#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Alma Andersson 2018-10-29

"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPC
from sklearn.preprocessing import LabelEncoder


from PIL import Image

def GenerateImage(center,
                  img_pth,
                  cat,
                  prob,
                  r=70.0):
    
    """
    use gaussian processes to visualize the classification and tumor extraction.
    
    Arguments: 
        - center : (numpy array) pixel-center coordinates
        - img_pth : (str) path to HE-image
        - cat : (numpy array) class assigned to each spot, same order as center
        - prob : (numpy array) probability vectors/matrix from classification
        - r: (float) radius of spot
        
    Output: 
        
        - simg : (PIL image) image with colors corresponding to each subtype
    
    """
    
    img = Image.open(img_pth)
    
    center = np.floor(center/r)
    center = center.astype(int)
    
    new_size = [np.round(x/r,0).astype(int) for x in img.size]
    scaling_factor = [float(img.size[0])/float(new_size[0]),float(img.size[1])/float(new_size[1])]
    
    rgb_values =  dict(luma = (73,216,166),
                       lumb = (12,138,153),
                       her2lum = (255,124,148),
                       tnbc=(122,172,73),
                       her2nonlum = (255,202,79),
                       nontumor=(66,66,66))

    simg = np.ones(shape = (new_size[0],new_size[1],3),dtype=np.uint8)*255
    coordinates = []
    tmp_coordinates = np.array([[x,y] for x in range(0, simg.shape[0]) for y in range(0, simg.shape[1])])
    
    del img
    
    for ii in range(tmp_coordinates.shape[0]):
        if np.min(np.linalg.norm(center - tmp_coordinates[ii,:],axis = 1)) <= 3.0:
            coordinates.append(tmp_coordinates[ii,:])
    coordinates = np.array(coordinates)
    
    gpc = GPC(kernel = None,
              optimizer = 'fmin_l_bfgs_b',
              n_restarts_optimizer=3)
    
    prob_adj = np.zeros((prob.shape[0],prob.shape[1]+1))
    prob_adj[:,0:prob.shape[1]] = prob
    prob_adj[cat == 'nontumor',0:prob.shape[1]] = 0.0
    prob_adj[cat == 'nontumor',-1] = 1.0
    
    fitted_gpc = gpc.fit(center,prob_adj)
    res = fitted_gpc.predict(coordinates)
    res = np.argmax(res,axis=1)
    
    subtypes = np.array(['luma','lumb','her2nonlum','her2lum','tnbc'])
    l1 = LabelEncoder()
    l1.fit(subtypes)
    
    backmap = np.append(l1.transform(subtypes),subtypes.shape[0])
    subtypes = np.append(subtypes,'nontumor')
    
    prob_to_cat = {backmap[ii]:subtypes[ii] for ii in range(subtypes.shape[0])}
    res = np.array(list(map(lambda x: prob_to_cat[x],res)))
    
    for ii in range(res.shape[0]):
        simg[coordinates[ii,0],coordinates[ii,1],:] = rgb_values[res[ii]]
    
    simg = Image.fromarray(simg,mode = 'RGB')
    simg = simg.transpose(method = Image.ROTATE_90)
    simg = simg.transpose(method = Image.FLIP_TOP_BOTTOM)
    simg = simg.resize((int(scaling_factor[0]*new_size[0]),int(scaling_factor[1]*new_size[1])),Image.ANTIALIAS)

    return simg
