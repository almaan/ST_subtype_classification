#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 19:45:12 2018

@author: Alma Andersson 

Function And Example of Quick estimation of spot radius in pixel units.

"""
import pandas as pd
import numpy as np

def EstimateRadius(spt,x_num= 33,y_num = 35,ratio = 4.0,):
    """
    Estimates the number the number of pixels equivalent to the radius. This is done by averageing over distance between neighboring spot centers.
    Only those spots with a complete neighboorhood (no holes) are taken into consideration as to avoid overestimating distances if the spot data has been
    currated. The radius is estimated by dividing the average spot center distance by four, since the diameter of one spot is supposed to be one fourth of
    of the spot center distance.
    
    provide the size of the **full** array (before curation) as the number of spot in each spatial direction via the paramters x_num and y_num. 
    Data could be entered either as the full spot-dataframe as loaded from the spot-detector files. Or as grid coordinates for spots and their pixel center
    coordinates. Important to note is that the the coordinate vectors are given in the format (vertical_coordinates,horizontal_coordinates), that is
    x-coordinates are given as the second column and y-coordinates as the first.
    
    Arguments:
        
        - spt : Pandas DataFrame - DataFrame containing spot and pixel coordinates
        
        - x_num : Integer - Number of spots along x-axis. Default set to 33.

        - y_num : Integer - Number of spots along y-axis. Default set to 35.

        - ratio : Float or Integer - Value of spot_center_distance/spot_radius Default set to 4.0
        
    """
    
    
    crd = spt[['y','x']].values
   
    
    full_arr = np.ones((y_num,x_num))
    spot_arr = np.zeros((y_num,x_num))
    dead_arr = np.zeros((y_num,x_num))
    #mark where spots are in the provided sample
    spot_arr[crd[:,0]-1,crd[:,1]-1] = 1
    
    for ii in range(1,full_arr.shape[0]-1):
        for jj in range(1,full_arr.shape[1]-1):
            if np.sum(spot_arr[ii-1:ii+2,jj-1:jj+2]) != 9:
                #if spot does not have all 8 neighbors mark as "dead"
                #dead spots are not used in radius estimation
                dead_arr[ii,jj] = 2

    spot_arr = spot_arr - dead_arr

    X,Y = np.meshgrid(np.arange(1,x_num+1),np.arange(1,y_num+1))
    
    #get coordinates for "alive" spots to be used in estimation
    alive = spot_arr == 1
    good_coord = np.vstack((Y[alive],X[alive])).T

    av_d = 0
    div = 0
    
    for ii in range(good_coord.shape[0]):
        #get pixel coordinate for alive spot
        sptval =  spt[['pixel_y','pixel_x']].loc[(spt['y'] == crd[ii,0]) &  (spt['x'] == crd[ii,1])].values
        #get pixel values for neighboring spots. Only one direction is used to reduce redundancy due to neighboring spots
        neighval_y = spt[['pixel_y','pixel_x']].loc[(spt['y'] == crd[ii,0]+1) &  (spt['x'] == crd[ii,1])].values
        neighval_x = spt[['pixel_y','pixel_x']].loc[(spt['y'] == crd[ii,0]) &  (spt['x'] == crd[ii,1]+1)].values
        
        #number of neighboring spots. Should be 0 or 1.
        nums = np.min([neighval_x.shape[0],neighval_y.shape[0]])
        try:
            #copute average distance between to neighboring spots
            diff = (np.linalg.norm(sptval-neighval_y[0:nums,:]) + np.linalg.norm(sptval-neighval_x[0:nums,:]))/2.0
            av_d = av_d + diff
            div += 1
        except ValueError:
           continue
    
    #compute average distance based on all spots    
    av_d = av_d / div
    #get radius
    av_r = av_d / float(ratio)
    
    return av_r
