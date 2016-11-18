# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:14:15 2016

@author: benjamin
"""

import os
import numpy as np
def PCK_pose_NMS(preds, scores, bboxes, matchThreds):
    #get bounding box sizes
    preds = np.array(preds)
    scores = np.array(scores)
    xmin=np.array([x[0] for x in bboxes]); ymin=np.array([x[1] for x in bboxes]); 
    xmax=np.array([x[2] for x in bboxes]); ymax=np.array([x[3] for x in bboxes]); 
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    Sizes=alpha*np.maximum(widths,heights)

    num_human = 0
        
    #initialize scores and preds coordinates
    img_preds = preds[:]; img_scores = np.mean(scores[:],axis = 1)
    img_ids = np.arange(len(bboxes)); ref_dists = Sizes[:]

    #do NMS by PCK
    pick = []
    NMS_preds = []
    NMS_scores = []
    while(img_scores.size != 0):
            
        #pick the one with highest score
        pick_id = np.argmax(img_scores)  
        pick.append(img_ids[pick_id])

        #get numbers of match keypoints by calling PCK_match 
        ref_dist=ref_dists[img_ids[pick_id]]
        num_match_points = PCK_match(img_preds[pick_id],img_preds,ref_dist)

        #delete humans who have more than matchThreds keypoints overlap with the seletced human.
        delete_ids = np.arange(img_scores.shape[0])[num_match_points > matchThreds]
        img_preds = np.delete(img_preds,delete_ids,axis=0); img_scores = np.delete(img_scores, delete_ids)
        img_ids = np.delete(img_ids, delete_ids); 

    #write the NMS result
    preds_pick = preds[pick]; scores_pick = scores[pick]
    for j in xrange(len(pick)):
        NMS_pred=[]
        NMS_score=[]
        for point_id in xrange(16):
            NMS_pred.append([preds_pick[j,point_id,0],preds_pick[j,point_id,1]])
            NMS_score.append(scores_pick[j,point_id,0])
        NMS_preds.append(NMS_pred)
        NMS_scores.append(NMS_score)
    
    return NMS_preds,NMS_scores
        
def PCK_match(pick_preds, all_preds,ref_dist):
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))
    num_match_keypoints = np.sum(dist/ref_dist <= 1,axis=1)
    return num_match_keypoints          

