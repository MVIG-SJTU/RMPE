# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:14:15 2016

@author: benjamin
"""

import numpy as np
def pose_NMS(preds_noNMS, scores_noNMS, bbox):
    """
    Here we use parametric NMS 
    one might use another pose NMS algorithm following the same API
    """
    #parameters of parametric pose NMS
    delta1 = 0.01; delta2 = 2.08; mu = 2.08; gamma = 22.48
    preds, scores = parametric_pose_NMS(np.array(preds_noNMS), np.array(scores_noNMS), bbox, delta1,delta2,mu,gamma)
    return preds, scores

def parametric_pose_NMS(preds_noNMS, scores_noNMS, bbox, delta1,delta2,mu,gamma):
    scoreThreds = 0.2
    preds_NMS=[];  
    scores_NMS=[];

    #get bounding box sizes    
    bbox = np.array(bbox)
    xmin = bbox[:,0]; ymin = bbox[:,1];
    xmax = bbox[:,2];ymax = bbox[:,3];
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    #keypoint bounding box size
    Sizes=alpha*np.maximum(widths,heights)
    
    start = 0; end = bbox.shape[0]-1
    #initialize scores and preds coordinates
    img_preds = preds_noNMS; img_scores = np.mean(scores_noNMS,axis = 1)
    img_ids = np.arange(end-start+1); ref_dists = Sizes[start:end+1];keypoint_scores = scores_noNMS;
    
    #do NMS by parametric
    pick = []
    merge_ids = []
    while(img_scores.size != 0):
        #pick the one with highest score
        pick_id = np.argmax(img_scores)  
        pick.append(img_ids[pick_id])
        #get numbers of match keypoints by calling PCK_match 
        ref_dist=ref_dists[img_ids[pick_id]]
        simi = get_parametric_distance(pick_id,img_preds, keypoint_scores,ref_dist, delta1, delta2, mu)
        #delete humans who have more than matchThreds keypoints overlap with the seletced human.
        delete_ids = np.arange(img_scores.shape[0])[simi > gamma]
        if (delete_ids.size == 0):
            delete_ids = pick_id
        merge_ids.append(img_ids[delete_ids])
        img_preds = np.delete(img_preds,delete_ids,axis=0); img_scores = np.delete(img_scores, delete_ids)
        img_ids = np.delete(img_ids, delete_ids); keypoint_scores = np.delete(keypoint_scores,delete_ids,axis=0)
        
    # get NMS results
    pick = [Id+start for Id in pick] 
    merge_ids = [Id+start for Id in merge_ids]
    assert len(merge_ids) == len(pick)
    preds_pick = preds_noNMS[pick]; scores_pick = scores_noNMS[pick]
    num_pick = 0
    for j in xrange(len(pick)):
        # merge poses
        merge_id = merge_ids[j]  
        merge_poses,merge_score = merge_pose(preds_pick[j],preds_noNMS[merge_id],scores_noNMS[merge_id],Sizes[pick[j]])
        #if the mean score is too low, ignore this pose estimation
        ids = np.arange(16)
        if (merge_score[0] < 0.1): ids = np.delete(ids,0);
        if (merge_score[5] < 0.1): ids = np.delete(ids,5);
        mean_score = np.mean(merge_score[ids])
        if (mean_score < scoreThreds):
            continue
        #add the person to pred_NMS
        num_pick += 1
        pred_NMS = []
        score_NMS = []
        for point_id in xrange(16):
            pred_NMS.append([merge_poses[point_id,0],merge_poses[point_id,1]])
            score_NMS.append(merge_score[point_id])
        preds_NMS.append(pred_NMS)
        scores_NMS.append(score_NMS)
    return preds_NMS, scores_NMS
    
def merge_pose(refer_pose, cluster_preds, cluster_keypoint_scores, ref_dist):
    dist = np.sqrt(np.sum(np.square(refer_pose[np.newaxis,:]-cluster_preds),axis=2))
    # mask is an nx16 matrix
    mask = (dist <= ref_dist)
    final_pose = np.zeros([16,2]); final_scores = np.zeros(16)
    if (cluster_preds.ndim == 2):
        cluster_preds = cluster_preds[np.newaxis,:,:]
        cluster_keypoint_scores = cluster_keypoint_scores[np.newaxis,:]
    if (mask.ndim == 1):
        mask = mask[np.newaxis,:]
    for i in xrange(16):
        cluster_joint_scores = cluster_keypoint_scores[:,i][mask[:,i]]
        # pick the corresponding i's matched keyjoint locations and do an weighed sum.
        cluster_joint_location = cluster_preds[:,i,:][np.tile(mask[:,i,np.newaxis],(1,2))].reshape(np.sum(mask[:,i,np.newaxis]),-1)
        # get an normalized score
        normed_scores = cluster_joint_scores / np.sum(cluster_joint_scores)
        # merge poses by a weighted sum
        final_pose[i,0] = np.dot(cluster_joint_location[:,0], normed_scores)
        final_pose[i,1] = np.dot(cluster_joint_location[:,1], normed_scores)
        final_scores[i] = np.max(cluster_joint_scores)
    return final_pose, final_scores
    
def get_parametric_distance(i,all_preds, keypoint_scores,ref_dist, delta1, delta2, mu):
    pick_preds = np.array(all_preds[i])
    pred_scores = np.array(keypoint_scores[i])
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))/ref_dist
    mask = (dist <= 1)
    # define a keypoints distances
    score_dists = np.zeros([np.array(all_preds).shape[0], 16])
    keypoint_scores = np.squeeze(keypoint_scores)
    if (keypoint_scores.ndim == 1) :
        keypoint_scores = keypoint_scores[np.newaxis,:]
    # the predicted scores are repeated up to do boastcast
    pred_scores = np.tile(pred_scores, [1,np.array(all_preds).shape[0]]).T
    score_dists[mask] = np.tanh(pred_scores[mask]/delta1)*np.tanh(keypoint_scores[mask]/delta1)
    point_dist = np.exp((-1)*dist/delta2)
    final_dist = np.sum(score_dists,axis=1)+mu*np.sum(point_dist,axis=1)
    return final_dist


