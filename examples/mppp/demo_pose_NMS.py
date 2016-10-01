# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:14:15 2016

@author: benjamin
"""

import os

def PCK_pose_NMS(filedir, matchThreds):
    
    os.chdir(filedir)
    #prepare data
    h5file = h5py.File("demo.h5",'r')
    preds = np.array(h5file['preds'])
    scores = np.array(h5file['scores'])
    indexs = [line.rstrip(' ').rstrip('\r').rstrip('\n') for line in open("index.txt")]
    
    #get bounding box sizes    
    bbox_file = h5py.File("demo_bbox.h5",'r')
    xmax=np.array(bbox_file['xmax']); xmin=bbox_file['xmin']; ymax=np.array(bbox_file['ymax']); ymin=bbox_file['ymin']
    widths=xmax-xmin; heights=ymax-ymin;
    alpha = 0.1
    Sizes=alpha*np.maximum(widths,heights)
    
    if (os.path.exists("PCK-pose-NMS") == False):
        os.mkdir("PCK-pose-NMS")    
    if (os.path.exists("PCK-pose-NMS/matchThreds{}".format(matchThreds)) == False):
        os.mkdir("PCK-pose-NMS/matchThreds{}".format(matchThreds))
    os.chdir("PCK-pose-NMS/matchThreds{}".format(matchThreds))
    NMS_preds = open("pred.txt",'w')
    NMS_scores = open("scores.txt",'w')
    NMS_index = open("index.txt",'w')
    num_human = 0
    
    #loop through images index
    for i in xrange(len(indexs)):
#    for i in [21]:
        index = indexs[i].split(' '); 
        img_name = index[0]; start = int(index[1])-1; end = int(index[2])-1;
        
        #initialize scores and preds coordinates
        img_preds = preds[start:end+1]; img_scores = np.mean(scores[start:end+1],axis = 1)
        img_ids = np.arange(end-start+1); ref_dists = Sizes[start:end+1]
        
        #do NMS by PCK
        pick = []
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
        
        #write the NMS result to files
        pick = [Id+start for Id in pick]        
        preds_pick = preds[pick]; scores_pick = scores[pick]
        for j in xrange(len(pick)):
            NMS_preds.write("{}".format(img_name))
            NMS_scores.write("{}".format(img_name))
            for point_id in xrange(16):
                NMS_preds.write("\t{}\t{}".format(preds_pick[j,point_id,0],preds_pick[j,point_id,1]))
                NMS_scores.write("\t{}".format(scores_pick[j,point_id,0]))
            NMS_preds.write("\n")
            NMS_scores.write("\n")
        NMS_index.write("{} {} {}\n".format(img_name, num_human+1, num_human + len(pick)))
        num_human += len(pick)
    
    NMS_preds.close();NMS_scores.close();NMS_index.close()
        
def PCK_match(pick_preds, all_preds,ref_dist):
    dist = np.sqrt(np.sum(np.square(pick_preds[np.newaxis,:]-all_preds),axis=2))
    num_match_keypoints = np.sum(dist/ref_dist <= 1,axis=1)
    return num_match_keypoints          

