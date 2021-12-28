import torch
import numpy as np
import math


def get_confmatrix(gt,preds,n_classes,include_background=True):
    start = 0
    if not include_background:
        start = 1

    conf_matrices = []
    for i in range(start,n_classes):
        y = gt[:,i,:]
        y_hat = preds[:,i,:]
        
        vec =  y - y_hat
        mask = vec==0
        tn = np.count_nonzero(mask & (y==0))
        tp = np.count_nonzero(mask & (y==1))
        fp = np.count_nonzero(vec==-1)
        fn = np.count_nonzero(vec==1)
        
        conf_matrices.append(np.array([[tp,fp],
                                      [fn,tn]]))
        
    
    return np.stack(conf_matrices,axis=0)

def dice_coefficient(confm,reduction="mean",eps=1e-8):
    dice = np.array([])
    for cl in range(confm.shape[0]):
        class_dice = (2*confm[cl,0,0])/(2*confm[cl,0,0]+confm[cl,0,1]+confm[cl,1,0]+eps) #(2TP/ 2TP+FP+FN)
        dice = np.append(dice,class_dice)
    if reduction == "mean":
        dice = np.mean(dice)
        
    return dice

def jaccard_index(confm,reduction="mean",eps=1e-8):
    jac = np.array([])
    for cl in range(confm.shape[0]):
        class_jac = confm[cl,0,0]/(confm[cl,0,0]+confm[cl,0,1]+confm[cl,1,0]+eps) #TP/ TP+FP+FN
        jac = np.append(jac,class_jac)
    if reduction == "mean":
        jac = np.mean(jac)
        
    return jac

def sensitivity_score(confm,reduction="mean",eps=1e-8):
    sensitivity = np.array([])
    for cl in range(confm.shape[0]):
        class_sensitivity = confm[cl,0,0]/(confm[cl,0,0]+confm[cl,1,0]+eps) #TP/ TP+FN
        sensitivity = np.append(sensitivity,class_sensitivity)
    if reduction == "mean":
        sensitivity = np.mean(sensitivity)
        
    return sensitivity


def specificity_score(confm,reduction="mean",eps=1e-8):
    specificity = np.array([])
    for cl in range(confm.shape[0]):
        class_specificity = confm[cl,1,1]/(confm[cl,1,1]+confm[cl,0,1]+eps) #TN/ TN+FP
        specificity = np.append(specificity,class_specificity)
    if reduction == "mean":
        specificity = np.mean(specificity)
        
    return specificity


def rocauc_score(confm,reduction="mean",eps=1e-8):

    #https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-015-0068-x
    rocauc = np.array([])
    for cl in range(confm.shape[0]):
        FPR = confm[cl,0,1]/(confm[cl,0,1]+confm[cl,1,1]+eps) #FP/ FP + TN
        FNR = confm[cl,1,0]/(confm[cl,1,0]+confm[cl,0,0]+eps) #FN / FN + TP
        class_auc = 1- (FPR+FNR)/2
        rocauc = np.append(rocauc,class_auc)
    if reduction == "mean":
        rocauc = np.mean(rocauc)
        
    return rocauc

def wall_distances(box1, box2):
    """
    Computes the wall distances between two 3D boxes
    :param box1: coordinates of boxA [xmin,ymin,zmin, xmax,ymax,zmax]
    :param box2: coordinates of box2 [xmin,ymin,zmin, xmax,ymax,zmax]
    :return: List of distances of same shape as the inputs
    """
    return [abs(box1[v] - box2[v]) for v in range(len(box1))]

def compute_distance(coord1, coord2):
    """
    Computes the euclidian distance between two points in 3D
    :param coord1: point with coordinates [x,y,z]
    :param coord2: point with coordinates [x,y,z]
    :return: the diastance betwen the two points scalar float
    """
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2)


def bbox_IoU(box1,box2):
    x1_min,y1_min,z1_min,x1_max,y1_max,z1_max = tuple(box1)
    x2_min,y2_min,z2_min,x2_max,y2_max,z2_max = tuple(box2)
    w1 = x1_max - x1_min
    h1 = y1_max - y1_min
    d1 = z1_max - z1_min
    w2 = x2_max - x2_min
    h2 = y2_max - y2_min
    d2 = z2_max - z2_min
    
    w_intersection = min(x1_min + w1, x2_min + w2) - max(x1_min, x2_min)

    h_intersection = min(y1_min + h1, y2_min + h2) - max(y1_min, y2_min)
    
    d_intersection = min(z1_min + d1, z2_min + d2) - max(z1_min, z2_min)

    if w_intersection <= 0 or h_intersection <= 0 or d_intersection <= 0: 

        return 0

    I = w_intersection * h_intersection * d_intersection

    U = w1 * h1 * d1 + w2 * h2 * d2 - I 

    return I / U
def bbox_Recall(box1,box2):
    x1_min,y1_min,z1_min,x1_max,y1_max,z1_max = tuple(box1)
    x2_min,y2_min,z2_min,x2_max,y2_max,z2_max = tuple(box2)
    w1 = x1_max - x1_min
    h1 = y1_max - y1_min
    d1 = z1_max - z1_min
    w2 = x2_max - x2_min
    h2 = y2_max - y2_min
    d2 = z2_max - z2_min
    
    w_intersection = min(x1_min + w1, x2_min + w2) - max(x1_min, x2_min)

    h_intersection = min(y1_min + h1, y2_min + h2) - max(y1_min, y2_min)
    
    d_intersection = min(z1_min + d1, z2_min + d2) - max(z1_min, z2_min)

    if w_intersection <= 0 or h_intersection <= 0 or d_intersection <= 0: 

        return 0

    TP = w_intersection * h_intersection * d_intersection
    
    P = w2 * h2 * d2
    

    return TP / P

def bbox_Precision(box1,box2):
    x1_min,y1_min,z1_min,x1_max,y1_max,z1_max = tuple(box1)
    x2_min,y2_min,z2_min,x2_max,y2_max,z2_max = tuple(box2)
    w1 = x1_max - x1_min
    h1 = y1_max - y1_min
    d1 = z1_max - z1_min
    w2 = x2_max - x2_min
    h2 = y2_max - y2_min
    d2 = z2_max - z2_min
    
    w_intersection = min(x1_min + w1, x2_min + w2) - max(x1_min, x2_min)

    h_intersection = min(y1_min + h1, y2_min + h2) - max(y1_min, y2_min)
    
    d_intersection = min(z1_min + d1, z2_min + d2) - max(z1_min, z2_min)

    if w_intersection <= 0 or h_intersection <= 0 or d_intersection <= 0: 

        return 0

    TP = w_intersection * h_intersection * d_intersection
    
    FN = w1 * h1 * d1 - TP
    

    return TP / (TP+FN)

def get_largest_component1d(array,window=5):

    #Get components by calculating the difference between neighbour values and filtering diff >1 out
    components = np.split(array,np.argwhere(np.diff(array) > window).flatten()+1)

    #Get largest component    
    largest_component = np.empty(0)
    for component in components:
        if largest_component.size < component.size:
            largest_component = component
            
    return largest_component

def IoU3D(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    zA = max(boxA[2], boxB[2])

    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])

    interArea = abs(max(0, xB - xA)) * abs(max(0, yB - yA)) * abs(max(0, zB - zA))

    if interArea == 0:
        return 0
    else:
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[3] - boxA[0]) * (boxA[4] - boxA[1]) * (
                boxA[5] - boxA[2] + 1))
        boxBArea = abs((boxB[3] - boxB[0]) * (boxB[4] - boxB[1]) * (boxB[5] - boxB[2]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou