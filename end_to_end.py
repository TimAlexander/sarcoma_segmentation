from data.Custom_Transforms import BBoxCrop3D,get_patches,get_volume
from model.model_zoo import load_model

from dipy.align.reslice import reslice
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import os
import glob
import argparse
import copy
from scipy import ndimage as nd


from scipy import ndimage as nd
def resample(img, nshape=None, spacing=None, new_spacing=None, order=0, mode='constant'):
    """
        Change image resolution by resampling

        Inputs:
        - spacing (numpy.ndarray): current resolution
        - new_spacing (numpy.ndarray): new resolution
        - order (int: 0-5): interpolation order

        Outputs:
        - resampled image
        """
    if nshape is None:
        if spacing.shape[0]!=1:
            spacing = np.transpose(spacing)

        if new_spacing.shape[0]!=1:
            new_spacing = np.transpose(new_spacing)

        if np.array_equal(spacing, new_spacing):
            return img

        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / img.shape

    else:
        if img.shape == nshape:
            return img
        real_resize_factor = np.array(nshape, dtype=float) / np.array(img.shape, dtype=float)

    image = nd.interpolation.zoom(img, real_resize_factor.ravel(), order=order, mode=mode)

    return image

def z_standardize(volume):
 return (volume-np.mean(volume))/(np.std(volume))

def split_in_patches(volume,max_patch_size):
    """Splits volume in patches if HxWxD > max_patch_size**3
    """

    shape = np.array(volume.shape[-3:])
    divs = np.ceil(shape/np.array(max_patch_size)).astype(np.int8)
    while np.prod(shape/np.array(divs)) > max_patch_size**3:
        max_dim = np.argmax(shape/np.array(divs))
        divs[max_dim] += 1 #divide in max dimension
    print(f"New patch shape {shape/np.array(divs)}")
    dim1_pad,dim2_pad,dim3_pad = np.mod(volume.shape,divs)
    volume = np.pad(volume,((0,divs[0]-dim1_pad),(0,divs[1]-dim2_pad),(0,divs[2]-dim3_pad)))
    volume = get_patches(volume,divs,offset=(5,5,5))
    return volume,divs #P x H x W x D

def pad_to_original_size(cropped_version,original_version,padding,bbox):


    
    y_cropped,x_cropped,z_cropped = cropped_version.shape
    y,x,z = original_version.shape
    
    
    
    #Remove the area we included by the padding
    cropped_version = cropped_version[padding[0]:y_cropped-padding[1],padding[2]:x_cropped-padding[3],padding[4]:z_cropped-padding[5]]


        
    return np.pad(cropped_version,((bbox[0],y-bbox[1]-1),(bbox[2],x-bbox[3]-1),(bbox[4],z-bbox[5]-1))).astype('float64')


def preprocess(volume,bbox,padding,max_patch_size):
    """Preproccessing of the volume :
    1. Crops around the bounding box with padding indicating the additional paddings in each direction from the bounding box
    2. Resamples the volume to a isotropic voxel size of 1mm^3
    3. Z-score standarization
    4. Creating patches if HxWxD > max_patch_size**3

    Args:
        volume NifTi: MRI Scan
        bbox list: x_min,y_min,z_min,x_max,y_max,z_max
        padding list: additional padding in all 6 directions (If data from the volume is available the data is taken, if not 0 padding is used)
        max_patch_size int: Creating patches if HxWxD > max_patch_size**3

    Returns:
        dict: containing preprocessed volume and additional information
    """
    croppingObj = BBoxCrop3D(padding,False)

    volume_data= volume.get_fdata()
    print('Volume shape {}'.format(volume_data.shape))
    volume_data = croppingObj(bbox,volume_data)
    #volume_data = volume_data[bbox[0]:bbox[1]+1,bbox[2]:bbox[3]+1,bbox[4]:bbox[5]+1]
    print('Volume shape after cropping {}'.format(volume_data.shape))
    shape_before_resample = volume_data.shape
    affine = volume.affine
    zooms = volume.header.get_zooms()[:3]
    volume, resampled_affine = reslice(volume_data, affine, zooms, (1,1,1))
    volume = resample(volume_data,spacing=np.array(zooms),new_spacing=np.array([1,1,1]))
    print('Volume shape after resampling {}'.format(volume.shape))
    volume = z_standardize(volume)

    divs = [1,1,1]
    if volume.size > max_patch_size**3:
        volume,divs = split_in_patches(volume,max_patch_size) # P x H x W x D

    return {'volume':volume,'original_zooms':zooms,'resampled_affine':resampled_affine,'divs':divs,'shape_before_resample':shape_before_resample}


def postprocess(prediction,original_volume,original_affine,shape_before_resample,padding,bbox):
    """Postprocessing of the volume
    1. Resamples the prediction to the original voxel resolution
    2. Padding of the prediction s.t. it matches the dimensions of the original volume
    3. Creating Nifti

    Args:
        prediction np.array: segmentation prediction of the network
        original_volume np.array: original volume before preprocessing
        croppingIndicies dict: information about where the original volume was cropped -> retrieved in the preprocessing step
        resampled_affine np.array: affine from the resampled volume -> retrived in the preprocessing step
        original_zooms tuple: voxel resolutions of the original volume -> retrieved in the preprocessing step

    Returns:
        NifTi: segmentation prediction matching the original volumes resolution and dimensions
    """
    print('Prediction shape before resampling & padding {}'.format(prediction.shape))
    #resampled_prediction,prediction_affine = reslice(prediction,resampled_affine,(1,1,1),original_zooms,order=0)

    resampled_prediction = resample(prediction,nshape=shape_before_resample)
    print('Prediction shape after resampling  {}'.format(resampled_prediction.shape))
    padded_prediction = pad_to_original_size(resampled_prediction,original_volume,padding,bbox)
    print('Prediction shape after padding {}'.format(padded_prediction.shape))

    prediction = nib.Nifti1Image(padded_prediction,original_affine)

    return prediction



def main(args):

    patients = os.listdir(args.input_directory)
    p2 = []
    for patient in patients:
        if not ('label' in patient):
            p2.append(patient)
    patients = sorted(p2)

    print(patients)
    with open(args.bbox_file) as bf:
        bboxes = []
        for i, line in enumerate(bf):
            linelist= [float(k) for k in line.split(',')]
            bboxes.append(np.asarray(linelist[-6:]).astype(np.int32))


    model_folder = args.model_folder #'/home/tomovt/sarcoma_project/logs/3DResUNetFinal/'
    model_name  = args.model_name #"3DResUNet"
    hparams ={}
    hparams["f_maps"] = args.f_maps
    hparams["levels"] = args.levels
    hparams["residual_block"] = args.residual_block
    hparams["se_block"] = args.se_block
    hparams["attention"]= args.attention
    hparams["MHTSA_heads"] = args.MHTSA_heads
    hparams["MHGSA_heads"] = args.MHGSA_heads
    hparams["trilinear"] = args.trilinear
    hparams["MSSC"] = args.MSSC

    device = args.device
    device = torch.device(device)
    models = []

    model_files = glob.glob(model_folder+'/*/')
    model_files.sort()

    for model_file in model_files:
        model = load_model(model_name,hparams)
        model_ft = model.to(device)
        model_file = glob.glob(model_file+'*/bestmodel_best.tar')[0]
        checkpoint = torch.load(model_file, map_location=device)
        model_ft.load_state_dict(checkpoint['model_state_dict'])
        model_ft.eval()
        models.append(model_ft)
        print("Model restored from file:", model_file)
        



    for patient,bbox in zip(patients,bboxes):
        print("Predicting Patient{}".format(patient))
        volume = nib.load(args.input_directory+patient)
        original_affine = volume.affine
        original_volume = copy.deepcopy(volume.get_fdata())
        preprocessed = preprocess(volume,bbox,args.padding,args.max_patch_size)
        volume = preprocessed['volume']
        resampled_affine = preprocessed['resampled_affine']
        original_zooms = preprocessed['original_zooms']
        divs = preprocessed['divs']
        shape_before_resample = preprocessed["shape_before_resample"]

        if len(volume.shape) == 3:
            volume = np.expand_dims(np.expand_dims(volume,axis=0),axis=0) #create channel and batch axis
        elif len(volume.shape) == 4: #patches
            volume = np.expand_dims(volume,axis=1) #create channel axis

        # B x C x H x W x D
        x = torch.tensor(volume).to(device, dtype=torch.float)
        num_patches = x.size()[0]


        if num_patches > 1:
            x_chunks = x.split(2)
        else:
            x_chunks = [x]

        with torch.no_grad():
            ensemble_logits = []
            for model_ft in models:
                logit_maps = []
                for x_in in  x_chunks: #loop over patches
                   

                    logits = model_ft(x_in)
                    print(logits.size())
                    logit_maps.extend(logits.squeeze(1).detach())

                logits = torch.stack(logit_maps)
                ensemble_logits.append(logits)


            logits = torch.mean(torch.stack(ensemble_logits,dim=0),dim=0)
            y_pred = torch.argmax(F.softmax(logits,dim=1),dim=1).unsqueeze(0)


            if num_patches > 1: #Reconstruct volume if patches were used
                print("Reconstructing Volume")
                y_pred = torch.tensor(get_volume(y_pred.squeeze(0).detach().cpu().numpy(),divs=divs,offset=(5,5,5))).unsqueeze(0).unsqueeze(0)
                x = torch.tensor(get_volume(x.squeeze().detach().cpu().numpy(),divs=divs,offset=(5,5,5))).unsqueeze(0).unsqueeze(0)

        
        prediction = postprocess(y_pred.squeeze().to('cpu').numpy(),original_volume,original_affine,shape_before_resample,args.padding,bbox)   

        assert prediction.dataobj.shape == original_volume.shape, "Prediction shape not matching original one"
        nib.save(prediction,args.output_directory+patient.split('.')[0]+"-prediction.nii.gz")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_directory', help='Path to input directory')
    parser.add_argument('--bbox_file',help="Path to txt file containing bboxes")
    parser.add_argument('--output_directory',help='Path to output directory')
    parser.add_argument('--model_folder',help="path to directory of the model(s) .tar")
    parser.add_argument('--max_patch_size',help="Creating patches if HxWxD > max_patch_size**3",default=200)
    parser.add_argument('--padding',help="additional padding in all 6 directions",default=[10,10,10,10,10,10])
    parser.add_argument('--device',default='cuda:6')
    parser.add_argument('--model_name',help="Name of model",default="3DResUNet")
    parser.add_argument('--f_maps',default=8)
    parser.add_argument('--levels',default =4)
    parser.add_argument('--residual_block',default=True)
    parser.add_argument('--se_block',default="CSSE")
    parser.add_argument('--attention',default=True)
    parser.add_argument('--MHTSA_heads',default=4)
    parser.add_argument('--MHGSA_heads',default=0)
    parser.add_argument('--trilinear',default=True)
    parser.add_argument('--MSSC',default="None")
    

    args = parser.parse_args()
    main(args)