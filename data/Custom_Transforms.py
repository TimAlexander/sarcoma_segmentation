import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform
from skimage.exposure import rescale_intensity
import math
import warnings






class BBoxCrop3D(object):

    
    def __init__(self,padding=[10,10,10,10,10,10],return_idx=False):

        self.padding = padding
        self.return_idx = return_idx
        
    def __call__(self,bbox,volume):
        
        #[x_min,y_min,z_min,x_max,y_max_z_max]
        top,bottom,left,right,front,back = bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],bbox[5]#bbox[0],bbox[3],bbox[1],bbox[4],bbox[2],bbox[5]
        
        
        t_pad = self.padding[0]
        b_pad = self.padding[1] 
        l_pad = self.padding[2] 
        r_pad = self.padding[3]
        f_pad = self.padding[4]
        bk_pad = self.padding[5]

        #Create new indicies
        new_top = top-t_pad
        new_bottom = bottom+b_pad
        new_left = left-l_pad
        new_right = right+r_pad
        new_front = front-f_pad
        new_back = back+bk_pad

        #Where can we use the new indicies and where do we have to pad because the image ends
        top_flag = new_top >= 0
        bottom_flag = new_bottom < volume.shape[0]
        left_flag = new_left >= 0
        right_flag = new_right < volume.shape[1]
        front_flag = new_front >= 0
        back_flag = new_back < volume.shape[2]

        #Cut of indicies if they are outside of the image and assign corresponding pads
        new_top,new_t_pad = (new_top,0) if top_flag else (0,t_pad-top)
        new_bottom,new_b_bad = (new_bottom,0) if bottom_flag else (volume.shape[0]-1,new_bottom-(volume.shape[0]-1))
        new_left,new_l_pad = (new_left,0) if left_flag else (0,l_pad-left)
        new_right,new_r_pad = (new_right,0) if right_flag else (volume.shape[1]-1,new_right-(volume.shape[1]-1))
        new_front,new_f_pad = (new_front,0) if front_flag else (0,f_pad-front)
        new_back,new_bk_pad = (new_back,0) if back_flag else (volume.shape[2]-1,new_back-(volume.shape[2]-1))
        
        if self.return_idx:
            return (np.pad(volume[new_top:new_bottom+1,new_left:new_right+1,new_front:new_back+1],((new_t_pad,new_b_bad),(new_l_pad,new_r_pad),(new_f_pad,new_bk_pad))),[new_top,new_bottom,new_left,new_right,new_front,new_back,new_t_pad,new_b_bad,new_l_pad,new_r_pad,new_f_pad,new_bk_pad])

        return np.pad(volume[new_top:new_bottom+1,new_left:new_right+1,new_front:new_back+1],((new_t_pad,new_b_bad),(new_l_pad,new_r_pad),(new_f_pad,new_bk_pad)))
    


def get_patches(volume, divs, offset):
    '''
    Args:
        - volume (np.array)         :   The volume to cut
                                        N Dimensions:
                                        single channel   : (X_1,..., X_N)
                                        multi channel    : (X_1,..., X_N, C)
        - divs (tuple)              :   Amount to divide each dimension
                                        len(divs) must be equal to N 
        - offset (tuple)            :   Offset for each div
                                        len(offset) must be equal to N
                                      
    Output:
        - patches (np.array)        :   patches stacked along first dimension
    '''
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)
    
    assert len(volume.shape) == len(divs) or len(volume.shape) == len(divs) + 1
    assert len(volume.shape) == len(offset) or len(volume.shape) == len(offset) + 1

    patches = []
    # simply iterate over all indices
    for idx in np.arange(np.prod(divs)):
        patches.append(get_patch(volume, idx, divs, offset))
    
    # TODO use stack
    return np.array(patches)


def get_patch(volume, index, divs=(2,2,2), offset=(6,6,6)):
    '''
    Args:
        - volume (np.array)         :   The volume to cut
                                        N Dimensions:
                                        single channel   : (X_1,..., X_N)
                                        multi channel    : (X_1,..., X_N, C)
        - index (int)               :   flattened patch iterator.
                                        in range 0 to prod(divs)-1
        - divs (tuple)              :   Amount to divide each dimension
                                        len(divs) must be equal to N 
        - offset (tuple)            :   Offset for each div
                                        len(offset) must be equal to N
                                        
    Output:
        - patch (np.array)          :   patch at index
    '''
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)
    
    assert len(volume.shape) == len(divs) or len(volume.shape) == len(divs) + 1
    assert len(volume.shape) == len(offset) or len(volume.shape) == len(offset) + 1
    
    
    
    if len(volume.shape) == len(divs) + 1:
        # multi channel
        shape = volume.shape[:-1]           
    else:
        # single channel
        shape = volume.shape
        
    if np.any(np.mod(shape, divs)):
        warnings.warn(('At least one dimension of the input volume can\'t be '
                       'divided by divs without remainder. Your input shape '
                       'and reconstructed shapes won\'t match.'))
        
    widths = [int(s/d) for s, d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths, offset)]
    
    # create nd index
    index_ = np.unravel_index(index, divs)
    
    # coordinates
    c = [s*d for s, d in zip(index_, widths)] 
    
    if len(volume.shape) == len(divs) + 1:
        patch_shape = tuple(patch_shape + [volume.shape[-1]])
    else:
        patch_shape = tuple(patch_shape)
        
    patch = np.zeros(patch_shape, dtype=volume.dtype)
    
    s_ = []
    e_ = []
    slice_idx = []
    slice_idx_patch = []
    # for every dimension X_1 ... X_N
    for dim in np.arange(len(c)):
        # calculate start and end index of the patch
        s_ = c[dim] - offset[dim] if c[dim] - offset[dim] >= 0 else 0
        e_ = c[dim] + widths[dim] + offset[dim] if \
            c[dim] + widths[dim] + offset[dim] <= shape[dim] else shape[dim]
        slice_idx.append(slice(s_, e_))
        
        # start and end index considering offset
        ps_ = offset[dim] - (c[dim] - s_)
        pe_ = ps_ + (e_ - s_)
        slice_idx_patch.append(slice(ps_, pe_))

    slice_idx = tuple(slice_idx)
    slice_idx_patch = tuple(slice_idx_patch)
    
    # cut out current patch
    vp = volume[slice_idx]
    
    # for offset
    patch[slice_idx_patch] = vp
    return patch


def get_volume(patches, divs = (2,2,3), offset=(6,6,6)):
    '''
    Args:
        - patches (np.array)         :  The patches to reconstruct. N_P patches
                                        are stacked along first dimension.
                                        single channel : (N_P, X_1,..., X_N)
                                        multi channel  : (N_P, X_1,..., X_N, C)
        - divs (tuple)              :   Amount to divide each dimension
                                        len(divs) must be equal to N 
        - offset (tuple)            :   Offset for each div
                                        len(offset) must be equal to N
                                      
    Output:
        - volume  (np.array)        :   patches reconstructed to volume
                                        single channel : (X_1,..., X_N)
                                        multi channel  : (X_1,..., X_N, C)
    '''
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)

    new_shape = [(ps -of*2)*int(d) \
                 for ps, of, d in zip(patches.shape[1:], offset, divs)]
    
    if len(patches.shape) == len(divs) + 2:
        # multi channel
        new_shape = tuple(new_shape + [patches.shape[-1]])
    else:
        # single channel
        new_shape = tuple(new_shape)
    
    volume = np.zeros(new_shape, dtype=patches.dtype)
    shape = volume.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    # iterate over patch indices
    for index in np.arange(np.prod(divs)):
        index_ = np.unravel_index(index, divs)
        slice_idx = []
        slice_idx_offs = []
        # iterate over dimension X_1 ... X_N
        for dim in np.arange(len(index_)):
            # calculate start and end index inside volume
            s_ = (index_[dim] * widths[dim])
            e_ = ((index_[dim] + 1) * widths[dim])
            slice_idx.append(slice(s_, e_))
            
            # calculate start and end index inside patch,
            # to ret rid of the offset
            ps_ = offset[dim]
            pe_ = offset[dim] + widths[dim]
            slice_idx_offs.append(slice(ps_, pe_))
            
        patch = patches[index,...]
        volume[tuple(slice_idx)] = patch[tuple(slice_idx_offs)]
    return volume


def create_Median_Patches(volume,median_patch_size):
    ##Creates Patches based on Median patch size with 50% overlap
    shape = np.array(volume.shape[-3:])
    divs = np.ceil(shape/np.array(median_patch_size)).astype(np.int8)
    if np.prod(divs) == 1:
        return volume,divs,(0,0,0)
    dim1_pad,dim2_pad,dim3_pad = np.mod(volume.shape,divs)
    volume = np.pad(volume,((0,divs[0]-dim1_pad),(0,divs[1]-dim2_pad),(0,divs[2]-dim3_pad)))  #Ensure get_patches is working
    target_shape = (volume.shape/divs).astype(np.int8)
    offset = (target_shape*(1/4)).astype(np.int8)
    volume = get_patches(volume,divs,offset)
    
    return volume,divs,offset

 