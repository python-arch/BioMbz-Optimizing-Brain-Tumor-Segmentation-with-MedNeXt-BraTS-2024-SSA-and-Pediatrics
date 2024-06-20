import os
import copy
import numpy as np
import torch
import cc3d
import scipy.ndimage
import random
import string
import pickle

from pathlib import Path
X = 0
from monai.transforms import (
    MapTransform,
    AsDiscrete,
    FillHoles,
    RemoveSmallObjects,
)

class GliomaLabelCorrection:
    def __call__(self, x):
        x[0][x[2] == 1] = 1.
        x[1][x[0] == 1] = 1.
        
        return x

class AdvancedAsDiscrete(MapTransform):
    def __init__(self, tc_threshold, wt_threshold, et_threshold):
        self.tc_threshold = tc_threshold
        self.wt_threshold = wt_threshold
        self.et_threshold = et_threshold
    
    def __call__(self, x):
        if isinstance(x['prob'], torch.Tensor):
            x['prob'] = x['prob'].numpy()
        y = x['prob'].copy()
        
        y[0] = (y[0] >= self.tc_threshold).astype(np.uint8)
        y[1] = (y[1] >= self.wt_threshold).astype(np.uint8)
        y[2] = (y[2] >= self.et_threshold).astype(np.uint8)
        
        return {'prob': x['prob'], 'pred': y, 'mri': x['mri'], 'filename': x['filename']}

class AdvancedFilterObjectsSingleChannel(MapTransform):
    def __init__(self, channel, min_size_up, min_size_low, min_prob_up, min_prob_mid, max_n_mid, connectivity=26, save_dir=None):
        assert min_size_up >= min_size_low
        assert min_prob_mid >= 0. and min_prob_mid < 1.
        assert min_prob_up >= 0. and min_prob_up < 1.
        # assert min_prob_up <= min_prob_mid
        assert max_n_mid > 0
        
        self.channel = channel
        self.min_size_up = min_size_up
        self.min_size_low = min_size_low
        self.min_prob_up = min_prob_up
        self.min_prob_mid = min_prob_mid
        self.max_n_mid = max_n_mid
        self.connectivity = connectivity
        self.save_dir = save_dir
        
        if self.save_dir:
            Path(os.path.join(self.save_dir, str(self.channel))).mkdir(parents=True, exist_ok=True)
    
    def __call__(self, x):
        prob, pred = x['prob'], x['pred']
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        yprob = copy.deepcopy(prob[self.channel])
        ypred = copy.deepcopy(pred[self.channel])
        if isinstance(yprob, torch.Tensor):
            yprob = yprob.numpy()
            ypred = ypred.numpy()
        
        y_cc = cc3d.connected_components(ypred, connectivity=self.connectivity)
        
        list_vol_cc = [[ycomp, np.sum(y_cc == ycomp)] for ycomp in range(1, 1 + np.max(y_cc))]
        list_vol_cc = sorted(list_vol_cc, key = lambda x: x[1], reverse=True)
        
        list_info_obj = []
        y_post = np.zeros_like(ypred)
        n_mid = 0
        # if self.channel == 2:
        #     print('-'*30)
        #     print(len(list_vol_cc), np.unique(y_cc), list_vol_cc)
        for ycomp, vol in list_vol_cc:
            mean = np.mean(yprob[y_cc == ycomp])
            list_info_obj.append({'ycomp': ycomp, 'vol': vol, 'mean_prob': mean})
            # if self.channel == 2: print(ycomp, vol, mean, n_mid)
            
            if vol < self.min_size_low:
                continue
            
            if vol >= self.min_size_up:
                if mean >= self.min_prob_up:
                    y_post[y_cc == ycomp] = 1
            else:
                if (mean >= self.min_prob_mid) and (n_mid < self.max_n_mid):
                    y_post[y_cc == ycomp] = 1
                    n_mid += 1
            
            # print(ycomp, vol, mean)
        
        pred[self.channel] = y_post
        
        if self.save_dir:
            filename = x['filename'] + '.pickle'
            path = os.path.join(self.save_dir, str(self.channel), filename)
            self.save_pickle(list_info_obj, path)
        
        return {'prob': prob, 'pred': pred, 'filename': x['filename']}
    
    @staticmethod
    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str
    
    @staticmethod
    def save_pickle(data, path):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)

class RemoveSmallObjectsSingleChannelWithDilation:
    def __init__(self, min_size=64, connectivity=1, channel=0, dilation_factor=None):
        self.min_size = min_size
        self.connectivity = connectivity
        self.channel = channel
        self.dilation_factor = dilation_factor
        
        self.dilation_struct = None
        if dilation_factor:
            self.dilation_struct = scipy.ndimage.generate_binary_structure(3, 2)
    
    def __call__(self, x):
        assert len(x.shape) == 4
        
        tmp = copy.deepcopy(x)
        
        y = copy.deepcopy(x[self.channel])
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        
        y_cc = cc3d.connected_components(y, connectivity=self.connectivity)
        
        if self.dilation_struct is not None:
            y_dilation = scipy.ndimage.binary_dilation(y, structure=self.dilation_struct, iterations=self.dilation_factor)
            y_dilation_cc = cc3d.connected_components(y_dilation, connectivity=self.connectivity)
            
            y_cc_comb = self.get_GTseg_combinedByDilation(
                gt_dilated_cc_mat = y_dilation_cc,
                gt_label_cc = y_cc,
            )
        else:
            y_cc_comb = y_cc
        
        y_post = np.zeros_like(y)
        for ycomp in range(1, 1 + np.max(y_cc_comb)):
            if np.sum(y_cc_comb == ycomp) >= self.min_size:
                y_post[y_cc_comb == ycomp] = 1
        
        y_post = torch.from_numpy(y_post)
        
        tmp[self.channel] = y_post
        
        return tmp
    
    @staticmethod
    def get_GTseg_combinedByDilation(gt_dilated_cc_mat, gt_label_cc):
        """
        Computes the Corrected Connected Components after combing lesions
        together with respect to their dilation extent
        
        Parameters
        ==========
        gt_dilated_cc_mat: Numpy Array/Matrix; Ground Truth Dilated Segmentation 
                        after CC Analysis
        gt_label_cc: Numpy Array/Matrix; Ground Truth Segmentation after 
                        CC Analysis
        
        Output
        ======
        gt_seg_combinedByDilation_mat: Numpy Array/Matrix; Ground Truth 
                                    Segmentation after CC Analysis and 
                                    combining lesions
        """    
        
        gt_seg_combinedByDilation_mat = np.zeros_like(gt_dilated_cc_mat)
        
        for comp in range(np.max(gt_dilated_cc_mat)):  
            comp += 1
            
            gt_d_tmp = np.zeros_like(gt_dilated_cc_mat)
            gt_d_tmp[gt_dilated_cc_mat == comp] = 1
            gt_d_tmp = (gt_label_cc*gt_d_tmp)
            
            np.place(gt_d_tmp, gt_d_tmp > 0, comp)
            gt_seg_combinedByDilation_mat += gt_d_tmp
            
        return gt_seg_combinedByDilation_mat


class RemoveSmallObjectsSingleChannel(RemoveSmallObjects):
    def __init__(self, min_size=64, connectivity=1, channel=0):
        super().__init__(min_size=min_size, connectivity=connectivity, independent_channels=True)
        self.channel = channel
    
    def __call__(self, x):
        assert len(x.shape) == 4
        
        y = copy.deepcopy(x)
        y[self.channel:self.channel+1] = super().__call__(y[self.channel:self.channel+1])
        
        return y

class ReplaceETToTC:
    def __init__(self, threshold=100):
        self.threshold = threshold
    
    def __call__(self, x):
        assert len(x.shape) == 4
        
        y = copy.deepcopy(x)
        
        # Replace ET with TC if detected ET is very small
        if y[2].sum() >= self.threshold:
            return y
        
        y[0] = torch.where((y[0] == 1) | (y[2] == 1), 1, 0)
        y[2] = 0
        
        return y

class ToDiscreteWithReplacingSmallET:
    def __init__(self, threshold=0.5, min_et=500, min_tc=500):
        self.as_discrete = AsDiscrete(argmax=False, threshold=threshold)
        self.min_et = min_et
        self.min_tc = min_tc
    
    def __call__(self, y_prob):
        assert len(y_prob.shape) == 4
        
        y_pred = self.as_discrete(y_prob)
        
        # Replace ET with TC if detected ET is very small
        if y_pred[2].sum() >= self.min_et:
            return y_pred
        
        y_pred[0] = torch.where((y_pred[0] == 1) | (y_pred[2] == 1), 1, 0)
        y_pred[2] = 0
        
        # Replace TC with WT if detected TC is very small
        if y_pred[0].sum() >= self.min_tc:
            return y_pred
        
        y_pred[1] = torch.where((y_pred[1] == 1) | (y_pred[0] == 1), 1, 0)
        y_pred[0] = 0
        
        return y_pred



# class FillHolesChannelWise(FillHoles):
#     def __call__(self, x):
#         assert len(x.shape) == 4
        
#         y = copy.deepcopy(x)
#         for c in range(y.shape[0]):
#             y[c:c+1] = super().__call__(y[c:c+1])
        
#         return y

class AdvancedFilterObjectsSingleChannelV2(MapTransform):
    def __init__(self, channel, min_size, min_prob, max_objects, sorted_by='size', connectivity=26, save_dir=None):
        assert min_prob >= 0. and min_prob < 1.
        assert max_objects > 0
        assert sorted_by in ['size', 'mean_prob']
        
        self.channel = channel
        self.min_size = min_size
        self.min_prob = min_prob
        self.max_objects = max_objects
        self.sorted_by = sorted_by
        self.connectivity = connectivity
        self.save_dir = save_dir
        
        if self.save_dir:
            # print(channel, min_size, min_prob, max_objects, connectivity)
            Path(os.path.join(self.save_dir, str(self.channel))).mkdir(parents=True, exist_ok=True)
    
    def __call__(self, x):
        prob, pred = x['prob'], x['pred']
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        yprob = copy.deepcopy(prob[self.channel])
        ypred = copy.deepcopy(pred[self.channel])
        if isinstance(yprob, torch.Tensor):
            yprob = yprob.numpy()
            ypred = ypred.numpy()
        
        y_cc = cc3d.connected_components(ypred, connectivity=self.connectivity)
        
        list_vol_cc = [[ycomp, np.sum(y_cc == ycomp), np.mean(yprob[y_cc == ycomp])] for ycomp in range(1, 1 + np.max(y_cc))]
        if self.sorted_by == 'size':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[1], reverse=True)
        elif self.sorted_by == 'mean_prob':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[2], reverse=True)
        # print(list_vol_cc)
        list_info_obj = []
        y_post = np.zeros_like(ypred)
        n_obj = 0
        
        for ycomp, vol, mean in list_vol_cc:
            list_info_obj.append({'ycomp': ycomp, 'vol': vol, 'mean_prob': mean})
            
            if vol < self.min_size:
                continue
            
            if mean < self.min_prob:
                continue
            
            y_post[y_cc == ycomp] = 1
            n_obj += 1
            
            if n_obj >= self.max_objects:
                break
        
        pred[self.channel] = y_post
        
        if self.save_dir:
            filename = x['filename'] + '.pickle'
            path = os.path.join(self.save_dir, str(self.channel), filename)
            self.save_pickle(list_info_obj, path)
        
        return {'prob': prob, 'pred': pred, 'filename': x['filename']}
    
    @staticmethod
    def get_random_string(length):
        # choose from all lowercase letter
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(length))
        return result_str
    
    @staticmethod
    def save_pickle(data, path):
        with open(path, 'wb') as handle:
            pickle.dump(data, handle)


class AdvancedFilterObjectsSingleChannelAfter(AdvancedFilterObjectsSingleChannelV2):
    def __init__(self, channel, min_size, min_prob, max_objects, sorted_by='size', connectivity=26, wt_channel=1, save_dir=None):
        super().__init__(channel, min_size, min_prob, max_objects, sorted_by, connectivity, save_dir)
        self.wt_channel = wt_channel
    
    def __call__(self, x):
        if x['pred'][self.wt_channel].sum() > 0:
            return x
        else:
            global X
            X += 1
            # print(X)
            print(x['filename'], x['pred'][self.channel].sum(), x['prob'][self.channel].max(), x['prob'][self.wt_channel].max())
            x = super().__call__(x)
            # print(x['pred'][self.wt_channel].sum())
            return x

class AdvanceETPost(MapTransform):
    def __init__(self, et_channel, min_size, min_prob, max_objects, sorted_by='size', connectivity=26):
        assert min_prob >= 0. and min_prob < 1.
        assert max_objects > 0
        assert sorted_by in ['size', 'mean_prob']
        
        self.et_channel = et_channel
        self.min_size = min_size
        self.min_prob = min_prob
        self.max_objects = max_objects
        self.sorted_by = sorted_by
        self.connectivity = connectivity
    
    def __call__(self, x):
        prob, pred = x['prob'], x['pred']
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        yprob = copy.deepcopy(prob[self.et_channel])
        ypred = copy.deepcopy(pred[self.et_channel])
        if isinstance(yprob, torch.Tensor):
            yprob = yprob.numpy()
            ypred = ypred.numpy()
        
        y_cc = cc3d.connected_components(ypred, connectivity=self.connectivity)
        
        list_vol_cc = [[ycomp, np.sum(y_cc == ycomp), np.mean(yprob[y_cc == ycomp])] for ycomp in range(1, 1 + np.max(y_cc))]
        if self.sorted_by == 'size':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[1], reverse=True)
        elif self.sorted_by == 'mean_prob':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[2], reverse=True)
        
        list_info_obj = []
        y_post = np.zeros_like(ypred)
        n_obj = 0
        
        for ycomp, vol, mean in list_vol_cc:
            list_info_obj.append({'ycomp': ycomp, 'vol': vol, 'mean_prob': mean})
            
            if vol < self.min_size:
                continue
            
            if mean < self.min_prob:
                continue
            
            y_post[y_cc == ycomp] = 1
            n_obj += 1
            
            if n_obj >= self.max_objects:
                break
        
        pred[self.et_channel] = y_post
        
        return {'prob': prob, 'pred': pred, 'mri': x['mri'], 'filename': x['filename']}

class AdvanceTCPost(MapTransform):
    def __init__(self, tc_channel, et_channel, min_size, min_prob, max_objects, sorted_by='size', connectivity=26):
        assert min_prob >= 0. and min_prob < 1.
        assert max_objects > 0
        assert sorted_by in ['size', 'mean_prob']
        
        self.tc_channel = tc_channel
        self.et_channel = et_channel
        self.min_size = min_size
        self.min_prob = min_prob
        self.max_objects = max_objects
        self.sorted_by = sorted_by
        self.connectivity = connectivity
    
    def __call__(self, x):
        prob, pred = x['prob'], x['pred']
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        yprob = copy.deepcopy(prob[self.tc_channel])
        ypred = copy.deepcopy(pred[self.tc_channel])
        if isinstance(yprob, torch.Tensor):
            yprob = yprob.numpy()
            ypred = ypred.numpy()
        
        ypred = ypred + x['pred'][self.et_channel]
        ypred[ypred > 0.] = 1.
        
        y_cc = cc3d.connected_components(ypred, connectivity=self.connectivity)
        et_cc = cc3d.connected_components(x['pred'][self.et_channel], connectivity=self.connectivity)
        
        list_vol_cc = [[ycomp, np.sum(y_cc == ycomp), np.mean(yprob[y_cc == ycomp])] for ycomp in range(1, 1 + np.max(y_cc))]
        if self.sorted_by == 'size':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[1], reverse=True)
        elif self.sorted_by == 'mean_prob':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[2], reverse=True)
        
        list_info_obj = []
        y_post = np.zeros_like(ypred)
        n_obj = 0
        
        for ycomp, vol, mean in list_vol_cc:
            list_info_obj.append({'ycomp': ycomp, 'vol': vol, 'mean_prob': mean})
            
            
            includes_et = False
            tc_obj = np.float32(y_cc == ycomp)
            for et_comp in range(1, 1 + np.max(et_cc)):
                et_obj = np.float32(x['pred'][self.et_channel] == et_comp)
                if np.sum(et_obj) == 0:
                    continue
                
                et_iou = np.sum(tc_obj * et_obj) / np.sum(et_obj)
                
                assert et_iou <= 1.
                if et_iou >= 0.5:
                    includes_et = True
                    break
            
            if includes_et == False:
                if vol < self.min_size:
                    continue
                if mean < self.min_prob:
                    continue
            
            y_post[y_cc == ycomp] = 1
            n_obj += 1
            
            if n_obj >= self.max_objects:
                break
        
        pred[self.tc_channel] = y_post
        
        return {'prob': prob, 'pred': pred, 'mri': x['mri'], 'filename': x['filename']}

class AdvanceWTPost(MapTransform):
    def __init__(self, wt_channel, tc_channel, et_channel, min_size, min_prob, max_objects, sorted_by='size', connectivity=26):
        assert min_prob >= 0. and min_prob < 1.
        assert max_objects > 0
        assert sorted_by in ['size', 'mean_prob']
        
        self.wt_channel = wt_channel
        self.tc_channel = tc_channel
        self.et_channel = et_channel
        self.min_size = min_size
        self.min_prob = min_prob
        self.max_objects = max_objects
        self.sorted_by = sorted_by
        self.connectivity = connectivity
    
    def __call__(self, x):
        prob, pred = x['prob'], x['pred']
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        yprob = copy.deepcopy(prob[self.wt_channel])
        ypred = copy.deepcopy(pred[self.wt_channel])
        if isinstance(yprob, torch.Tensor):
            yprob = yprob.numpy()
            ypred = ypred.numpy()
        
        ypred = ypred + x['pred'][self.et_channel]
        ypred = ypred + x['pred'][self.tc_channel]
        ypred[ypred > 0.] = 1.
        
        y_cc = cc3d.connected_components(ypred, connectivity=self.connectivity)
        tc_cc = cc3d.connected_components(x['pred'][self.tc_channel], connectivity=self.connectivity)
        
        list_vol_cc = [[ycomp, np.sum(y_cc == ycomp), np.mean(yprob[y_cc == ycomp])] for ycomp in range(1, 1 + np.max(y_cc))]
        if self.sorted_by == 'size':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[1], reverse=True)
        elif self.sorted_by == 'mean_prob':
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[2], reverse=True)
        
        list_info_obj = []
        y_post = np.zeros_like(ypred)
        n_obj = 0
        
        for ycomp, vol, mean in list_vol_cc:
            list_info_obj.append({'ycomp': ycomp, 'vol': vol, 'mean_prob': mean})
            
            includes_tc = False
            wt_obj = np.float32(y_cc == ycomp)
            for tc_comp in range(1, 1 + np.max(tc_cc)):
                tc_obj = np.float32(x['pred'][self.tc_channel] == tc_comp)
                if np.sum(tc_obj) == 0:
                    continue
                
                tc_iou = np.sum(wt_obj * tc_obj) / np.sum(tc_obj)
                
                assert tc_iou <= 1.
                
                if tc_iou >= 0.5:
                    includes_tc = True
                    break
            
            if includes_tc == False:
                if vol < self.min_size:
                    continue
                if mean < self.min_prob:
                    continue
            
            y_post[y_cc == ycomp] = 1
            n_obj += 1
            
            if n_obj >= self.max_objects:
                break
        
        pred[self.wt_channel] = y_post
        
        return {'prob': prob, 'pred': pred, 'mri': x['mri'], 'filename': x['filename']}

class PostPost(MapTransform):
    def __init__(self, wt_channel, tc_channel, et_channel, min_et_size, connectivity=26):
        
        self.wt_channel = wt_channel
        self.tc_channel = tc_channel
        self.et_channel = et_channel
        self.min_et_size = min_et_size
        self.connectivity = connectivity
    
    def __call__(self, x):
        prob, pred = x['prob'], x['pred']
        if x['pred'].sum() > 0:
            return x
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        prob = copy.deepcopy(prob)
        pred = copy.deepcopy(pred)
        if isinstance(pred, torch.Tensor):
            prob = prob.numpy()
            pred = pred.numpy()
        
        th = prob.max() / 2.
        pred[prob >= th] = 1.
        pred[prob <  th] = 0.
        
        y_post = np.zeros_like(pred)
        
        def func(ipred):
            cc = cc3d.connected_components(ipred, connectivity=self.connectivity)
            list_vol_cc = [[ycomp, np.sum(cc == ycomp), np.mean(prob[self.et_channel][cc == ycomp])] for ycomp in range(1, 1 + np.max(cc))]
            list_vol_cc = sorted(list_vol_cc, key = lambda x: x[1], reverse=True)
            return cc, list_vol_cc
        
        et_cc, list_vol_et_cc = func(pred[self.et_channel])
        for ycomp, vol, mean in list_vol_et_cc[:1]:
            if vol < self.min_et_size: continue
            y_post[self.et_channel][et_cc == ycomp] = 1
        
        pred[self.tc_channel] = pred[self.tc_channel] + pred[self.et_channel]
        pred[self.tc_channel][pred[self.tc_channel] > 0.] = 1
        
        tc_cc, list_vol_tc_cc = func(pred[self.tc_channel])
        for ycomp, vol, mean in list_vol_tc_cc[:1]:
            y_post[self.tc_channel][tc_cc == ycomp] = 1
        
        pred[self.wt_channel] = pred[self.wt_channel] + pred[self.tc_channel]
        pred[self.wt_channel][pred[self.wt_channel] > 0.] = 1
        
        wt_cc, list_vol_wt_cc = func(pred[self.wt_channel])
        for ycomp, vol, mean in list_vol_wt_cc[:1]:
            y_post[self.wt_channel][wt_cc == ycomp] = 1
        
        pred = y_post
        print(x['filename'], pred[2].sum(), pred[0].sum(), pred[1].sum(), prob[2].max(), prob[0].max(), prob[1].max())
        return {'prob': prob, 'pred': pred, 'mri': x['mri'], 'filename': x['filename']}

class PostCheat(MapTransform):
    def __init__(self, wt_channel, tc_channel, et_channel, spacing):
        
        self.wt_channel = wt_channel
        self.tc_channel = tc_channel
        self.et_channel = et_channel
        self.spacing = spacing
    
    def __call__(self, x):
        prob, pred = x['prob'], x['pred']
        if x['pred'].sum() > 0:
            return x
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        prob = copy.deepcopy(prob)
        pred = copy.deepcopy(pred)
        y_post = np.zeros_like(pred)
        
        size = prob.shape[1:]
        
        def func(size):
            y = np.zeros(size)
            y[::self.spacing] = 1
            y[:, ::self.spacing] = 1
            y[:, :, ::self.spacing] = 1
            
            cc = cc3d.connected_components(y, connectivity=6)
            
            if not cc.max() == 1:
                print('Hmm', cc.max())
            
            return y
        
        grid = func(size)
        y_post[0] = grid
        y_post[1] = grid
        y_post[2] = grid
        
        pred = y_post
        
        print(x['filename'], pred[2].sum(), pred[0].sum(), pred[1].sum(), prob[2].max(), prob[0].max(), prob[1].max())
        
        return {'prob': prob, 'pred': pred, 'filename': x['filename']}


class PostCheatV2(MapTransform):
    def __init__(self, wt_channel, tc_channel, et_channel, apply_on_et=True):
        
        self.wt_channel = wt_channel
        self.tc_channel = tc_channel
        self.et_channel = et_channel
        self.apply_on_et = apply_on_et
    
    def __call__(self, x):
        prob, pred, mri = x['prob'], x['pred'], x['mri']
        if x['pred'].sum() > 0:
            return x
        
        assert len(prob.shape) == 4
        assert len(pred.shape) == 4
        
        prob = copy.deepcopy(prob)
        pred = copy.deepcopy(pred)
        y_post = np.zeros_like(pred)
        
        def func(mri):
            out = np.zeros_like(mri[:4])
            
            out[mri[:4] != 0] = 1.
            out = out.sum(axis=0)
            out[out > 0] = 1.
            
            cc = cc3d.connected_components(out, connectivity=6)
            
            if not cc.max() == 1:
                print('Hmm', cc.max())
            
            return out
        
        grid = func(mri)
        y_post[self.tc_channel] = grid
        y_post[self.wt_channel] = grid
        if self.apply_on_et == True:
            y_post[self.et_channel] = grid
        
        pred = y_post
        
        # print(x['filename'], pred[2].sum(), pred[0].sum(), pred[1].sum(), prob[2].max(), prob[0].max(), prob[1].max())
        
        return {'prob': prob, 'pred': pred, 'filename': x['filename']}