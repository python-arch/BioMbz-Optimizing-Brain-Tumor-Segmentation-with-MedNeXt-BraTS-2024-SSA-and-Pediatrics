import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import torch
import shutil
import pprint

from tqdm import tqdm

from monai import transforms

from biomedmbz_glioma.transforms_utils import ConvertToMultiChannelBasedOnBrats2023Classesd
from biomedmbz_glioma.postprocessing import *

from brats2023_eval import get_LesionWiseResults

def main(args):
    fn_postprocessing = transforms.Compose([
        AdvancedAsDiscrete(tc_threshold=args.thr_tc, wt_threshold=args.thr_wt, et_threshold=args.thr_et),
        AdvanceETPost(2, args.ms_et, args.mp_et, args.max_n_et, args.sorted_by, 26),
        AdvanceTCPost(0, 2, args.ms_tc, args.mp_tc, args.max_n_tc, args.sorted_by, 26),
        AdvanceWTPost(1, 0, 2, args.ms_wt, args.mp_wt, args.max_n_wt, args.sorted_by, 26),
        lambda x: torch.from_numpy(x['pred']),
    ])
    
    true_transform = ConvertToMultiChannelBasedOnBrats2023Classesd(keys="label")
    
    class EvalCVDatasetClass(torch.utils.data.Dataset):
        def __init__(self, list_samples):
            self.list_samples = list_samples
        
        def __len__(self):
            return len(self.list_samples)
        
        def __getitem__(self, index):
            sample = self.list_samples[index]
            
            name = sample['name']
            true = sample['true']
            pred = sample['pred']
            
            pred = np.load(pred)
            true = np.load(true)
            
            true = true_transform({'label': true})['label']
            mri  = np.load(sample['mri'])
            pred = fn_postprocessing({'prob': pred, 'mri': mri, 'filename': name})
            
            return {
                'name': name,
                'pred': pred,
                'true': true,
                'mri' : mri,
            }
    
    def run_eval_fold(list_samples, dir_csv_fold_results):
        if dir_csv_fold_results:
            os.makedirs(dir_csv_fold_results)
        
        dict_metrics = {
            'Dice': {'WT': [], 'TC': [], 'ET': []},
            'HD95': {'WT': [], 'TC': [], 'ET': []},
        }
        dict_legacy_metrics = {
            'Legacy Dice': {'WT': [], 'TC': [], 'ET': []},
            'Legacy HD95': {'WT': [], 'TC': [], 'ET': []},
        }
        dict_tp_fp_fn = {
            'Num_TP': {'WT': [], 'TC': [], 'ET': []},
            'Num_FP': {'WT': [], 'TC': [], 'ET': []},
            'Num_FN': {'WT': [], 'TC': [], 'ET': []},
        }
        
        dataset = EvalCVDatasetClass(list_samples)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, drop_last=False,
        )
        
        for sample in tqdm(dataloader):
            name = sample['name'][0]
            true = sample['true'][0]
            pred = sample['pred'][0]
            
            res_df = get_LesionWiseResults(
                pred.cpu().numpy(), true.cpu().numpy(), 'BraTS-GLI',
                output=os.path.join(dir_csv_fold_results, str(name)) if dir_csv_fold_results else None,
            )
            
            for _, row in res_df.iterrows():
                label = row['Labels']
                dict_metrics['Dice'][label].append(row['LesionWise_Score_Dice'])
                dict_metrics['HD95'][label].append(row['LesionWise_Score_HD95'])
                dict_legacy_metrics['Legacy Dice'][label].append(row['Legacy_Dice'])
                dict_legacy_metrics['Legacy HD95'][label].append(row['Legacy_HD95'])
                dict_tp_fp_fn['Num_TP'][label].append(row['Num_TP'])
                dict_tp_fp_fn['Num_FP'][label].append(row['Num_FP'])
                dict_tp_fp_fn['Num_FN'][label].append(row['Num_FN'])
        
        for dct in [dict_metrics, dict_legacy_metrics, dict_tp_fp_fn]:
            for metric_name in dct.keys():
                for label in dct[metric_name].keys():
                    dct[metric_name][label] = np.array(dct[metric_name][label]).mean()
        
        return dict_metrics, dict_legacy_metrics, dict_tp_fp_fn
    
    dict_metrics = {
        'Dice': {'WT': [], 'TC': [], 'ET': [], 'AVG': []},
        'HD95': {'WT': [], 'TC': [], 'ET': [], 'AVG': []},
    }
    dict_legacy_metrics = {
        'Legacy Dice': {'WT': [], 'TC': [], 'ET': [], 'AVG': []},
        'Legacy HD95': {'WT': [], 'TC': [], 'ET': [], 'AVG': []},
    }
    dict_tp_fp_fn = {
        'Num_FN': {'WT': [], 'TC': [], 'ET': [], 'AVG': []},
        'Num_FP': {'WT': [], 'TC': [], 'ET': [], 'AVG': []},
        'Num_TP': {'WT': [], 'TC': [], 'ET': [], 'AVG': []},
    }
    
    if args.dir_csv_results:
        if os.path.exists(args.dir_csv_results) and os.path.isdir(args.dir_csv_results):
            shutil.rmtree(args.dir_csv_results)
        os.makedirs(args.dir_csv_results)
    
    for fold in range(5):
        dir_prediction = os.path.join(args.dir_prediction, str(fold))
        
        list_name = [name.split('.')[0] for name in os.listdir(dir_prediction)]
        
        list_samples = [{
            'name': name,
            'mri' : os.path.join(args.dir_train, f'{name}_x.npy'),
            'true': os.path.join(args.dir_train, f'{name}_y.npy'),
            'pred': os.path.join(dir_prediction, f'{name}.npy'),
        } for name in list_name] # IMPORTANT
        
        tmp_dict_metrics, tmp_dict_legacy_metrics, tmp_dict_tp_fp_fn = run_eval_fold(
            list_samples,
            os.path.join(args.dir_csv_results, str(fold)) if args.dir_csv_results else None,
        )
        
        for dct, tmp_dct in zip([dict_metrics, dict_legacy_metrics, dict_tp_fp_fn], [tmp_dict_metrics, tmp_dict_legacy_metrics, tmp_dict_tp_fp_fn]):
            for metric_name in dct.keys():
                dct[metric_name]['AVG'].append(0)
                for i, label in enumerate(tmp_dct[metric_name].keys()):
                    dct[metric_name][label].append(tmp_dct[metric_name][label])
                    dct[metric_name]['AVG'][-1] += tmp_dct[metric_name][label]
                dct[metric_name]['AVG'][-1] /= (i + 1)
    
    for dct in [dict_metrics, dict_legacy_metrics, dict_tp_fp_fn]:
        for metric_name in dct.keys():
            for label in dct[metric_name].keys():
                mean = np.array(dct[metric_name][label]).mean()
                std  = np.array(dct[metric_name][label]).std()
                if metric_name == 'Dice':
                    mean = mean * 100
                    std  = std * 100
                if dct != dict_tp_fp_fn:
                    dct[metric_name][label] = f"{mean:.2f} +- {std:.2f}"
                else:
                    dct[metric_name][label] = f"{mean:.4f} +- {std:.4f}"
    
    if args.pprint:
        pp = pprint.PrettyPrinter(depth=4)
        pp.pprint(dict_metrics)
        pp.pprint(dict_tp_fp_fn)
        pp.pprint(dict_legacy_metrics)
    
    return dict_metrics, dict_legacy_metrics, dict_tp_fp_fn

if __name__ == '__main__':
    class Args:
        dir_prediction = ... # Output from cv_get_predictions.py (it should be os.path.join(save_dir, eval_name))
        dir_train = .... # Preprocessed data dir
        dir_csv_results = ... # Save CSV path
        thr_tc=0.5
        thr_wt=0.5
        thr_et=0.5
        ms_tc=60
        ms_wt=60
        ms_et=60
        mp_tc=0.775
        mp_wt=0.825
        mp_et=0.825
        max_n_tc=5
        max_n_wt=5
        max_n_et=5
        sorted_by='size'
        seed=42
        num_workers=4
        pprint=True
    
    args = Args()
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    dict_metrics, dict_legacy_metrics, dict_tp_fp_fn = main(args)
    
    avg_dice = float(dict_metrics['Dice']['AVG'].split('+-')[0])
    avg_hd95 = float(dict_metrics['HD95']['AVG'].split('+-')[0])
    
    print('Avg Dice:', avg_dice)
    print('Avg HD95:', avg_hd95)