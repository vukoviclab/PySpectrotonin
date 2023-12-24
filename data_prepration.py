import glob
import pandas as pd

def process_wave_data(wave_list_path='imm_0.0_tol_0.05/n_gram_1/*', 
                      threshold_path='imm_*_tol_*', 
                      defined_format='csv', 
                      sequence_file='predict.csv',
                      rerun=False,
                     
                     ):
    
    # Extract wave list
    wave_list = sorted([i.split('/')[-1] for i in glob.glob(wave_list_path)], key=lambda x: float(x))
    
    # Extract threshold list
    threshold_list = [_dir for _dir in glob.glob(threshold_path) 
                      if defined_format not in _dir and 'png' not in _dir]
    
    # Create threshold format list
    threshold_format_list = sorted(
        [_dir for _dir in glob.glob(f'imm_*_tol_*.{defined_format}')],
        key=lambda x: (float(x.split('_')[1]), float(x.split('_')[3].replace(f'.{defined_format}', '')))
    )
    if rerun:
        for threshold in threshold_list:
            df_final = pd.read_csv(sequence_file)
            for wave_csv in wave_list:
                csv_per_wave = glob.glob(f'{threshold}/n_gram_1/{wave_csv}/0*/prediction_validate.csv')

                if len(csv_per_wave) != 0:
                    df_tmp = pd.concat([pd.read_csv(csv_file) for csv_file in csv_per_wave], axis=1)

                    # Remove duplicated columns and compute mean
                    df_tmp_ = df_tmp.loc[:, ~df_tmp.columns.duplicated()].copy()
                    df_tmp_['mean'] = df_tmp_.mean(axis=1)

                    df_final.loc[df_tmp_.index, wave_csv] = df_tmp_['mean']
                    df_final.to_csv(f'{threshold}.{defined_format}', index=False)

    return threshold_format_list,threshold_list


