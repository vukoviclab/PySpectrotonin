from threshold import *
import pandas as pd
import glob
from sklearn.svm import SVR

df_full = pd.read_csv('full.csv')
df_predict = pd.read_csv('predict.csv')
drop_duplicates(df_full, 'sequence')
sequence_removing(df_full, df_predict)
df_selected, df_minmax, df_minmax_selected, wavelength_value, wavelength_value_selected = min_max_mean(df_full, 1.5)

for threshold_parameter in [(0.0, 0.05), (0.0, 0.1), (0.05, 0.05), (0.05, 0.1), (0.1, 0.05), (0.1, 0.1),
                            (0.15, 0.05), (0.15, 0.1), (0.2, 0.05), (0.2, 0.1)]:
    for ng in [1]:
        for wavelength in list(df_selected.columns)[1:]:
            threshold_value = threshold_sequence_removing(df=df_selected, wavelength=wavelength,
                                                          increase_mean_med=threshold_parameter[0],
                                                          tolerance=threshold_parameter[1])
            if threshold_value:
                threshold_path = "_".join(('imm', str(threshold_parameter[0]), 'tol', str(threshold_parameter[1])))
                if not os.path.exists(threshold_path):
                    os.makedirs(threshold_path)
                sequence_to_numeric(df_in_threshold=threshold_value[0],
                                    df_out_threshold=threshold_value[1],
                                    n_gram=ng,
                                    wavelength=wavelength,
                                    path_wavelength=threshold_path + '/n_gram_' + str(ng) + '/' + wavelength,
                                    method="regression",
                                    nucleotide_list=('A', 'C', 'G', 'T')
                                    )

imm_tol_dir = sorted(glob.glob('imm_*'))
for imm_tol in imm_tol_dir:
    print(imm_tol)
    machine_learning_regression(
        model=SVR(kernel='rbf'),
        name="SVM_RBF",
        numpy_array_path=imm_tol + '/n_gram_1/*',
        model_numbers=5,
        n_gram=[1],
        r2_score_limit=0.4,
        metric_csv_name=imm_tol + '/final_result.csv',
        validate_sequence=True,
        df_validate=df_predict,
        nucleic_list=['A', 'C', 'G', 'T'],
        max_time_hour=3
    )
