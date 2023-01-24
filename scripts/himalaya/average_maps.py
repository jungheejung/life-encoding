import numpy as np
import os, sys
from os.path import join
import nibabel as nib
"""
avg_all: average across all 4 runs (within run, average across participants)
avg_run: average within each run (across participants)
"""

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    for row in data:
        gda = nib.gifti.GiftiDataArray(row)
        gii.add_gifti_data_array(gda)
    nib.gifti.giftiio.write(gii, output_fn)

alignment = sys.argv[2]
suma_dir = '/Users/h/suma-fsaverage6'
main_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding'
output_dir = os.path.join(main_dir, 'results', 'himalaya', alignment)
n_splits = 40
index = int(sys.argv[1])
result_list = [ 'bg-r', 'agents-r', 'actions-r', 'ridge-coef', 'comb-r','comb-r2', 'split-pred', 'split-r2', 'bg-r2', 'agents-r2', 'actions-r2', 'comb-pred'
result = result_list[index]
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']
runs = [1,2,3,4]
hemis = ['lh', 'rh']
for hemisphere in hemis:
    avg_all = []
    for test_run in runs:
        avg_run = []
        for test_subject in subjects:
            run_data = np.load(f"{output_dir}/{result}_align-{alignment}_{test_subject}_run-{test_run}_hemi-{hemisphere}.npy")
            avg_run.append(run_data)
        if result[-1] == 'r': # split-r combine-r
            avg_run = np.tanh(np.mean(np.vstack(np.arctanh(avg_run)), axis = 0))
        else:
            avg_run = np.mean(np.vstack(avg_run), axis = 0)                
        np.save(f"{output_dir}/{result}_align-{alignment}_sub-mean_run-{test_run}_hemi-{hemisphere}.npy", avg_run)
        write_gifti(np.nan_to_num(avg_run.astype(float)),
                    template_fn = f'/dartfs/rc/lab/D/DBIC/DBIC/life_data/life_dataset/sub-rid000041_task-life_acq-412vol_run-04.{hemisphere}.tproject.gii',  
                    output_fn = join(output_dir,f'{result}_align-{alignment}_sub-mean_run-{test_run}_hemi-{hemisphere}.gii'))
        avg_all.append(avg_run)
        print(f"finished averageing subjects for run {test_run}")
    if result[-1] == 'r':
        avg_all =  np.tanh(np.mean(np.vstack(np.arctanh(avg_all)), axis = 0))
    else:
        avg_all = np.mean(np.vstack(avg_all), axis = 0)
    np.save(f"{output_dir}/{result}_align-{alignment}_sub-mean_run-mean_hemi-{hemisphere}.npy", avg_all)
    write_gifti(np.nan_to_num(avg_all.astype(float)),
                template_fn = f'/dartfs/rc/lab/D/DBIC/DBIC/life_data/life_dataset/sub-rid000041_task-life_acq-412vol_run-04.{hemisphere}.tproject.gii', 
                output_fn = join(output_dir,f'{result}_align-{alignment}_sub-mean_run-mean_hemi-{hemisphere}.gii'))
    print(f"finished recomining vertices:  testrun-{test_run} {hemisphere}")
