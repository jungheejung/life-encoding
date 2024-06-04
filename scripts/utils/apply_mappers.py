#!/usr/bin/env python2
# This script requires PyMVPA!
import os
from scipy.stats import zscore
import mvpa2 as mv
from mvpa2.base.hdf5 import h5load
from mvpa2.datasets.gifti import gifti_dataset
import sys
import nibabel as nib
import numpy as np


print("python version {0}".format(sys.version))
# Assign/create directories
user_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f000fb6'
scratch_dir = os.path.join('/dartfs-hpc/scratch', user_dir)
data_dir = '/dartfs/rc/lab/D/DBIC/DBIC/life_data/'

# Hyperalignment directory (/idata/DBIC/cara/life/pymvpa/)
hyper_dir = os.path.join(data_dir, 'hyperalign_mapper')

# Life fMRI data directory (/idata/DBIC/snastase/life)
fmri_dir = os.path.join(data_dir, 'life_dataset')
# Set up some hard-coded experiment variables
subjects = ['sub-rid000001', 'sub-rid000005', 'sub-rid000006',
            'sub-rid000009', 'sub-rid000012', 'sub-rid000014',
            'sub-rid000017', 'sub-rid000019', 'sub-rid000024',
            'sub-rid000027', 'sub-rid000031', 'sub-rid000032',
            'sub-rid000033', 'sub-rid000034', 'sub-rid000036',
            'sub-rid000037', 'sub-rid000038', 'sub-rid000041']
hemispheres = [sys.argv[1]]# ['lh', 'rh']
runs = [1, 2, 3, 4]
test_runs = [1, 2, 3, 4]
fmri_durs = {1: 374, 2: 346, 3: 377, 4: 412}

# Helper functions to read/write in GIfTI files
def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :]
                      for da in gii.darrays]) 
    return data

def write_gifti(data, output_fn, template_fn):
    gii = nib.load(template_fn)
    for i in np.arange(gii.numDA):
        gii.remove_gifti_data_array(0)
    gda = nib.gifti.GiftiDataArray(data)
    gii.add_gifti_data_array(gda)
    nib.gifti.giftiio.write(gii, output_fn)

# Loop through hemispheres and runs to apply mappers
for hemisphere in hemispheres:
    for test_run in test_runs:
        
        # Get training runs excluding test run
        # train_runs = [r for r in run_labels if r != test_run] 
        #train_runs = [r for r in test_runs if r != test_run] 
        # Load pre-trained hyperalignment mappers
        mappers = h5load(os.path.join(hyper_dir,
            'search_hyper_mappers_life_mask_nofsel_'
            '{0}_leftout_{1}.hdf5'.format(hemisphere, test_run)))                
        print("loading mappers...for hemi: {0} test_run: {1}".format(hemisphere, test_run))        
        

	# Project all subjects into shared space
	common_data = {}
        for subject in subjects:
            
            common_data[subject] = {}
            for run in runs:
	        tproject_gii = os.path.join(fmri_dir,'{0}_task-life_acq-{1}vol_run-0{2}.{3}.tproject.gii'.format( subject, fmri_durs[run], run, hemisphere)) 
                run_data = gifti_dataset(tproject_gii)
		print("life 74: initial run_data {0}".format(run_data.shape))
                print("check nans initial run_data: {0}".format(np.sum(np.all(np.isnan(np.squeeze(run_data)), axis=0))))
                # Z-score run data before hyperalignment just to make sure
                run_data = np.nan_to_num(zscore(run_data, axis=0))
		print("life 77: z-scored run_data {0}".format(run_data.shape))
		print("check nans zscored run_data {0}".format(np.sum(np.all(np.isnan(np.squeeze(run_data)), axis=0))))
                
		# Project training subjects into common space
                run_common = mappers[subject].forward(run_data)
		# print()
		print("life 80: mappers {0}".format(run_common.shape))
		print("check nans initial mappers {0}".format(np.sum(np.all(np.isnan(np.squeeze(run_common)), axis=0))))
                # Z-score run data before hyperalignment just to make sure
                run_common = zscore(run_common, axis=0)
		print("life 83: z-scored mappers {0}".format(run_common.shape))
		print("check nans zscored mappers {0}".format(np.sum(np.all(np.isnan(np.squeeze(run_common)), axis=0))))
                # Hold onto all subjects in common space
                common_data[subject][run] = run_common
		#print("life 86: common data {0}".format(common_data.shape))
                                
                # Save training subjects in common space
                write_gifti(run_common, os.path.join(fmri_dir,
                    ('{0}_task-life_acq-{1}vol_run-0{2}_'
                     'desc-HAcommon{3}.{4}.gii').format(
                        subject, fmri_durs[run], run,
                        test_run, hemisphere)), tproject_gii)
                print("finished mapping for subject-{0} test_run{1}".format(subject, test_run)) 
	



        # Loop through test subjects and reverse-project other subjects
        for test_subject in subjects:

            # Get training runs excluding test run
            #train_subjects = [s for s in subjects if s != test_subject] 
            for subject in subjects:#train_subjects:       
		for run in runs: 
		# TODO add [run] for every eleemnt
                # Reverse-project training subject into test subject space
                    run_test = mappers[test_subject].reverse(common_data[subject][run])
                    #print("life 109: run_test {0}".format(run_test.shape))
		    print("check nans run_test mappers {0}".format(np.sum(np.all(np.isnan(np.squeeze(run_test)), axis=0))))
                #TODO: NEED TEST SUBJECT LABEL IN FILENAME????
                # Save training subjects in test subject space
                    write_gifti(run_test, os.path.join(fmri_dir,
                    ('{0}_task-life_acq-{1}vol_run-0{2}_'
                     'desc-HAtestR{3}{4}.{5}.gii').format(
                        subject, fmri_durs[run], run,
                        test_run, test_subject.split('-')[-1], hemisphere)), tproject_gii)
                print("finished projecting train_subject-{0} back to test-subject-{1}".format(subject, test_subject))
