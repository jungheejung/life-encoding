import moten
import os
import numpy as np

for run in [1,2,3,4]:
    # Stream and convert the RGB video into a sequence of luminance images
    video_file = f'/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/stimuli/video-{run}_fps-30.mp4'
    #video_file = f'/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/stimuli/video-1_fps-30_trim.mp4'
    luminance_images = moten.io.video2luminance(video_file, size = (192,108)) #, nimages=100)
    # Create a pyramid of spatio-temporal gabor filters
    nimages, vdim, hdim = luminance_images.shape
    print(nimages, vdim, hdim)
    # TODO: may be worth running ffmpeg
    pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=30)

    # Compute motion energy features
    moten_features = pyramid.project_stimulus(luminance_images)
    print(moten_features.shape)
    print(moten_features)

    # save
    save_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/motion_energy'
    np.save(os.path.join(save_dir, f'moten_run-{run}.npy'), moten_features)

