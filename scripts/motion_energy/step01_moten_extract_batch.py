import moten
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--run", choices=[1, 2, 3, 4],
                    type=int, help="specify test run")
args = parser.parse_args()
run = args.run
#for run in [1,2,3,4]:
# Stream and convert the RGB video into a sequence of luminance images
video_file = f'/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/stimuli/video-{run}_fps-30.mp4'
luminance_images = moten.io.video2luminance(video_file, size = (192, 108)) #, nimages=100)
print('luminance complete', flush=True)

# Create a pyramid of spatio-temporal gabor filters
nimages, vdim, hdim = luminance_images.shape
print(nimages, vdim, hdim)
# TODO: may be worth running ffmpeg
pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=30)
filter_temporal_width=20
filter_temporal_width = pyramid.definition['filter_temporal_width']
window = int(np.ceil((filter_temporal_width/2)))
# Compute motion energy features
#moten_features = pyramid.project_stimulus(luminance_images)

nbatches = 5
batch_size = int(np.ceil(nimages/nbatches))
batched_data = []
for bdx in range(nbatches):
    start_frame, end_frame = batch_size*bdx, batch_size*(bdx + 1)
    print('Batch %i/%i [%i:%i]'%(bdx+1, nbatches, start_frame, end_frame))

    # Padding
    batch_start = max(start_frame - window, 0)
    batch_end = end_frame + window
    stimulus_batch = luminance_images[batch_start:batch_end]
    batched_responses = pyramid.project_stimulus(stimulus_batch)

    # Trim edges
    if bdx == 0:
        batched_responses = batched_responses[:-window]
    elif bdx + 1 == nbatches:
        batched_responses = batched_responses[window:]
    else:
        batched_responses = batched_responses[window:-window]
    batched_data.append(batched_responses)

batched_data = np.vstack(batched_data)

print(batched_data.shape)
#print(moten_features)

# save
save_dir = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/motion_energy'
np.save(os.path.join(save_dir, f'moten_run-{run}.npy'), batched_data) #moten_features)
