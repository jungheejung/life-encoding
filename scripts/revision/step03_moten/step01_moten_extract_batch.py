import moten
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--video-fname",
                    type=str, help="specify video name")
args = parser.parse_args()
video_fname = args.video_fname

MAIN_DIR = '/Users/h/Documents/projects_local/life-encoding/scripts/revision/videos'#'/dartfs/rc/lab/H/HaxbyLab/heejung'

SAVE_DIR = '/Users/h/Documents/projects_local/life-encoding/data/revision_moten'
# Stream and convert the RGB video into a sequence of luminance images

from collections import OrderedDict, namedtuple
Video = namedtuple('Video', ['TR', 'fps'])
od = OrderedDict()

od['ses-01_run-01_order-02_content-wanderers']      = Video(TR=430, fps=25)
od['ses-01_run-02_order-02_content-HB']             = Video(TR=123, fps=23.98)
od['ses-01_run-03_order-01_content-huggingpets']    = Video(TR=230, fps=29.97)
od['ses-01_run-03_order-04_content-dancewithdeath'] = Video(TR=295, fps=25)
od['ses-01_run-04_order-02_content-angrygrandpa']   = Video(TR=736, fps=29.97)
od['ses-02_run-02_order-03_content-menrunning']     = Video(TR=441, fps=25)
od['ses-02_run-03_order-01_content-unefille']       = Video(TR=252, fps=25)
od['ses-02_run-03_order-04_content-war']            = Video(TR=134, fps=25)
od['ses-03_run-02_order-01_content-planetearth']    = Video(TR=321, fps=25)
od['ses-03_run-02_order-03_content-heartstop']      = Video(TR=426, fps=23.98)
od['ses-03_run-03_order-01_content-normativeprosocial2'] = Video(TR=228, fps=23.98)
od['ses-04_run-01_order-02_content-gockskumara']    = Video(TR= 389, fps=25)



video_file = f'{MAIN_DIR}/{video_fname}.mp4'
luminance_images = moten.io.video2luminance(video_file, size = (108,192)) #, nimages=100)
print('luminance complete', flush=True)

# Create a pyramid of spatio-temporal gabor filters
nimages, vdim, hdim = luminance_images.shape
print(nimages, vdim, hdim)
# TODO: may be worth running ffmpeg
FPS = int(np.round(od[video_fname].fps, decimals=0))
pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=FPS)
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
save_dir = SAVE_DIR#f'{SAVE_DIR}'
np.save(os.path.join(save_dir, f'moten_video-{video_fname}.npy'), batched_data) #moten_features)
