import matplotlib; matplotlib.use('agg')
import numpy as np
import mvpa2.suite as mv
from famfaceangles import encmodel
import os, imageio

movie_dir = '/home/aconnoll/orig_life_movies/'
step = 5

def grouped_mean(arr, n=step):
    result = np.cumsum(arr, 0)[n-1::n]/float(n)
    result[1:] = result[1:] - result[:-1]
    return result

parts= []
for part in range(1,5):
    print(part)
    reader = imageio.get_reader(os.path.join(movie_dir, 'life_part{0}.mp4'.format(part)))
    fps = reader.get_meta_data()['fps']
    num_frames = reader.get_length()
    frame_shape = reader.get_data(0).shape

    vid = np.empty((num_frames, frame_shape[0], frame_shape[1], 3))
    print(vid.shape)
    for num in range(reader.get_length()):
        image = reader.get_data(num)
        vid[num,:,:,:] = image

    reader.close()

    print('fps: ', fps)
    print('vid shape: ', vid.shape)

    # resampled_vid = vid[::3].copy()
    motion = encmodel.global_motion(vid, prev_frame=None, max_frames=vid.shape[0])
    np.save('motion_{0}.npy'.format(part), motion)

    motion_ds = grouped_mean(motion,75)
    np.save('motion_downsampled_{0}.npy'.format(part), motion_ds)
    parts.append(motion_ds)

np.save('motion_downsampled_complete.npy', np.concatenate(parts, axis=0))
