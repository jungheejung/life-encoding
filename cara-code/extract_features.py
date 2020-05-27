# # for the life project
# # semantic feature vectors from annotated movie
# # narration/music spectral: MelspectrogramExtractor - extracts mel-scaled spectrogram
# # narration w2v: WordEmbeddingExtractor - extracts word2vec embedding, can feed audio file and Watson API convert automatically
# # image labeling: GoogleVisionAPILabelExtractor- uses Google Vision API to extract labels from images
# # saliency: SaliencyExtractor- extracts saliency of the image using Itti & Koch (1998) algorithm implemented in pySaliencyMap
# # motion-energy: FarnebackOpticalFlowExtractor - extracts total amount of dense optical flow between every pair of video frames
# # CVU 2018
#
#
#     # SPECTROGRAM FEATURE VECTORS
#
#     # train_stim = []
#     # test_stim = []
#     # run_list = [[],[],[],[]]
#     # for run in range(1, 5):
#     #     directory = os.path.join(cara_data_dir, 'spectral', 'life_part{0}'.format(run))
#     #     for f in os.listdir(directory):
#     #         s = pd.read_csv(directory, f))
#     #         filter_col = [col for col in s if col.startswith('mel')]
#     #         tr_s = np.array(s[filter_col])
#     #         avg = np.mean(tr_s, axis=0)
#     #         run_list[run-1].append(avg)
#     #
#     # for i in range(len(run_list)):
#     #     this = np.vstack(run_list[i])
#     #     run_list[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
#     #
#     #     print(run_list[i].shape)
#     #
#     # train_stim = np.concatenate((run_list[0], run_list[1], run_list[3]), axis=0)
#     # test_stim = run_list[2]
#
#     # SALIENCY FEATURES
#     run_list = []
#     for run in range(1, 5):
#         directory = os.path.join(cara_data_dir, 'saliency')
#         for f in os.listdir(directory):
#             p = pd.read_csv(os.path.join(directory, f))
#             filter_col = [col for col in s if col.startswith('SaliencyExtractor') or col in ['onset', 'duration']]
#             tr_s = downsample(s[filter_col], 2.5)
#             delayed_tr_s = delayed(tr_s)
#             run_list[i] = delayed_tr_s
#
#     train_stim = np.concatenate((run_list[0], run_list[1], run_list[3]), axis=0)
#     test_stim = run_list[2]
#
#
# def downsample(data, band):
#     max_time = data['onset'][-1] + data['duration'][-1]
#     tr_list = []
#     for i in np.arange(0, max_time, band):
#         dat = data[data['onset'] >= i and data['onset'] <= i+band]
#         filter_col = [col for col in s if col not in ['onset', 'duration']]
#         np_dat = np.array(dat[filter_col])
#         avg_dat = np.mean(np_dat, axis=0)
#         tr_list.append(avg_dat)
#     return(np.vstack(tr_list))
#
# def delayed(data):
#     return(np.concatenate((data[3:,:], data[2:-1,:], data[1:-2,:], data[:-3,:]), axis=1))
#
# # FLOW FEATURES
# flow_list = []
# for run in range(1, 5):
#     directory = os.path.join(data_dir, 'flow')
#     for f in os.listdir(directory):
#         p = pd.read_csv(os.path.join(directory, f))
#         filter_col = [col for col in s if 'flow' in col or col in ['onset', 'duration']]
#         tr_s = downsample(s[filter_col], 2.5)
#         delayed_tr_s = delayed(tr_s)
#         flow_list[i] = delayed_tr_s
#
# for i in run_list:
#     run_list[i] = np.concatenate((run_list[i], flow_list[i]))
# train_stim = np.concatenate((run_list[0], run_list[1], run_list[3]), axis=0)
# test_stim = run_list[2]

import numpy as np
# import pandas as pd
import moviepy.editor as mp
import os, pickle

from pliers.stimuli import VideoStim, AudioStim
from pliers.converters import VideoToAudioConverter, GoogleSpeechAPIConverter, VideoFrameIterator
from pliers.filters import FrameSamplingFilter, TokenRemovalFilter
from pliers.extractors import MelspectrogramExtractor, SaliencyExtractor, WordEmbeddingExtractor, FarnebackOpticalFlowExtractor, GoogleVisionAPILabelExtractor, merge_results
from pliers import config
config.set_option('cache_transformers', False)

print('made it')
data_dir = '/idata/DBIC/cara/life/data/'
a_data_dir = '/home/aconnoll/orig_life_movies/'
parts_list = ['life_part1', 'life_part2', 'life_part3', 'life_part4', 'complete']

def write_data(key, part, stim):
    path = os.path.join(data_dir, key)
    if not os.path.exists(path):
        os.makedirs(path)
    stim.save(os.path.join(path, '{0}.wav'.format(part)))
    # stim.to_csv()

saliency = SaliencyExtractor()
flow = FarnebackOpticalFlowExtractor()
spect = MelspectrogramExtractor()

for f in os.listdir(a_data_dir):
    fp = os.path.join(a_data_dir, f)
    part = f[:-4]
    print(fp, part)
    v_stim = VideoStim(filename=fp)

    # Sample 2 frames per second
    filt = VideoFrameIterator()
    selected_frames_stim = filt.transform(v_stim)
    # get audio from movie
    v2a = VideoToAudioConverter()
    a_stim = v2a.transform(v_stim)

    write_data('audio', part, a_stim)
#
#     # get saliency of images and concat (should average?)
#     saliency_trans = saliency.transform(selected_frames_stim)
#     saliency_stim = merge_results(saliency_trans)
#     print(type(saliency_stim))
#
#     # get dense optical flow of video
#     flow_stim = flow.transform(v_stim).to_df()
#     print(type(flow_stim))
#
#
#     # get spectrogram data of audio
#     spect_stim = spect.transform(a_stim).to_df()
#     print(type(spect_stim))
#
#
#     write_data('saliency', part, saliency_stim)
#     write_data('flow', part, flow_stim)
#     write_data('spectral', part, spect_stim)
#
#
#
# # # semantic feature vectors from annotated movie
# # cam = np.load('/ihome/cara/life/w2v_src/google_w2v_ca.npy')
# semantic_f = []
# semantic_f.append(proc[:366])
# semantic_f.append(proc[366:704])
# semantic_f.append(proc[704:1073])
# semantic_f.append(proc[1073:])
#
# for i in range(len(semantic_f)):
# 	this = semantic_f[i]
# 	semantic_f[i] = np.concatenate((this[3:,:], this[2:-1,:], this[1:-2,:], this[:-3,:]), axis=1)
#
# def round_down(num, divisor):
#     return num - (num%divisor)
# #
# # # extract audio from movie in 2.5 s chunks
# # for filename in os.listdir('/home/aconnoll/orig_life_movies/'):
# #     if (filename.endswith('.mp4')): #or .avi, .mpeg, whatever.
# #         a = os.path.join(data_dir, 'audio/{0}.wav'.format(filename[:-4]))
# #         f = os.path.join('/home/aconnoll/orig_life_movies/', filename)
# #         print(f, a)
# #         clip = mp.VideoFileClip(f)
# #         clip.write_videofile(os.path.join(data_dir, 'visual/{0}.mp4'.format(filename[:-4])))
# #         # clip.audio.write_audiofile(a)
# #         # aud = mp.AudioFileClip(a)
# #
# #         v_parts_dir = os.path.join(data_dir, 'visual', filename[:-4])
# #         if not os.path.exists(v_parts_dir):
# #             os.makedirs(v_parts_dir)
# #
# #         for i in np.arange(0.0, round_down(clip.duration, 2.5), 2.5):
# #             subc = clip.subclip(i, i+2.5)
# #             subc_file = os.path.join(v_parts_dir, '{0}_{1}.mp4'.format(filename[:-4], str(i)))
# #             print(subc_file, clip.duration)
# #             subc.write_videofile(subc_file)
# #
# #         a_parts_dir = os.path.join(data_dir, 'audio', filename[:-4])
# #         if not os.path.exists(a_parts_dir):
# #             os.makedirs(a_parts_dir)
# #
# #         for i in np.arange(0.0, round_down(clip.duration, 2.5), 2.5):
# #             subaud = aud.subclip(i, i+2.5)
# #             subaud_file = os.path.join(a_parts_dir, '{0}_{1}.wav'.format(filename[:-4], str(i)))
# #             print(subaud_file, clip.duration)
# #             subaud.write_audiofile(subaud_file)
#
# # spect = MelspectrogramExtractor()
# # speech2text = GoogleSpeechAPIConverter()
# # text2vec = WordEmbeddingExtractor('/ihome/cara/life/w2v_src/GoogleNews-vectors-negative300.bin')
# #
# # image_labels = GoogleVisionAPILabelExtractor(batch_size=15)
#
# for part in parts_list:
#     print(part)
#     # a_parts_dir = os.path.join(data_dir, 'audio', part)
#     #
#     # for tr in os.listdir(a_parts_dir):
#     #     if (tr.endswith('.wav')):
#     #
#     #         f = os.path.join(a_parts_dir, tr)
#     #         audio_stim = AudioStim(filename=f, sampling_rate=44100)
#     #
#     #         # get spectrogram
#     #         spectral_stim = spect.transform(audio_stim)
#     #         path = os.path.join(data_dir, 'spectral', part)
#     #         if not os.path.exists(path):
#     #             os.makedirs(path)
#     #         spectral_stim.to_df().to_csv(os.path.join(path, '{0}.csv'.format(tr[:-4])))
#
#                     # # get transcription with google/watson speech api
#                     # transcribed_stim = speech2text.transform(audio_stim)
#                     # transcribed_stim.save(os.path.join(data_dir, 'transcribed', part[:-4], tr[:-4]))
#                     #
#                     # # convert transcription to w2v
#                     # transcribed_vec_stim = text2vec.transform(audio_stim)
#                     # transcribed_vec_stim.save(os.path.join(data_dir, 'transcribed_vec', part[:-4], tr[:-4]))
#
#     v_parts_dir = os.path.join(data_dir, 'visual', part)
#     for tr in os.listdir(v_parts_dir):
#             if (tr.endswith('.mp4')):
#                 f = os.path.join(v_parts_dir, tr)
#                 print(f)
#                 vid_stim = VideoStim(filename=f)
#
#                 # Convert the VideoStim to a list of ImageStims
#                 conv = VideoFrameIterator()
#                 frames = conv.transform(stim)
#
#                 # Sample 2 frames per second
#                 filt = FrameSamplingFilter(hertz=10)
#                 selected_frames_stim = filt.transform(frames)
#
#                 # get saliency of images and concat (should average?)
#                 saliency_stim = saliency.transform(selected_frames_stim)
#                 saliency_data = merge_results(saliency_stim)
#                 path = os.path.join(data_dir, 'saliency', part)
#                 if not os.path.exists(path):
#                     os.makedirs(path)
#                 saliency_data.to_df().to_csv(os.path.join(path, '{0}.csv'.format(tr[:-4])))
#
#                 # # get image labels from selected images with google vision api
#                 # image_labels_stim = image_labels.transform(selected_frames_stim)
#                 # image_labels_data = merge_results(image_labels_stim)
#                 # image_labels_data.save(os.path.join(data_dir, 'image_labels', part[:-4], tr[:-4]))
#                 #
#                 # get dense optical flow of video
#                 flow_stim = flow.transform(vid_stim)
#                 path = os.path.join(data_dir, 'flow', part)
#                 if not os.path.exists(path):
#                     os.makedirs(path)
#                 flow_stim.to_df().to_csv(os.path.join(path, '{0}.csv'.format(tr[:-4])))
# #
#
# # video = VideoStim('my_movie.mpg')
# #
# # # Convert the VideoStim to a list of ImageStims
# # conv = VideoFrameIterator()
# # frames = conv.transform(video)
# #
# # # Sample 2 frames per second
# # filt = FrameSamplingFilter(hertz=2)
# # selected_frames = filt.transform(frames)
# #
# # # Detect faces in all frames
# # ext = GoogleVisionAPIFaceExtractor()
# # face_features = ext.transform(selected_frames)
# #
# # # Merge results from all frames into one big pandas DataFrame
# # data = merge_results(face_features)
