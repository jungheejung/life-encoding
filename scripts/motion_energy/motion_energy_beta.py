import moten

# Stream and convert the RGB video into a sequence of luminance images
video_file = '/dartfs/rc/lab/D/DBIC/DBIC/f0042x1/life-encoding/data/stimuli/new_part1.mp4'
luminance_images = moten.io.video2luminance(video_file, nimages=100)

# Create a pyramid of spatio-temporal gabor filters
nimages, vdim, hdim = luminance_images.shape
print(nimages, vdim, hdim)
pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=30)#29.97)

# Compute motion energy features
moten_features = pyramid.project_stimulus(luminance_images)
print(moten_features.shape)
print(moten_features)
