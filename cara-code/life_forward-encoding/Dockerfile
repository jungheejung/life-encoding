# installs dependencies for Res(2+1)D - OpenCV, ffmpeg, and Caffe2
# based on FB VMZ installation guide
# CVU 2019

# use the most recent centOS
FROM centos:latest

COPY docker_r21d_dependencies.sh /tmp/docker_r21d_dependencies.sh
RUN bash /tmp/docker_r21d_dependencies.sh

# to confirm installed correctly
RUN python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
