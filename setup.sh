#! /bin/bash

#################################################################################
#                                                                               #
# Python 3.7.3                                                                  #
# pip 20.1 from /home/pi/.local/lib/python3.7/site-packages/pip (python 3.7)    #
#                                                                               #
# cat /etc/os-release                                                           #
#                                                                               #
# PRETTY_NAME="Raspbian GNU/Linux 10 (buster)"                                  #
# NAME="Raspbian GNU/Linux"                                                     #
# VERSION_ID="10"                                                               #
# VERSION="10 (buster)"                                                         #
# VERSION_CODENAME=buster                                                       #
# ID=raspbian                                                                   #
# ID_LIKE=debian                                                                #
# HOME_URL="http://www.raspbian.org/"                                           #
# SUPPORT_URL="http://www.raspbian.org/RaspbianForums"                          #
# BUG_REPORT_URL="http://www.raspbian.org/RaspbianBugs"                         #
#                                                                               #
#################################################################################

# for cv2
apt-get update
apt-get upgrade
apt-get install libhdf5-dev \
                libhdf5-serial-dev \
                libatlas-base-dev \
                libjasper-dev \
                libqtgui4 \
                libqt4-test -y

echo "export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1" >> ~/.bashrc
source ~/.bashrc

# install pip lib
pip3 install -r requirements.txt

# for audio player
apt-get install python-gst-1.0 gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly gstreamer1.0-tools -y
