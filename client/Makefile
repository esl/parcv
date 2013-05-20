CC=gcc
#CFLAGS=-Wall -fPIC -shared -g -I/usr/lib/erlang/erts-5.10.1/include `pkg-config --cflags --libs opencv` #R16
CFLAGS=-Wall -fPIC -shared -g -I/usr/lib/erlang/erts-5.9.3.1/include `pkg-config --cflags --libs opencv` #R15B03-1

all: camgrab_nif

install-debs:
	sudo apt-get install libavformat-dev ffmpeg libcv2.3 libcvaux2.3 \
			  libhighgui2.3 python-opencv opencv-doc libcv-dev \
			  libcvaux-dev libhighgui-dev

camgrab_nif: 
	$(CC) -o camgrab_nif.so camgrab_nif.c $(CFLAGS)

clean:
	rm -f camgrab_nif.so *.beam

