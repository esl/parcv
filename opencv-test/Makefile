CC=gcc
CFLAGS=-Wall -pedantic -g `pkg-config --cflags --libs opencv`
#CFLAGS=-Wall -pedantic -I/usr/local/include/opencv -I/usr/local/include

all: filter camfilter 

install-debs:
	sudo apt-get install libavformat-dev ffmpeg libcv2.3 libcvaux2.3 \
			  					libhighgui2.3 python-opencv opencv-doc libcv-dev \
							  	libcvaux-dev libhighgui-dev
filter:
	$(CC) -o filter filter.c $(CFLAGS)

camfilter:
	$(CC) -o camfilter camfilter.c $(CFLAGS)

clean:
	rm -f *.o filter camfilter
