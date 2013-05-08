#include <stdlib.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>

int
main ()
{

	CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
	uchar *data;
	int i, j, k, height, width, step, channels, key, filter = 0;

	if ( !capture ) {
		fprintf(stderr, "[error] capture is NULL\n");
		return -1;
	}

	// Create a window in which the captured images will be presented
	cvNamedWindow( "camfilter", CV_WINDOW_AUTOSIZE );

	IplImage* frame = cvQueryFrame( capture );
	// get the image data
	height    = frame->height;
	width     = frame->width;
	step      = frame->widthStep;
	channels  = frame->nChannels;
	data      = (uchar *)frame->imageData;
	printf("[debug] height/width: %dx%d, channels: %d\n",height,width,channels); 

	printf("Keys:\n  f:\ttoggle filter\n  ESC:\tquit\n");

	while ( true ) {
		// Get one frame
		IplImage* frame = cvQueryFrame( capture );
		if ( !frame ) {
			fprintf(stderr, "[error] frame is NULL\n");
			break;
		}

		// invert the image if flag is set
		if ( filter ) {
			for(i=0; i<height; i++){
				for(j=0; j<width; j++){
					for(k=0; k<channels; k++){
						data[i*step+j*channels+k] = 255-data[i*step+j*channels+k];
					}
				}
			}
		}

		cvShowImage("camfilter", frame);
		
		// keyboard handling
		key = cvWaitKey(20) & 255;
		//fprintf(stderr, "%d\n", key);
	  	if( key == 27 ) {	// 'ESC'
			break;
		} else if( key == 102 ) {	// 'f'
			filter ^= 1;
		}
	}

	// Release the capture device housekeeping
	cvReleaseCapture(&capture);
	cvDestroyWindow("camfilter");

	return 0;

}
