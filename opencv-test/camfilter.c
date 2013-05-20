#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cv.h>
#include <highgui.h>
//#include <imgproc.h>

int
main ()
{

	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);

	uchar *data;
	int i, j, k, height, width, step, channels, key, size, filter = 0;
	char debug[100];

	if ( !capture ) {
		fprintf(stderr, "[error] capture is NULL\n");
		return -1;
	}

	// Create a window in which the captured images will be presented
	cvNamedWindow("camfilter", CV_WINDOW_AUTOSIZE);

	IplImage *frame = cvQueryFrame(capture);
	IplImage  *framegray, *framebinary, *framecorners; 
	CvSeq *circles;	
	CvMemStorage *circlestorage;

	CvFont font;
	cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, 2, 2, 0, 0.5, CV_AA );


	// get the image data
	height    = frame->height;
	width     = frame->width;
	step      = frame->widthStep;
	channels  = frame->nChannels;
	size  = frame->imageSize;
	data      = (uchar *)frame->imageData;
	printf("[debug] height/width: %dx%d, channels: %d, size: %d\n",height,width,channels, size); 

	printf("Keys:\n  f:\ttoggle filter\n  ESC:\tquit\n");

	for (;;)  {
		// Get one frame
		IplImage* frame = cvQueryFrame( capture );
		if ( !frame ) {
			fprintf(stderr, "[error] frame is NULL\n");
			break;
		}

		// rgb -> grayscale
		framegray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
		cvCvtColor(frame, framegray, CV_RGB2GRAY);		

		// grayscale -> binary
		framebinary = cvCreateImage(cvGetSize(framegray), IPL_DEPTH_8U, 1);
		//cvThreshold(framegray, framebinary, 50, 255, CV_THRESH_OTSU); //use otsu's algorithm for threshold
		cvThreshold(framegray, framebinary, 100, 255, CV_THRESH_BINARY);

		// canny edge detect
		framecorners = cvCreateImage(cvGetSize(framebinary), IPL_DEPTH_8U, 1);
		cvCanny(framebinary, framecorners, 200, 20, 3);

		// apply gaussian blur
		cvSmooth(framegray, framegray, CV_GAUSSIAN, 5, 5, 2.0, 2.0);

		// find circles
		circlestorage = cvCreateMemStorage(0);
		circles = cvHoughCircles(framegray, circlestorage, CV_HOUGH_GRADIENT,
            		1,	// dp - inverse ratio of the accumulator resolution
					50,	// min_dist - minimum distance between circle centres
					30,	// param1 - higher threshold value for Canny
					70,	// param2 - accumulator threshold for the circle centers; smaller->more false circles
					20,	// min_radius - minimum radius
					150);	// max_radius - maximum radius

		// draw circles over the original image
		printf("circles == %d\n", circles->total);
		int x;
		float *p;
		for (x = 0; x < circles->total; x++) {
			p = (float*)cvGetSeqElem(circles, x);
			CvPoint center = cvPoint(cvRound(p[0]), cvRound(p[1]));
			CvScalar val = cvGet2D(framegray, center.y, center.x);
			if (val.val[0] < 1) continue;
			sprintf(debug, "%d %d %d", cvRound(p[0]), cvRound(p[1]), cvRound(p[2]));
			cvCircle(frame, center, cvRound(p[2]), CV_RGB(0,255,0), 1, CV_AA, 0);
			cvPutText(frame, debug, cvPoint(cvRound(p[0]), cvRound(p[1])), &font, CV_RGB(255,0,0));
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

		//sleep(1);
	}

	// Release the capture device housekeeping
	cvReleaseCapture(&capture);
	cvDestroyWindow("camfilter");

	return 0;

}
