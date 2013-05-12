#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <cv.h>
#include <highgui.h>
#include <erl_nif.h>

#define DEBUG
#ifdef DEBUG
	#define debug(...) fprintf(stderr, __VA_ARGS__)
#else
	#define debug(...) ;
#endif

CvCapture* cvcapture;
IplImage* frame; 
ErlNifBinary bindata;
int fheight, fwidth, fsize, fchannels = 0;
bool init_flag = 0;

void
init ()
{
	cvcapture = cvCaptureFromCAM(CV_CAP_ANY);

	if ( !cvcapture ) {
		// FIXME: return Erlang term here
		debug("%s", "[error] cvcapture is NULL\n");
	}

	// do an initial capture to determine frame params and alloc binary term
	frame = cvQueryFrame(cvcapture);

	fheight = frame->height;
	fwidth = frame->width;
	fchannels = frame->nChannels;
	fsize = frame->imageSize;

	debug("\n[debug] height: %d, width: %d, channels: %d, size: %d\n", fheight, fwidth, fchannels, fsize); 

	enif_alloc_binary(fsize, &bindata);

	//cvReleaseCapture(&cvcapture);
	
	init_flag = true;
}

ERL_NIF_TERM
capture (ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
	//FIXME: Should this be removed and init() called explicitly via NIF?
	if ( !init_flag )
	{
		init();
	}

	//cvcapture = cvCaptureFromCAM(CV_CAP_ANY);
	
	frame = cvQueryFrame(cvcapture);
	//strcpy(bindata.data, fdata);
	memcpy(bindata.data, frame->imageData, fsize);

	//cvReleaseCapture(&cvcapture);
	
	return enif_make_binary(env, &bindata); 
}

static ErlNifFunc nif_funcs[] = {
    {"capture", 0, capture}
};

ERL_NIF_INIT(camgrab, nif_funcs, NULL, NULL, NULL, NULL)

