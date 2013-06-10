#include <erl_nif.h>
#include <highgui.h>
#include <cv.h>

//------------------------------------------------------------------------------
// Resources
//------------------------------------------------------------------------------

static ErlNifResourceType* device_res = NULL;
static ErlNifResourceType* frame_res = NULL;

typedef struct _device_t {
  struct CvCapture* _device;
} device_t;

typedef struct _frame_t {
  IplImage* _frame;
} frame_t;

//------------------------------------------------------------------------------
// NIF callbacks
//------------------------------------------------------------------------------

static void
device_cleanup(ErlNifEnv* env, void* arg) {
  enif_free(arg);
}
static void
frame_cleanup(ErlNifEnv* env, void* arg) {
  enif_free(arg);
}
static void
image_cleanup(ErlNifEnv* env, void* arg) {
  enif_free(arg);
}

static int 
load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info)
{
  ErlNifResourceFlags flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
  device_res = enif_open_resource_type(env, "ocv_nif", "ocv_device",
				       &device_cleanup,
				       flags, 0);
  frame_res = enif_open_resource_type(env, "ocv_nif", "ocv_frame",
				      &frame_cleanup,
				      flags, 0);
  image_res = enif_open_resource_type(env, "ocv_nif", "ocv_image",
				      &image_cleanup,
				      flags, 0);
  return 0;
}

//------------------------------------------------------------------------------
// API Implementation
//------------------------------------------------------------------------------

// any_device/0 :: () -> {ok, Device} | {error, no_capture_device}
static ERL_NIF_TERM
any_device(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  device_t* dev = enif_alloc_resource(device_res, sizeof(device_t));

  dev->_device = (struct CvCapture*) cvCaptureFromCAM(CV_CAP_ANY);

  if (!dev->_device) {
    // return {error, no_capture_device}
    return enif_make_tuple2(env, enif_make_atom(env, "error"), 
			    enif_make_atom(env, "no_capture_device"));
  } 

  // return {ok, <<CaptureDevice>>}
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), 
			  enif_make_resource(env, dev));
}

// new_frame/1 :: (device) -> frame()
static ERL_NIF_TERM
new_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  device_t* dev;
  if (!enif_get_resource(env, argv[0], device_res, (void**) &dev)) {
    return enif_make_badarg(env);
  }
  frame_t* frame = enif_alloc_resource(frame_res, sizeof(frame_t));
  frame->_frame = (IplImage*) cvQueryFrame(dev->_device);
  if (!frame->_frame->data)
    return enif_make_atom(e, "error");
  cvCvtColor(frame->_frame, frame->_frame, CV_BGR2GRAY);
  cvThreshold(frame->_frame, frame->_frame, 20, 255, THRESH_BINARY);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), 
			  enif_make_resource(env, gray));
}

// query_frame/1 :: (device, frame) -> ok | error
static ERL_NIF_TERM
query_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  device_t* dev;
  frame_t* frame;
  if (!enif_get_resource(env, argv[0], device_res, (void**) &dev) ||
      !enif_get_resource(env, argv[1], frame_res, (void**) &frame)) {
    return enif_make_badarg(env);
  }
  frame->_frame = (IplImage*) cvQueryFrame(dev->_device);
  if (!frame->_frame->data)
    return enif_make_atom(e, "error");
  cvCvtColor(frame->_frame, frame->_frame, CV_BGR2GRAY);
  cvThreshold(frame->_frame, frame->_frame, 20, 255, THRESH_BINARY);
  return enif_make_atom(env, "ok");
}

// frame_to_tuple/1 :: (frame) -> {ok, {W, H, NChannels, ImageSize, ImageData}}
static ERL_NIF_TERM
frame_to_tuple(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  ERL_NIF_TERM result;
  frame_t* frame;
  if (!enif_get_resource(env, argv[0], frame_res, (void**) &frame)) {
    return enif_make_badarg(env);
  }
  ERL_NIF_TERM* arr = 
    (ERL_NIF_TERM*) malloc(sizeof(ERL_NIF_TERM) * frame->_frame->imageSize);
  int i;
  for (i = 0; i < frame->_frame->imageSize; i++) {
    arr[i] = enif_make_int(env, frame->_frame->imageData[i]);
  }
  ERL_NIF_TERM list = 
    enif_make_list_from_array(env, arr, frame->_frame->imageSize);
  result = enif_make_tuple5(env,
			    enif_make_int(env, frame->_frame->width),
			    enif_make_int(env, frame->_frame->height),
			    enif_make_int(env, frame->_frame->nChannels),
			    enif_make_int(env, frame->_frame->imageSize),
			    list);
  return result;
}

static ErlNifFunc nif_funcs[] = {
  {"any_device", 0, any_device},
  {"new_frame", 1, new_frame},
  {"query_frame", 2, query_frame},
  {"frame_to_tuple", 1, frame_to_tuple}
};

ERL_NIF_INIT(ocv_nif, nif_funcs, load, NULL, NULL, NULL)
