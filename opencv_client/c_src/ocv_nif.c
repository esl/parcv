#include <erl_nif.h>
#include <opencv/cv.h>

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
device_cleanup(ErlNifEnv* env, void* arg) {}
static void
frame_cleanup(ErlNifEnv* env, void* arg) {}

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
  dev->_device = (struct CvCapture*) cvCaptureFromCAM(-1);

  if (!dev->_device) {
    // return {error, no_capture_device}
    return enif_make_tuple2(env, enif_make_atom(env, "error"), 
			    enif_make_atom(env, "no_capture_device"));
  } 

  // return {ok, <<CaptureDevice>>}
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), 
			  enif_make_resource(env, dev));
}

static ERL_NIF_TERM
free_device(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  device_t* dev;
  assert_badarg(enif_get_resource(env, argv[0], device_res, (void**) &dev), env);
  enif_free_resource(dev);
  return enif_make_atom(env, "ok");
}

// new_frame/1 :: (device) -> frame()
static ERL_NIF_TERM
new_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  device_t* dev;
  assert_badarg(enif_get_resource(env, argv[0], device_res, (void**) &dev), env);
  frame_t* frame = enif_alloc_resource(frame_res, sizeof(frame_t));
  frame->_frame = (IplImage*) cvQueryFrame(dev->_device);
  return enif_make_tuple2(env, enif_make_atom(env, "ok"), 
			  enif_make_resource(env, frame));
}

static ERL_NIF_TERM
free_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  frame_t* frame;
  assert_badarg(enif_get_resource(env, argv[0], frame_res, (void**) &frame), env);
  enif_free_resource(frame);
  return enif_make_atom(env, "ok");
}

// query_frame/1 :: (device, frame) -> ok | error
static ERL_NIF_TERM
query_frame(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  device_t* dev;
  frame_t* frame;
  assert_badarg(enif_get_resource(env, argv[0], device_res, (void**) &dev), env);
  assert_badarg(enif_get_resource(env, argv[1], frame_res, (void**) &frame), env);
  frame->_frame = (IplImage*) cvQueryFrame(dev->_device);
  return enif_make_atom(env, "ok");
}

// frame_to_tuple/1 :: (frame) -> {ok, {H, W, NChannels, ImageSize, ImageData}}
static ERL_NIF_TERM
frame_to_tuple(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  ERL_NIF_TERM result;
  frame_t* frame;
  assert_badarg(enif_get_resource(env, argv[0], frame_res, (void**) &frame), env);
  ErlNifBinary* imageData;
  enif_alloc_binary(frame->_frame->imageSize, imageData);
  memcpy(imageData->data, frame->_frame->imageData, frame->_frame->imageSize);
  result = enif_make_tuple5(env,
			    enif_make_int(env, frame->_frame->height),
			    enif_make_int(env, frame->_frame->width),
			    enif_make_int(env, frame->_frame->nChannels),
			    enif_make_int(env, frame->_frame->imageSize),
			    enif_make_binary(env, imageData));
  enif_free(imageData);
  return result;
}

static ErlNifFunc nif_funcs[] = {
  {"any_device", 0, any_device},
  {"free_device", 1, free_device},
  {"new_frame", 1, new_frame},
  {"free_frame", 1, free_frame},
  {"query_frame", 2, query_frame},
  {"frame_to_tuple", 1, frame_to_tuple}
};

ERL_NIF_INIT(ocv_c, nif_funcs, load, NULL, NULL, NULL)
