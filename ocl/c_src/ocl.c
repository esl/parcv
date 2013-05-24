#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
#include "erl_nif.h"

//------------------------------------------------------------------------------
// Resources
//------------------------------------------------------------------------------
static ErlNifResourceType* platform_r = NULL;
static ErlNifResourceType* device_r = NULL;
static ErlNifResourceType* context_r = NULL;
static ErlNifResourceType* command_queue_r = NULL;
static ErlNifResourceType* mem_r = NULL;
static ErlNifResourceType* program_r = NULL;
static ErlNifResourceType* kernel_r = NULL;
static ErlNifResourceType* event_r = NULL;
static ErlNifResourceType* float_arr_r = NULL;

//------------------------------------------------------------------------------
static void
cleanup(ErlNifEnv* e, void* x)
{
  enif_free(x);
}

static void
event_dtor(ErlNifEnv* e, void* ptr) 
{
  cl_event* evt = (cl_event*) ptr;
  clReleaseEvent(evt[0]);
  enif_free(evt);
}

static void
kernel_dtor(ErlNifEnv* e, void* ptr)
{
  cl_kernel* k = (cl_kernel*) ptr;
  clReleaseKernel(k[0]);
  enif_free(k);
}

static void
program_dtor(ErlNifEnv* e, void* ptr)
{
  cl_program* p = (cl_program*) ptr;
  clReleaseProgram(p[0]);
  enif_free(p);
}

static void
mem_dtor(ErlNifEnv* e, void* ptr) 
{
  cl_mem* m = (cl_mem*) ptr;
  clReleaseMemObject(m[0]);
  enif_free(m);
}

static void
float_arr_dtor(ErlNifEnv* e, void* ptr)
{
  float* arr = (float*) ptr;
  free(arr);
}

static void
command_queue_dtor(ErlNifEnv* e, void* ptr)
{
  cl_command_queue* cmdqp = (cl_command_queue*) ptr;
  clReleaseCommandQueue(cmdqp[0]);
  enif_free(cmdqp);
}

static void
context_dtor(ErlNifEnv* e, void* ptr)
{
  cl_context* ctxt = (cl_context*) ptr;
  clReleaseContext(ctxt[0]);
  enif_free(ctxt);
}

//------------------------------------------------------------------------------
static int
load(ErlNifEnv* e, void** priv, ERL_NIF_TERM load_info)
{
  ErlNifResourceFlags flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
  platform_r = enif_open_resource_type(e, "ocl", "platform", &cleanup, flags, 0);
  device_r = enif_open_resource_type(e, "ocl", "device", &cleanup, flags, 0);
  context_r = enif_open_resource_type(e, "ocl", "context", &context_dtor, flags, 0);
  command_queue_r = enif_open_resource_type(e, "ocl", "command_queue", &command_queue_dtor, flags, 0);
  mem_r = enif_open_resource_type(e, "ocl", "mem", &mem_dtor, flags, 0);
  program_r = enif_open_resource_type(e, "ocl", "program", &program_dtor, flags, 0);
  kernel_r = enif_open_resource_type(e, "ocl", "kernel", &kernel_dtor, flags, 0); 
  event_r = enif_open_resource_type(e, "ocl", "event", &event_dtor, flags, 0);
  float_arr_r = enif_open_resource_type(e, "ocl", "float_arr", &float_arr_dtor, flags, 0);
  return 0;
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------
static ERL_NIF_TERM
make_error(ErlNifEnv* e) 
{
  return enif_make_atom(e, "error");
}

static ERL_NIF_TERM
make_epiphany_error(ErlNifEnv* e)
{
  return enif_make_atom(e, "epiphany_error");
}

static ERL_NIF_TERM
make_ok(ErlNifEnv* e)
{
  return enif_make_atom(e, "ok");
}

static int
get_float_list(ErlNifEnv* e, ERL_NIF_TERM list, float* arr, unsigned len)
{
  ERL_NIF_TERM hd, tl;

  double* tmp = (double*) malloc(sizeof(double)*len);

  int i = 0;
  while(enif_get_list_cell(e, list, &hd, &tl)) {
    if (!enif_get_double(e, hd, &tmp[i])) {
      return -1;
    }
    i++;
    list = tl;
  }

  for(i = 0; i < len; i++)  {
    arr[i] = (float) tmp[i];
    printf("arr[%d] = %f\n", i, arr[i]);
  }

  return 0;
}

static int
get_event_list(ErlNifEnv* e, ERL_NIF_TERM list, cl_event* arr, unsigned len)
{
  printf("Getting event list\n");

  ERL_NIF_TERM hd, tl;

  cl_event** event_rs = (cl_event**) malloc(sizeof(cl_event*));

  int i = 0;
  while(enif_get_list_cell(e, list, &hd, &tl)) {
    if (!enif_get_resource(e, hd, event_r, (void**) &event_rs[i])) {
      return -1;
    }
    arr[i] = event_rs[i][0];
    i++;
    list = tl;
  }
  
  free(event_rs);

  for (i = 0; i < len; i++) {
    int ret, err;
    err = clGetEventInfo(arr[i],CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof(cl_int),
			 &ret, NULL);
    switch (ret) {
    case CL_QUEUED:
      printf("Command was queued\n");
      break;
    case CL_SUBMITTED:
      printf("Command was submitted\n");
      break;
    case CL_RUNNING:
      printf("Command is running\n");
      break;
    case CL_COMPLETE:
      printf("Command is complete\n");
      break;
    default:
      printf("error: command abnormally terminated\n");
      break;
    }
  }

  printf("Got event list\n");

  return 0;
}

static int
get_uint_list(ErlNifEnv* e, ERL_NIF_TERM list, size_t* arr, unsigned len)
{
  ERL_NIF_TERM hd, tl;

  int i = 0;
  while(enif_get_list_cell(e, list, &hd, &tl)) {
    if (!enif_get_uint(e, hd, &arr[i])) {
      return -1;
    }
    i++;
    list = tl;
  }

  for(i = 0; i < len; i++)  {
    printf("arr[%d] = %d\n", i, arr[i]);
  }

  return 0;
}

//------------------------------------------------------------------------------
// Functions
//------------------------------------------------------------------------------
static ERL_NIF_TERM
get_platform_ids(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) 
{
  char buffer[256];
  cl_uint nplatforms;
  cl_platform_id* platforms;
  cl_platform_id platform;

  if (clGetPlatformIDs(0,0,&nplatforms) != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "cl_invalid_value"));

  platforms = (cl_platform_id*) malloc(nplatforms*sizeof(cl_platform_id));

  if (clGetPlatformIDs(nplatforms, platforms, 0) != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "cl_invalid_value"));

  int i;
  for (i = 0; i < nplatforms; i++) {
    platform = platforms[i];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 256, buffer, 0);
    if (!strcmp(buffer, "coprthr-e")) {
      break;
    }
  }

  if (i < nplatforms)
    platform = platforms[i];
  else
    return enif_make_tuple2(e, make_error(e), make_epiphany_error(e));

  cl_platform_id* platform_i = 
    enif_alloc_resource(platform_r, sizeof(cl_platform_id));
  *platform_i = platform;

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, platform_i));
}

static ERL_NIF_TERM
acquire_device(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  cl_uint ndevices;
  cl_device_id* devices;
  cl_device_id dev;
  cl_platform_id* platform;

  if (!enif_get_resource(e, argv[0], platform_r, (void**) &platform)) {
    return enif_make_badarg(e);
  }

  if (clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ACCELERATOR,0,0,&ndevices) != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "device_failure"));

  devices = (cl_device_id*)malloc(ndevices*sizeof(cl_device_id));

  if (clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ACCELERATOR,ndevices,devices,0) != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "device_failure"));
  
  if (ndevices) 
    dev = devices[0];
  else 
    return enif_make_tuple2(e, make_error(e), make_epiphany_error(e));

  cl_device_id* di = enif_alloc_resource(device_r, sizeof(cl_device_id));
  *di = dev;

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, di));
}

static ERL_NIF_TERM
acquire_context(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  printf("Acquiring context\n");

  cl_platform_id* platform;
  cl_device_id* device;
  if (!enif_get_resource(e, argv[0], platform_r, (void**) &platform)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_resource(e, argv[1], device_r, (void**) &device)) {
    return enif_make_badarg(e);
  }

  cl_context_properties ctxtprop[3] = {
    (cl_context_properties) CL_CONTEXT_PLATFORM,
    (cl_context_properties) platform[0],
    (cl_context_properties) 0
  };

  int err;
  cl_context* ctxt = enif_alloc_resource(context_r, sizeof(cl_context));
  *ctxt = clCreateContext(ctxtprop, 1, &(device[0]), 0, 0, &err);
  
  if (err != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "context_fail"));

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, ctxt));
}

static ERL_NIF_TERM
acquire_command_queue(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  printf("Acquiring command queue\n");

  cl_context* context;
  cl_device_id* device;
  if (!enif_get_resource(e, argv[0], context_r, (void**) &context)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_resource(e, argv[1], device_r, (void**) &device)) {
    return enif_make_badarg(e);
  }

  int err;
  cl_command_queue cmdq = clCreateCommandQueue(context[0], device[0], 0, &err);
  cl_command_queue* cmdqp = 
    enif_alloc_resource(command_queue_r, sizeof(cl_command_queue));
  *cmdqp = cmdq;

  if (err != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "command_queue_fail"));
  
  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, cmdqp));
}

static ERL_NIF_TERM
create_float_array(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  unsigned len;
  if (!enif_get_uint(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  float* arr = (float*) enif_alloc_resource(float_arr_r, sizeof(float)*len);
  memset(arr, 0.0, sizeof(float)*len);
  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, arr));
}

static ERL_NIF_TERM
print_float_array(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) 
{
  float* arr;
  if (!enif_get_resource(e, argv[0], float_arr_r, (void**) &arr)) {
    return enif_make_badarg(e);
  }

  unsigned len;
  if (!enif_get_uint(e, argv[1], &len)) {
    return enif_make_badarg(e);
  }

  int i;
  for (i = 0; i < len; i++) {
    printf("arr[%d] = %f\n", i, arr[i]);
  }
      
  return make_ok(e);
}

static ERL_NIF_TERM
create_float_buffer(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  cl_context* context;
  if (!enif_get_resource(e, argv[0], context_r, (void**) &context)) {
    return enif_make_badarg(e);
  }

  unsigned int len;
  if (!enif_get_list_length(e, argv[1], &len)) {
    return enif_make_badarg(e);
  }

  float* arr = (float*) malloc(sizeof(float)*len);
  if (get_float_list(e, argv[1], arr, len) != 0) {
    return enif_make_badarg(e);
  }

  int err;
  cl_mem* memp = enif_alloc_resource(mem_r, sizeof(cl_mem));  
  memp[0] = clCreateBuffer(context[0],CL_MEM_USE_HOST_PTR,len*sizeof(float),arr,&err);

  free(arr);

  if (err != CL_SUCCESS)
    return enif_make_tuple(e, make_error(e), enif_make_atom(e, "create_buffer_failure"));

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, memp));
}

static ERL_NIF_TERM
create_program(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  cl_context* context;
  cl_device_id* device;
  if (!enif_get_resource(e, argv[0], context_r, (void**) &context)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_resource(e, argv[1], device_r, (void**) &device)) {
    return enif_make_badarg(e);
  }

  unsigned len;
  if (!enif_get_list_length(e, argv[2], &len)) {
    return enif_make_badarg(e);
  }

  char buf[len];
  if (len > 0) {
    if (!enif_get_string(e, argv[2], buf, len, ERL_NIF_LATIN1)) {
      return enif_make_badarg(e);
    }
  } else {
    buf[0] = '\0';
  }

  const char* src[1] = { buf };
  size_t src_sz = sizeof(buf);

  int err;
  cl_program* prgp = enif_alloc_resource(program_r, sizeof(cl_program));
  prgp[0] = clCreateProgramWithSource(context[0], 1, (const char**)&src, &src_sz, &err);

  if (err != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "create_program_failure"));

  if (clBuildProgram(prgp[0],1,&device[0],0,0,0) != CL_SUCCESS)
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "build_program_failure"));

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, prgp));
}

static ERL_NIF_TERM
create_kernel(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  cl_program* program;
  cl_device_id* device;
  if (!enif_get_resource(e, argv[0], device_r, (void**) &device)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_resource(e, argv[1], program_r, (void**) &program)) {
    return enif_make_badarg(e);
  }
  unsigned len;
  if (!enif_get_list_length(e, argv[2], &len)) {
    return enif_make_badarg(e);
  }
  char buf[len];
  if (!enif_get_string(e, argv[2], buf, len, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }

  int err;
  cl_kernel* kernel = enif_alloc_resource(kernel_r, sizeof(cl_kernel));
  kernel[0] = clCreateKernel(program[0], "matvecmult_kern", &err);

  switch (err) {
  case CL_INVALID_PROGRAM:
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_program"));
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_program_executable"));
  case CL_INVALID_KERNEL_NAME:
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_kernel_name"));
  case CL_INVALID_KERNEL_DEFINITION:
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_kernel_definition"));
  case CL_INVALID_VALUE:
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_value"));
  case CL_OUT_OF_HOST_MEMORY:
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "out_of_host_memory"));
  }

  printf("kernel valid\n");
  
  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, kernel));
}

static ERL_NIF_TERM
create_event(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  cl_event* event = enif_alloc_resource(event_r, sizeof(cl_event));
  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, event));
}

static ERL_NIF_TERM
set_kernel_arg(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  cl_kernel* kernel;
  if (!enif_get_resource(e, argv[0], kernel_r, (void**) &kernel)) {
    return enif_make_badarg(e);
  }

  int i;
  if (!enif_get_int(e, argv[1], &i)) {
    return enif_make_badarg(e);
  }

  unsigned len;
  if (!enif_get_atom_length(e, argv[2], &len, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }

  char buf[len];
  if (!enif_get_atom(e, argv[2], buf, len+1, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }

  if (strcmp(buf, "cl_uint") == 0) {
    unsigned x;
    if (!enif_get_uint(e, argv[3], &x)) {
      return enif_make_badarg(e);
    }

    if (clSetKernelArg(kernel[0], i, sizeof(cl_uint), &x) != CL_SUCCESS)
      return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "set_kernel_arg_failure"));

    return make_ok(e);
  } else if (strcmp(buf, "cl_mem") == 0) {
    cl_mem* mem;
    if (!enif_get_resource(e, argv[3], mem_r, (void**) &mem)) {
      return enif_make_badarg(e);
    }

    if (clSetKernelArg(kernel[0], i, sizeof(cl_mem), &mem[0]) != CL_SUCCESS)
      return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "set_kernel_arg_failure"));

    return make_ok(e);
  } else {
    printf("strcmp failed\n");
    return enif_make_badarg(e);
  }
}

static ERL_NIF_TERM
enqueue_nd_range_kernel(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  cl_command_queue* cmdqp;
  if (!enif_get_resource(e, argv[0], command_queue_r, (void**) &cmdqp)) {
    return enif_make_badarg(e);
  }

  cl_kernel* kernel;
  if (!enif_get_resource(e, argv[1], kernel_r, (void**) &kernel)) {
    return enif_make_badarg(e);
  }

  unsigned work_dim;
  if (!enif_get_uint(e, argv[2], &work_dim)) {
    return enif_make_badarg(e);
  }

  unsigned glen, llen;
  if (!enif_get_list_length(e, argv[3], &glen)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_list_length(e, argv[4], &llen)) {
    return enif_make_badarg(e);
  }

  size_t* gsize = (size_t*) malloc(sizeof(size_t)*glen);
  if (get_uint_list(e, argv[3], gsize, glen) != 0) {
    return enif_make_badarg(e);
  }

  size_t* lsize = (size_t*) malloc(sizeof(size_t)*llen);
  if (get_uint_list(e, argv[4], lsize, llen) != 0) {
    return enif_make_badarg(e);
  }

  cl_event* event = (cl_event*) enif_alloc_resource(event_r, sizeof(cl_event));
  if (clEnqueueNDRangeKernel(cmdqp[0],kernel[0],work_dim,0,gsize,lsize,
			     0,0,&event[0]) != CL_SUCCESS)
    return enif_make_badarg(e);

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, event));
}

static ERL_NIF_TERM
enqueue_read_buffer(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  // printf("Enqueueing read buffer start\n");
  cl_command_queue* cmdqp;
  if (!enif_get_resource(e, argv[0], command_queue_r, (void**) &cmdqp)) {
    enif_make_badarg(e);
  }
  cl_mem* mem;
  if (!enif_get_resource(e, argv[1], mem_r, (void**) &mem)) {
    enif_make_badarg(e);
  }

  unsigned len;
  if (!enif_get_atom_length(e, argv[2], &len, ERL_NIF_LATIN1)) {
    enif_make_badarg(e);
  }

  char buf[len];
  if (!enif_get_atom(e, argv[2], buf, len+1, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }
  
  unsigned offset;
  if (!enif_get_uint(e, argv[3], &offset)) {
    return enif_make_badarg(e);
  }

  unsigned cb;
  if (!enif_get_uint(e, argv[4], &cb)) {
    return enif_make_badarg(e);
  }

  float* arr;
  if (!enif_get_resource(e, argv[5], float_arr_r, (void**) &arr)) {
    return enif_make_badarg(e);
  }

  cl_event* event;
  if (strcmp(buf, "true") == 0) {
    event = (cl_event*) enif_alloc_resource(event_r, sizeof(cl_event));

    if (clEnqueueReadBuffer(cmdqp[0], mem[0], CL_TRUE, offset, cb, 
			    arr, 0, 0, &event[0]) != CL_SUCCESS) {
      return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "error_read_buffer"));
    }

    return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, event));
  } else if (strcmp(buf, "false") == 0) {
    event = (cl_event*) enif_alloc_resource(event_r, sizeof(cl_event));

    if (clEnqueueReadBuffer(cmdqp[0], mem[0], CL_FALSE, offset, cb, 
			    arr, 0, 0, &event[0]) != CL_SUCCESS) {
      return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "error_read_buffer"));
    }

    return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, event));
  } else {
    return enif_make_badarg(e);
  }
}

static ERL_NIF_TERM
wait_for_events(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  unsigned len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  
  cl_event* events = (cl_event*) malloc(sizeof(cl_event)*len);
  if (get_event_list(e, argv[0], events, len) != 0) {
    return enif_make_badarg(e);
  }
  
  printf("Waiting on %d events\n", len);

  cl_int err = clWaitForEvents(len, events);

  switch (err) {
  case CL_INVALID_VALUE: 
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_value"));
  case CL_INVALID_CONTEXT: 
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_context"));
  case CL_INVALID_EVENT: 
    return enif_make_tuple2(e, make_error(e), enif_make_atom(e, "invalid_event"));
  }

  printf("Waiting on events succeeded\n");

  return make_ok(e);
}
  
ErlNifFunc nif_funcs[] = {
  {"get_platform_ids", 0, get_platform_ids},
  {"acquire_device", 1, acquire_device},
  {"acquire_context", 2, acquire_context},
  {"acquire_command_queue", 2, acquire_command_queue},
  {"create_float_array", 1, create_float_array},
  {"print_float_array", 2, print_float_array},
  {"create_float_buffer", 2, create_float_buffer},
  {"create_program", 3, create_program},
  {"create_kernel", 3, create_kernel},
  {"create_event", 0, create_event},
  {"set_kernel_arg", 4, set_kernel_arg},
  {"enqueue_nd_range_kernel", 5, enqueue_nd_range_kernel},
  {"enqueue_read_buffer", 6, enqueue_read_buffer},
  {"wait_for_events", 1, wait_for_events}
};

ERL_NIF_INIT(ocl, nif_funcs, load, NULL, NULL, NULL);
  
