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
  return enif_make_atom(e, "ephiphany_error");
}

static ERL_NIF_TERM
make_ok(ErlNifEnv* e)
{
  return enif_make_atom(e, "ok");
}

static int
get_event_list(ErlNifEnv* e, ERL_NIF_TERM list, cl_event* arr, unsigned len)
{
  ERL_NIF_TERM hd, tl;

  cl_event** arr0 = (cl_event**) malloc(sizeof(cl_event*)*len);
  arr = (cl_event*) malloc(sizeof(cl_event)*len);

  int i = 0;
  while(enif_get_list_cell(e, list, &hd, &tl)) {
    if (!enif_get_resource(e, hd, event_r, (void**) &arr0[i])) {
      return -1;
    }
    i++;
    list = tl;
  }

  int j;
  for(j = 0; j < len; j++) {
    arr[j] = arr0[j][0];
  }

  return 0;
}

static int
get_uint_list(ErlNifEnv* e, ERL_NIF_TERM list, unsigned* arr, unsigned len)
{
  ERL_NIF_TERM hd, tl;

  arr = (unsigned*) malloc(sizeof(unsigned)*len);

  int i = 0;
  while(enif_get_list_cell(e, list, &hd, &tl)) {
    if (!enif_get_uint(e, hd, &arr[i])) {
      return -1;
    }
    i++;
    list = tl;
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

  clGetPlatformIDs(0,0,&nplatforms);
  platforms = (cl_platform_id*) malloc(nplatforms*sizeof(cl_platform_id));
  clGetPlatformIDs(nplatforms, platforms, 0);

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

  clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ACCELERATOR,0,0,&ndevices);
  devices = (cl_device_id*)malloc(ndevices*sizeof(cl_device_id));
  clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_ACCELERATOR,ndevices,devices,0);
  
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

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, ctxt));
}

static ERL_NIF_TERM
acquire_command_queue(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
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
  cl_command_queue* cmdqp = enif_alloc_resource(command_queue_r, sizeof(cl_command_queue));
  *cmdqp = cmdq;
  
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

  ERL_NIF_TERM list = argv[1];
  unsigned int len;
  ERL_NIF_TERM hd, tl;

  if (!enif_get_list_length(e, list, &len)) {
    return enif_make_badarg(e);
  }

  float* arr = (float*) malloc(sizeof(float)*len);

  int i = 0;
  while(enif_get_list_cell(e, list, &hd, &tl)) {
    if (!enif_get_double(e, hd, &arr[i])) {
      return enif_make_badarg(e);
    }
    i++;
    list = tl;
  }

  int err;
  cl_mem* bufp = enif_alloc_resource(mem_r, sizeof(cl_mem));  
  cl_mem buf = clCreateBuffer(context[0],CL_MEM_USE_HOST_PTR,len*sizeof(float),(float*)arr,&err);
  *bufp = buf;

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, bufp));
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
  cl_program prg = clCreateProgramWithSource(context[0], 1, (const char**)&src, &src_sz, &err);

  clBuildProgram(prg,1,&(device[0]),0,0,0);

  *prgp = prg;

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
  cl_kernel krn = clCreateKernel(program[0], buf, &err);
  *kernel = krn;
  
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

    clSetKernelArg(kernel[0], i, sizeof(cl_uint), &x);
    return make_ok(e);
  } else if (strcmp(buf, "cl_mem") == 0) {
    cl_mem* mem;
    if (!enif_get_resource(e, argv[3], mem_r, (void**) &mem)) {
      return enif_make_badarg(e);
    }

    clSetKernelArg(kernel[0], i, sizeof(cl_mem), &mem[0]);
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
  cl_kernel* kernel;
  unsigned work_dim;
  if (!enif_get_resource(e, argv[0], command_queue_r, (void**) &cmdqp)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_resource(e, argv[1], kernel_r, (void**) &kernel)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_uint(e, argv[2], &work_dim)) {
    return enif_make_badarg(e);
  }

  unsigned global_work_size_len, local_work_size_len;
  if (!enif_get_list_length(e, argv[3], &global_work_size_len)) {
    return enif_make_badarg(e);
  }
  if (!enif_get_list_length(e, argv[4], &local_work_size_len)) {
    return enif_make_badarg(e);
  }

  unsigned* global_work_size;
  if (get_uint_list(e, argv[3], global_work_size, global_work_size_len) != 0) {
    return enif_make_badarg(e);
  }

  unsigned* local_work_size;
  if (get_uint_list(e, argv[4], local_work_size, local_work_size_len) != 0) {
    return enif_make_badarg(e);
  }

  cl_event event;
  clEnqueueNDRangeKernel(cmdqp[0], kernel[0], work_dim, 0,
			 global_work_size, local_work_size,
			 0,0,&event);

  cl_event* evt = (cl_event*) enif_alloc_resource(event_r, sizeof(cl_event));
  *evt = event;

  return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, evt));
}

static ERL_NIF_TERM
enqueue_read_buffer(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[])
{
  // printf("Enqueueing read buffer start\n");
  cl_command_queue* cmdqp;
  if (!enif_get_resource(e, argv[0], command_queue_r, (void**) cmdqp)) {
    enif_make_badarg(e);
  }
  cl_mem* mem;
  if (!enif_get_resource(e, argv[1], mem_r, (void**) mem)) {
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
    //  printf("Enqueueing read buffer true\n");
    event = (cl_event*) enif_alloc_resource(event_r, sizeof(cl_event));
    cl_event evt;
    clEnqueueReadBuffer(cmdqp[0], mem[0], CL_TRUE, offset, cb, arr, 0, 0, &evt);
    *event = evt;
    return enif_make_tuple2(e, make_ok(e), enif_make_resource(e, event));
  } else if (strcmp(buf, "false") == 0) {
    //    printf("Enqueueing read buffer false\n");
    event = (cl_event*) enif_alloc_resource(event_r, sizeof(cl_event));
    cl_event evt;
    clEnqueueReadBuffer(cmdqp[0], mem[0], CL_FALSE, offset, cb, arr, 0, 0, &evt);
    *event = evt;
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
  
  cl_event* events;
  if (get_event_list(e, argv[0], events, len) != 0) {
    return enif_make_badarg(e);
  }
  
  printf("Waiting on %d events\n", len);

  clWaitForEvents(len, events);

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
  
