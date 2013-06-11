#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "imgproc.h"
#include "erl_nif.h"

//------------------------------------------------------------------------------
static ErlNifResourceType* image_r = NULL;

static void 
image_cleanup (ErlNifEnv* e, void* arg) {
  free(arg);
}

static int
load(ErlNifEnv* e, void** priv, ERL_NIF_TERM load_info) {
  ErlNifResourceFlags flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;
  image_r = enif_open_resource_type(e, "imgproc_nif", "image", 
				    &image_cleanup, flags, 0);
  return 0;
}

//------------------------------------------------------------------------------
// NIF
//------------------------------------------------------------------------------
static ERL_NIF_TERM
initialize(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get string (main kernel)
  unsigned len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  char* str = (char*) malloc(sizeof(char) * len);
  if (!enif_get_string(e, argv[0], str, len+1, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }
  str[len-1] = '\0';

  printf("initialising NIF with %s\n", str);

  cl_initialize("kernels/edge_detect.cl");
  return enif_make_atom(e, "ok");
}

static ERL_NIF_TERM
teardown(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  cl_teardown();
  return enif_make_atom(e, "ok");
}

static ERL_NIF_TERM
transform(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get byte* handle
  byte* image;
  if (!enif_get_resource(e, argv[0], image_r, (void**) image)) {
    return enif_make_badarg(e);
  }

  // get image_width
  int image_w, image_h;
  if (!enif_get_int(e, argv[1], &image_w)) {
    return enif_make_badarg(e);
  }

  // get image_height
  if (!enif_get_int(e, argv[2], &image_h)) {
    return enif_make_badarg(e);
  }

  cl_transform(image, image_w, image_h);

  return enif_make_atom(e, "ok");
}

static ERL_NIF_TERM
read_png(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get filename
  unsigned len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  char str[len];
  if (!enif_get_string(e, argv[0], str, len, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }

  // get image_width
  int image_w, image_h;
  if (!enif_get_int(e, argv[1], &image_w)) {
    return enif_make_badarg(e);
  }

  // get image_height
  if (!enif_get_int(e, argv[2], &image_h)) {
    return enif_make_badarg(e);
  }

  byte* image = png_read(str, image_w, image_h);

  // ret byte* handle
  return enif_make_tuple(e, enif_make_resource(e, image));
}

static ERL_NIF_TERM
write_png(ErlNifEnv* e, int argc, const ERL_NIF_TERM argv[]) {
  // get filename
  unsigned len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }
  char buf[len];
  if (!enif_get_string(e, argv[0], buf, len, ERL_NIF_LATIN1)) {
    return enif_make_badarg(e);
  }

  // get byte* handle
  byte* image;
  if (!enif_get_resource(e, argv[1], image_r, (void**) image)) {
    return enif_make_badarg(e);
  }

  // get image_width
  int image_w, image_h;
  if (!enif_get_int(e, argv[2], &image_w)) {
    return enif_make_badarg(e);
  }
  // get image_height
  if (!enif_get_int(e, argv[3], &image_h)) {
    return enif_make_badarg(e);
  }

  png_write(buf, image, image_w, image_h);

  return enif_make_atom(e, "ok");
}

static ERL_NIF_TERM
list_to_image(ErlNifEnv* e, int argc, ERL_NIF_TERM argv[]) 
{
  ERL_NIF_TERM list, hd, tl;

  uint len;
  if (!enif_get_list_length(e, argv[0], &len)) {
    return enif_make_badarg(e);
  }

  int i;
  int* tmp = (int*) malloc(sizeof(int) * len);
  while (enif_get_list_cell(e, list, &hd, &tl)) {
    if (!enif_get_int(e, hd, &tmp[i])) {
      return make_badarg(e);
    }
    i += 1;
    list = tl;
  }
  
  byte* image = (byte*) enif_alloc_resource(image_r, sizeof(byte) * len);
  for (i = 0; i < len; i++)
    image[i] = tmp[i] & 0x000000FF;

  return enif_make_tuple2(e, enif_make_atom(e, "ok"), enif_make_resource(e, image));
}

static ErlNifFunc nif_funcs[] = {
  {"initialize", 1, initialize},
  {"teardown", 0, teardown},
  {"transform", 3, transform},
  {"read_png", 3, read_png},
  {"write_png", 4, write_png}
};

ERL_NIF_INIT(imgproc_nif, nif_funcs, load, NULL, NULL, NULL);
