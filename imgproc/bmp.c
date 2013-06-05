#include "bmp.h"

int main(int argc, char* argv[]) {
  int w = 256;
  int h = 256;
  bmpfile_t* bmp;
  bmp = bmp_create(256, 256, 8);

  for (i = 0; i < w; i++) {
    for (j = 0; j < h; j++) {
      bmp_set_pixel(bmp, i, j, rgb_t);
    }
  }

  bmp_save(bmp, "output.bmp");
  bmp_destroy(bmp);
}

