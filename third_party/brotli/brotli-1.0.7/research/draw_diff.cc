/* Copyright 2016 Google Inc. All Rights Reserved.
   Author: zip753@gmail.com (Ivan Nikulin)

   Distributed under MIT license.
   See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
*/

/* Tool for drawing diff PPM images between two input PGM images. Normally used
   with backward reference histogram drawing tool. */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  /* exit, EXIT_FAILURE */
#include <vector>

#if !defined(CHECK)
#define CHECK(X) if (!(X)) exit(EXIT_FAILURE);
#endif

typedef uint8_t* ScanLine;
typedef ScanLine* Image;

void ReadPGM(FILE* f, Image* image, size_t* height, size_t* width) {
  int colors;
  CHECK(fscanf(f, "P5\n%lu %lu\n%d\n", width, height, &colors) == 3);
  assert(colors == 255);
  ScanLine* lines = new ScanLine[*height];
  *image = lines;
  for (int i = *height - 1; i >= 0; --i) {
    ScanLine line = new uint8_t[*width];
    lines[i] = line;
    CHECK(fread(line, 1, *width, f) == *width);
  }
}

void CalculateDiff(int** diff, Image image1, Image image2,
                   size_t height, size_t width) {
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      diff[i][j] = static_cast<int>(image1[i][j]) - image2[i][j];
    }
  }
}

void DrawDiff(int** diff, Image image1, Image image2,
              size_t height, size_t width, FILE* f) {
  int max = -1234;
  int min = +1234;
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      if (max < diff[i][j]) max = diff[i][j];
      if (min > diff[i][j]) min = diff[i][j];
      int img_min = std::min(255 - image1[i][j], 255 - image2[i][j]);
      if (max < img_min) max = img_min;
    }
  }

  int abs_max = -min;
  if (abs_max < max) abs_max = max;

  fprintf(f, "P6\n%lu %lu\n%d\n", width, height, abs_max);

  uint8_t* row = new uint8_t[3 * width];
  for (int i = height - 1; i >= 0; --i) {
    for (int j = 0; j < width; ++j) {
      int min_val = std::min(255 - image1[i][j], 255 - image2[i][j]);
      int max_val = std::max(min_val, abs(diff[i][j]));
      if (diff[i][j] > 0) { /* red */
        row[3 * j + 0] = abs_max - max_val + diff[i][j];
        row[3 * j + 1] = abs_max - max_val;
        row[3 * j + 2] = abs_max - max_val + min_val;
      } else { /* green */
        row[3 * j + 0] = abs_max - max_val;
        row[3 * j + 1] = abs_max - max_val - diff[i][j];
        row[3 * j + 2] = abs_max - max_val + min_val;
      }
    }
    fwrite(row, 1, 3 * width, f);
  }
  delete[] row;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    printf("usage: %s pgm1 pgm2 diff_ppm_path\n", argv[0]);
    return 1;
  }

  Image image1, image2;
  size_t h1, w1, h2, w2;

  FILE* fimage1 = fopen(argv[1], "rb");
  ReadPGM(fimage1, &image1, &h1, &w1);
  fclose(fimage1);

  FILE* fimage2 = fopen(argv[2], "rb");
  ReadPGM(fimage2, &image2, &h2, &w2);
  fclose(fimage2);

  if (!(h1 == h2 && w1 == w2)) {
    printf("Images must have the same size.\n");
    return 1;
  }

  int** diff = new int*[h1];
  for (size_t i = 0; i < h1; ++i) diff[i] = new int[w1];
  CalculateDiff(diff, image1, image2, h1, w1);

  FILE* fdiff = fopen(argv[3], "wb");
  DrawDiff(diff, image1, image2, h1, w1, fdiff);
  fclose(fdiff);

  return 0;
}
