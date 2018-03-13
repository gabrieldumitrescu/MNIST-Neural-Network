/*Image class for MNIST
 *Stores pixels for a single image from the dataset
 */
#ifndef MNIST_Image_H
#define MNIST_Image_H

#include "util.h"

class MNISTImage
{
  int numRows,numColumns;
  std::vector<ubyte> pixels;
public:
  MNISTImage();
  MNISTImage(int nRows, int nCols);
  MNISTImage(int nRows, int nCols, ubyte vals[]);
  ubyte getPixel(int row, int col) const;
  void setPixel(int row, int col, ubyte val);
  void getFloatPixels(std::vector<float> &fPxls) const;
};

#endif //MNIST_Image_H
