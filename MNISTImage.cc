#include "MNISTImage.h"


MNISTImage::MNISTImage():
  numRows(0),numColumns(0)
{

}

MNISTImage::MNISTImage(int nRows, int nCols):
  numRows(nRows),numColumns(nCols)
{
  pixels.reserve(numRows*numColumns);
}

MNISTImage::MNISTImage(int nRows, int nCols, ubyte vals[]):
  numRows(nRows),numColumns(nCols),
  pixels(vals, vals+(nRows*nCols))
{
  pixels.reserve(numRows*numColumns);
}


ubyte MNISTImage::getPixel(int row, int col) const
{
  return pixels[col*numColumns + row];
}

void MNISTImage::setPixel(int row,int col, ubyte val)
{
  pixels[col*numColumns + row]=val;
}

void MNISTImage::getFloatPixels(std::vector<float> &fPxls) const
{
  fPxls.clear();
  for(size_t i=0; i<pixels.size();++i)
  {
    fPxls.push_back((float)pixels.at(i)/255.0f );
  }
}
