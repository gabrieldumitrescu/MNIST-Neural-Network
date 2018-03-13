/*MNISTDataset implementation*/

#include "MNISTDataset.h"

MNISTDataset::MNISTDataset(const char* trainLabelsFilename, const char* trainImagesFilename,
                          const char* testLabelsFilename, const char* testImagesFilename)
{

  loadLabels(trainLabelsFilename, trainLabels);
  printf("Loaded %lu labels from file %s!\n", trainLabels.size(), trainLabelsFilename);
  loadImages(trainImagesFilename, trainImages);
  printf("Loaded %lu images from file %s!\n", trainImages.size(), trainImagesFilename);
  loadLabels(testLabelsFilename, testLabels);
  printf("Loaded %lu labels from file %s!\n", testLabels.size(), testLabelsFilename);
 loadImages(testImagesFilename, testImages);
  printf("Loaded %lu images from file %s!\n", testImages.size(), testImagesFilename);
  
}

std::vector<ubyte>& MNISTDataset::getTestLabels()
{
  return testLabels;
}

std::vector<ubyte>& MNISTDataset::getTrainLabels()
{
  return trainLabels;
}

std::vector<MNISTImage>& MNISTDataset::getTrainImages()
{
  return trainImages;
}

std::vector<MNISTImage>& MNISTDataset::getTestImages()
{
  return testImages;
}


/* Read the header of a MNIST labels file */
LabelsHeader MNISTDataset::readLabelsHeader(std::unique_ptr<FILE, FileDeleter> &fLabels)
{
  LabelsHeader header;
  char buf[sizeof(header)];
  size_t nElem = fread(buf,sizeof(header),1,fLabels.get());
  if(nElem !=1) puts("Error reading labels header!");
  char* cursor=buf;
  header.magicNumber=MSBtoLSB(cursor);
  cursor=cursor+sizeof(header.magicNumber);
  header.numLabels=MSBtoLSB(cursor);
  return header;
}

/* Read the header of a MNIST images file */
ImagesHeader MNISTDataset::readImagesHeader(std::unique_ptr<FILE, FileDeleter> &fImages)
{
  ImagesHeader header;
  char buf[sizeof(header)];
  size_t nElem=fread(buf,sizeof(header),1,fImages.get());
  if(nElem !=1) puts("Error reading images header!");
  char* cursor=buf;
  header.magicNumber=MSBtoLSB(cursor);
  cursor=cursor+sizeof(header.magicNumber);
  header.numImages=MSBtoLSB(cursor);
  cursor=cursor+sizeof(header.numImages);
  header.numRows=MSBtoLSB(cursor);
  cursor=cursor+sizeof(header.numRows);
  header.numColumns=MSBtoLSB(cursor);
  return header;
}


/* Read the data of a labels file into a vector of unsigned chars */
bool MNISTDataset::readLabelsData(std::unique_ptr<FILE, FileDeleter> &fLabels,
                    LabelsHeader &header,
                    std::vector<ubyte> &labels)
{
  size_t sz=header.numLabels;
  if(header.magicNumber !=2049)
  {
    printf("Magic number dosen't match a labels file: %d\n", header.magicNumber);
    return false;
  }
  labels.reserve(sz);
  ubyte buf[sz];
  size_t nElem =fread(buf,sz,1,fLabels.get());
  if(nElem !=1) puts("Error reading labels data!");
  for(size_t i=0; i<sz; i++)
  {
    labels.push_back(buf[i]);
  }
  return true;
}

/* Read the data from an images file into a vector of images */
bool MNISTDataset::readImagesData(std::unique_ptr<FILE, FileDeleter> &fImages,
                    ImagesHeader &header,
                    std::vector<MNISTImage> &images)
{
  if(header.magicNumber !=2051)
  {
    printf("Magic number dosen't match a labels file: %d\n", header.magicNumber);
    return false;
  }
  size_t rows=header.numRows;
  size_t cols=header.numColumns;
  size_t imgSz=rows*cols;
  // size_t sz=header.numImages * rows * cols;
  images.reserve(header.numImages);
  for(size_t i=0; i<(size_t)header.numImages; i++)
  {
    ubyte buf[imgSz];
    size_t nElem = fread(buf,imgSz,1,fImages.get());
    if(nElem !=1) printf("Error reading image no %lu data!",i);
    MNISTImage cImg(rows,cols,buf);
    images.push_back(cImg);
  }
  return true;
}

void MNISTDataset::loadLabels(const char* fileName, std::vector<ubyte> &labels)
{
  std::unique_ptr<FILE, FileDeleter> fLabels(fopen(fileName,"rb"));
  LabelsHeader hLabels=readLabelsHeader(fLabels);
  // printf("Magic number: %d\tNumber of labels: %d\n",hLabels.magicNumber,hLabels.numLabels);
  readLabelsData(fLabels,hLabels,labels);
}

void MNISTDataset::loadImages(const char* fileName, std::vector<MNISTImage> &images)
{
  std::unique_ptr<FILE, FileDeleter> fImages(fopen(fileName,"rb"));
  ImagesHeader hImages=readImagesHeader(fImages);
  // printf("Magic number: %d\tNumber of images: %d\n",hImages.magicNumber,hImages.numImages);
  // printf("Number of rows/columns: %d/%d\n",hTrainImages.numRows,hTrainImages.numColumns);
  readImagesData(fImages,hImages,images);
}
