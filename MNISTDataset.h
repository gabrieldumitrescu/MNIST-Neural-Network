/*MNISTDataset class loads the images and lables from the mnist dataset files*/

#ifndef MNIST_DATASET_H
#define MNIST_DATASET_H

#include <vector>

#include "util.h"
#include "MNISTImage.h"

/* Header for a MNIST labels file
 * magicNumber should always be 0x801 (2049) for a labels file
 */
struct LabelsHeader
{
  int magicNumber;
  int numLabels;
};

/* Header for a MNIST images file
 * magicNumber should always be 0x803 (2051) for a labels file
 * numRows and numColumns for MNIST should both be 28
 */
struct ImagesHeader
{
  int magicNumber;
  int numImages;
  int numRows;
  int numColumns;
};

class MNISTDataset
{
    std::vector<ubyte> trainLabels;
    std::vector<MNISTImage> trainImages;
    std::vector<ubyte> testLabels;
    std::vector<MNISTImage> testImages;
    /* Read the header of a MNIST labels file */
    LabelsHeader readLabelsHeader(std::unique_ptr<FILE, FileDeleter> &fLabels);

    ImagesHeader readImagesHeader(std::unique_ptr<FILE, FileDeleter> &fImages);

    /* Read the data of a labels file into a vector of unsigned chars */
    bool readLabelsData(std::unique_ptr<FILE, FileDeleter> &fLabels,
                                        LabelsHeader &header,
                                        std::vector<ubyte> &labels);

    /* Read the data from an images file into a vector of images */
    bool readImagesData(std::unique_ptr<FILE, FileDeleter> &fImages,
                        ImagesHeader &header,
                        std::vector<MNISTImage> &images);

    void loadLabels(const char* fileName, std::vector<ubyte> &labels);

    void loadImages(const char* fileName, std::vector<MNISTImage> &images);

    /* Prevent copying */
    MNISTDataset(MNISTDataset const &);
    MNISTDataset& operator=(MNISTDataset const &);
    
  public:
    MNISTDataset(const char* trainLabelsFilename, const char* trainImagesFilename,
                              const char* testLabelsFilename, const char* testImagesFilename);
    std::vector<ubyte>& getTrainLabels();
    std::vector<ubyte>& getTestLabels();
    std::vector<MNISTImage>& getTrainImages();
    std::vector<MNISTImage>& getTestImages();



};


#endif // MNIST_DATASET_H
