/**/

#include <SDL2/SDL.h>
#include <sstream>
#include "Network.h"
#include "MNISTImage.h"
#include "MNISTDataset.h"
#include "util.h"
#include <string.h>

/* File names of the MNIST dataset*/
const char* TRAIN_LABELS_NAME="train-labels.idx1-ubyte";
const char* TRAIN_IMAGES_NAME="train-images.idx3-ubyte";
const char* TEST_LABELS_NAME="t10k-labels.idx1-ubyte";
const char* TEST_IMAGES_NAME="t10k-images.idx3-ubyte";

Matrix convertImageToMatrix(const MNISTImage& im)
{
	std::vector<float> cPixels;
	im.getFloatPixels(cPixels);
	Matrix mat(cPixels.size(), 1, cPixels);
	return mat;
}

std::vector<Matrix> convertIm(const std::vector<MNISTImage>& im, 
								size_t start, size_t num)
{
	std::vector<Matrix> res;
	//printf("Converting %lu images.\n",num);
	for (size_t i = start; i < start+num; ++i)
	{
		res.push_back(convertImageToMatrix(im[i]));
	}
	//puts("OK");
	return res;
}

std::vector<Matrix> convertL(const std::vector<ubyte>& labels, 
								size_t start, size_t num)
{
	std::vector<Matrix> res;
	//printf("Converting %lu labels.\n",num);
	for (size_t i = start; i < (start+num); ++i)
	{
		std::vector<float> cLabels(10,0.0f);
		cLabels[(size_t) labels[i]] = 1.0f;
		Matrix cM(cLabels.size(), 1, cLabels);
		res.push_back(cM);
	}
	//puts("OK");
	return res;
}

void shuffle(std::vector<Matrix>& im, std::vector<Matrix>& lab)
{
	for (int i = im.size() -1 ; i >0;--i)
	{
		int sI = std::rand() % (i+1);
		swap(im[i], im[sI]);
		swap(lab[i], lab[sI]);
	}
}

void trainNetwork(std::vector<Matrix>& trainInputs, std::vector<Matrix>& trainLabels,
				  std::vector<Matrix>& validInputs, std::vector<Matrix>& validLabels)
{ 	

	std::vector<size_t> layerSizes = {784,100,10};
	Network nNet(layerSizes);

	puts("Initial performance:");
	nNet.evaluate(validInputs, validLabels);

	int numEpochs=30;
	float eta=.5f;
	size_t batchSz=10;

	for (int i = 0; i < numEpochs; ++i)
	{
		printf("Training epoch %i.....\n",i+1);
		nNet.trainNetwork(trainInputs,trainLabels,eta,batchSz);
		//size_t cp = nNet.evaluate(validInputs, validLabels);
		shuffle(trainInputs,trainLabels);	
	}	

	size_t cp = nNet.evaluate(validInputs, validLabels);
	std::ostringstream os;
	os<<"net"<<cp<<".bin";
	printf("Predicted correctly %lu / %lu\n", cp, validInputs.size());
	nNet.save(os.str());
}


const int SCREEN_WIDTH=640;
const int SCREEN_HEIGHT=480;

void printSDL2Error(const char* msg)
{
	puts(msg);
	printf("SDL_Error: %s\n",SDL_GetError());
}

bool initWindow(SDL_Window* &wnd)
{
	if(SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printSDL2Error("SDL could not initialize!");
		return false;
	}
	wnd=SDL_CreateWindow("SDL2Project", SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,SCREEN_WIDTH,SCREEN_HEIGHT,SDL_WINDOW_SHOWN);
	if(wnd==NULL)
	{
		printSDL2Error("SDL could not create window!");
		return false;
	}
	return true;
}

int ShowImagesSDL(Network& net,std::vector<MNISTImage>& images,
				  std::vector<ubyte>& labels)
{
	SDL_Window* wnd=NULL;
	if(initWindow(wnd))
	{
	    //Create renderer for window
	    SDL_Renderer* renderer =NULL;
	    renderer=SDL_CreateRenderer( wnd, -1, SDL_RENDERER_ACCELERATED );
    	if( renderer == NULL )
	    {
	        printf( "Renderer could not be created! SDL Error: %s\n", SDL_GetError() );
	        return 1;
	    }
			//gScreenSurface = SDL_GetWindowSurface(wnd);
	    bool quit=false;
	    SDL_Event e;
	    int cImageIdx=0;
	    bool idxChgd=true;
	    while(!quit)
	    {

	      while( SDL_PollEvent(&e) )
	      {
	        if(e.type == SDL_QUIT)
	          quit=true;
	        else if(e.type == SDL_KEYDOWN)
	        {
	            switch( e.key.keysym.sym )
	            {
	              case SDLK_DOWN:
		              cImageIdx++;
					  if(cImageIdx>=(int)images.size()) cImageIdx=0;
		              idxChgd=true;
		              break;
	              case SDLK_UP:
		              cImageIdx--;
					  if(cImageIdx<0) cImageIdx=images.size()-1;
		              idxChgd=true;
					  break;
	            }
	        }
	      }
	      //Clear screen
	      SDL_SetRenderDrawColor( renderer, 0xFF, 0xFF, 0xFF, 0xFF );
	      SDL_RenderClear( renderer );
	      //Render MNISTImage to screen
	      MNISTImage cImage=images[cImageIdx];
	      int pixelSize=7;
	      int startx=((SCREEN_WIDTH - pixelSize * 28) / 2),
	          starty=((SCREEN_HEIGHT - pixelSize * 28) / 2);
	      for (int j=0;j<28;j++)
	      {
	        startx=((SCREEN_WIDTH - pixelSize * 28) / 2);
	        for (int i=0;i<28;i++)
	        {
	          ubyte cAlpha=cImage.getPixel(i,j);
	          SDL_Rect fillRect = {startx, starty, pixelSize, pixelSize };
	          SDL_SetRenderDrawColor( renderer, 0xff - cAlpha,0xff - cAlpha, 0xff -cAlpha, 0xFF );
	          SDL_RenderFillRect( renderer, &fillRect );
	          startx+=pixelSize;
	        }
	        starty+=pixelSize;
	      }
	      //Print the label corresponding to the image
	      if(idxChgd)
	      {
	        int lbl=(int)(labels[cImageIdx]);
	        printf("Label for image no. %d is %d.\n", cImageIdx,lbl);
	        Matrix mat=convertImageToMatrix(cImage);
	        printf("Network predicted label is %lu.\n", net.predict(mat));
	        idxChgd=false;
	      }
	      //Update screen
	      SDL_RenderPresent( renderer );
	    }
      //Destroy window
    SDL_DestroyRenderer( renderer );
    SDL_DestroyWindow( wnd);
    wnd = NULL;
    renderer = NULL;
  }
//Quit SDL subsystems
  SDL_Quit();
  return 0;
}

int main(int argc, char* argv[])
{
	bool trainNewNetwork = true;
	char* netName;
	if(argc>1)
	{
		if(strcmp(argv[1],"-l")==0)
		{
			if(argc<3) 
			{
				puts("Please provide the name of the file to load!");
				return 1;
			}
			else
			{
				netName = argv[2];
				trainNewNetwork = false;
			}
		}
	}
	MNISTDataset data(TRAIN_LABELS_NAME,TRAIN_IMAGES_NAME,
					  TEST_LABELS_NAME,TEST_IMAGES_NAME);

	std::vector<MNISTImage> images(data.getTrainImages());
	std::vector<ubyte> labels(data.getTrainLabels());
     
    size_t trainSz=50000;
    size_t validSz=images.size() - trainSz;

	std::vector<Matrix> trainInputs(convertIm(images,0,trainSz));
	std::vector<Matrix> trainLabels(convertL(labels,0,trainSz)); 

	std::vector<Matrix> validInputs(convertIm(images,trainSz,validSz));
	std::vector<Matrix> validLabels(convertL(labels,trainSz,validSz));

	

	if(trainNewNetwork)
	{
		trainNetwork(trainInputs,trainLabels,validInputs,validLabels);
	}
	else
	{
		Network nNet(netName);
		size_t cp = nNet.evaluate(validInputs, validLabels);
		printf("Network correctly predicts %lu / %lu images from the validation set.\n", cp, validInputs.size());

		puts("Showing images from the test dataset");
		std::vector<MNISTImage> testImages(data.getTestImages());
		std::vector<ubyte> testLabels(data.getTestLabels());
		ShowImagesSDL(nNet, testImages,testLabels);

	}

	return 0;
}
