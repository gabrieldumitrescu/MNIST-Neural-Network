# MNIST-Neural-Network

A Neural Network to recognise digits from the MNIST database.
The goal was to implement solely in C++ without needing a machine learning framework so I can uderstand how machine learning works.
## Getting Started
Only needs a C++11 compatible compiler and the SDL2 library for displaying the images from the MNIST database. 
### Prerequisites
A C++11 compatible compiler such as a current version of gcc.

To install SDL2 on an Ubuntu system just run:
```
 sudo apt-get install sdl2-2.0 sdl2-dev
```
### Installing
Compile by running:
```
make
```
## Running

Running ./MnistAI without arguments trains a new network on 50000 of the test images from the MNIST test database. The trainning is over 30 epochs using an eta value of 0.5 and a batch size of 10. The networks has 3 layers,first being the input layer having 784 neurons corresponding to the 28x28 pixels of the MNIST images. Second layer has 100 neurons and the last output layer 10. Parameteres can be adjusted in the function trainNetwork(...) from the file MnistAI.cc. Using the above parameters the networks gets above 95% prediction accuracy.After trainning the netowrk is evaluated over 10000 validation images and saved as a net[number of correctly predicted images].bin file.Further details can be found in Michael Nielsen's online book "Neural Networks and Deep Learning" which can be found at http://neuralnetworksanddeeplearning.com. This book is fantastic and even though I do not pretend to understand every bit of math in there I could not have written this without it.

Training the network with the default parameters takes around 4 minutes on my Ryzen machine.

After training a network the program can be run with -l followed by the name of a previously saved network to test it against the 10000 test set images from MNIST. The images are displayed and the console list the actual digit and the network predicted one. To navigate use the UP and DOWN keys.

## Author

* **Gabriel Dumitrescu** - *Initial work* -

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgements

*  Michael Nielsen - http://neuralnetworksanddeeplearning.com
*  MNIST Dataset - http://yann.lecun.com/exdb/mnist/



