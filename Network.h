/* Neural network for MNIST digits recognition
 */
#ifndef NETWORK_H
#define NETWORK_H

#include <chrono>
#include <random>
#include <math.h>
#include "Matrix.h"
#include "util.h"

//sigmoid activation function S
float sigmoid(float val);
//derivative of sigmoid function - S'
float sigmoidPrime(float val);

struct NetworkFileHeader
{
  int magicNum;
  int numLayers;
};

typedef enum CostFunctionT
{
  QuadraticCostFunction,
  CrossEntropyCostFunction
} CostFunction;

typedef enum RegularizationTypeT
{
  NoRegularization,
  L1Regularization,
  L2Regularization,
  DropoutRegularization
} RegularizationType;


struct NetworkParameters
{
  CostFunction costFn;
  RegularizationType regType;
  bool weightInitOpt;
  NetworkParameters():
    costFn(CrossEntropyCostFunction),
    regType(L2Regularization),
    weightInitOpt(true)
    {}
  NetworkParameters(CostFunction c,RegularizationType t,bool weightOpt):
    costFn(c),
    regType(t),
    weightInitOpt(weightOpt)
    {}
};

/* Neural network class for MNIST digits recognition */
class Network
{
    class Layer
    {
      Matrix m_weights, m_biases,
             m_weightedInputs,
             m_activations,m_errors,
             m_biasesErr,m_weightsErr;
      bool m_outLayer;
    public:
      Layer(Matrix& weights, Matrix& biases,bool outLayer=false);
      void printBiases() const;
      void printWeights() const;
      void printWeightedInputs() const;

      Matrix feedForward(Matrix& prevAct);
      void gradientDescent(Matrix& prevAct);
      void updateParams(float etaM, float reg_factor);

      Matrix getWeightedInputs() const;
      void setErrors(const Matrix& err);
      const Matrix& getErrors() const;
      const Matrix& getWeights() const;
      const Matrix& getBiases() const;
      const Matrix& getActivations() const;
    };

    NetworkParameters config;
    std::vector<size_t> layerSizes; // vector of sizes of the neuron layers
    std::vector<Layer*> layers;
  public:
    Network(std::vector<size_t> &sizes,NetworkParameters cfg);
    Network(const std::string& filename);

    //print the biases vector and the weights matrix for each layer
    void printNetwork() const;

    /*float getBiasAt(size_t layer, size_t neuronI) const;
    float getWeightAt(size_t layer, size_t Row, size_t Col) const;*/



    void printBiases() const;
    void printWeights() const;
    void printWeightedInputs() const;
    void printActivations() const;



    /* Compute the resulting activation vector given the input vector of activations */
    Matrix feedForward(Matrix& input);
    Matrix backpropagateError(Matrix& input, Matrix& desiredOut);
    void gradientDescent(Matrix& input, Matrix& desiredOut);
    void trainNetwork(std::vector<Matrix>& inputs,
                      std::vector<Matrix>& outputs,
                      float eta, size_t batchSz,float reg_factor);
    size_t predict(Matrix& input);
    size_t evaluate(std::vector<Matrix>& inputs,
                       std::vector<Matrix>& outputs);

    void save(const std::string& filename) const;
    void load(const std::string& filename);

};

#endif //NETWORK_H
