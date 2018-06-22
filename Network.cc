/*Network class implem.*/

#include "Network.h"


float sigmoid(float val)
{
  float result=1.0f / (1.0f + exp(-val));
  return result;
}

float sigmoidPrime(float val)
{
  float result=sigmoid(val) * (1-sigmoid(val));
  return result;
}




Network::Layer::Layer(Matrix& weights, Matrix& biases, bool outLayer):
  m_weights(weights), m_biases(biases), 
  m_weightedInputs(weights.getNumRows(),1),
  m_activations(weights.getNumRows(),1),
  m_errors(weights.getNumRows(),1),
  m_biasesErr(biases.getNumRows(),1),
  m_weightsErr(weights.getNumRows(),weights.getNumCol()),
  m_outLayer(outLayer)
{
}


void Network::Layer::printWeights() const
{
  m_weights.print();
  puts("");
}

void Network::Layer::printBiases() const
{
  m_biases.print();
  puts("");
}

void Network::Layer::printWeightedInputs() const
{
  m_weightedInputs.print();
  puts("");
}


Matrix Network::Layer::feedForward(Matrix& prevAct)
{
    m_weightedInputs=m_weights * prevAct + m_biases;
    Matrix result(m_weightedInputs);
    result.map(sigmoid);
    m_activations=result;
    return result;
}

void Network::Layer::gradientDescent(Matrix& prevAct)
{
  m_biasesErr = m_biasesErr + m_errors;
  m_weightsErr= m_errors *(prevAct.transpose());
}

void Network::Layer::updateParams(float etaM,float reg_factor)
{
  m_biases=m_biases - (m_biasesErr * etaM);
  m_weights = (m_weights * reg_factor) - (m_weightsErr * etaM);
  Matrix nBErr(m_biases.getNumRows(), 1);
  Matrix nWErr(m_weights.getNumRows(), m_weights.getNumCol());
  m_biasesErr = nBErr;
  m_weightsErr = nWErr;
}


Matrix Network::Layer::getWeightedInputs() const
{
  return m_weightedInputs;
}


void Network::Layer::setErrors(const Matrix& err)
{
  m_errors=err;
}

const Matrix& Network::Layer::getErrors() const
{
  return m_errors;
}

const Matrix& Network::Layer::getWeights() const
{
  return m_weights;
}

const Matrix& Network::Layer::getBiases() const
{
  return m_biases;
}

const Matrix& Network::Layer::getActivations() const
{
  return m_activations;
}



//-----------------------------------Layer--------------------------------------------------





/* Construct the network from the vector containing the size of each layer
    Total number of layers is sizes.size();
    The first layer is the input layer and has no weights and biases, activation values
    are provided as input for the network.
*/
Network::Network(std::vector<size_t> &sizes,NetworkParameters cfg):
  layerSizes(sizes),
  config(cfg)
  {
    size_t numNeurons=0;
    size_t numWeights=0;
    for (size_t i = 0; i < sizes.size(); i++)
    {
      numNeurons+=sizes[i];
      if(i>0)
        numWeights+=sizes[i-1] * sizes[i];
    }
    // construct a trivial random generator engine from a time-based seed:
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //seed=2;//fixed values for debug
    std::default_random_engine generator (seed);
    std::normal_distribution<float> distribution (0.0f,1.0f);

    float weights[numWeights];
    float biases[numNeurons-sizes[0]];
    /* create random weights and biases */
    for (size_t i=0; i<numWeights; ++i)
      weights[i]=distribution(generator);
    for (size_t i=0; i<numNeurons-sizes[0]; ++i) // no biases for the first layer
      biases[i]=distribution(generator);
    
    // Create layers

    float* cW=weights;
    float* cB=biases;
    for (size_t i=0; i<sizes.size(); ++i)
    {
        if(i==0) 
        { 
          Matrix weightsM(sizes[i],1);
          Matrix biasesM(sizes[i],1);
          Layer* current = new Layer(weightsM, biasesM);
          layers.push_back(current);
        }
        else
        {
          Matrix weightsM(sizes[i],sizes[i-1],cW);
          //optimize weights initialization
          if(config.weightInitOpt)
          {
            weightsM=weightsM * (1.0f / sqrt((float)sizes[i-1]));
          }
          Matrix biasesM(sizes[i],1,cB);
          Layer* current = new Layer(weightsM, biasesM, (i==sizes.size() - 1));
          layers.push_back(current);
          cW+=(sizes[i]*sizes[i-1]);
          cB+=(sizes[i]);
        }
        //printf("Layer %lu ok.\n",i);
    }
}

/* Construct the network from a save file.
*/
Network::Network(const std::string& filename)
{
  load(filename);
}

void Network::printNetwork() const
{
  for (size_t i = 1; i < layers.size(); i++)
  {
      Layer* c=layers.at(i);
      printf("Layer %lu biases:\n",i);
      c->printBiases();
      printf("Layer %lu weights:\n",i);
      c->printWeights();
  }
}


void Network::printBiases() const
{
  for (size_t i = 1; i < layers.size(); i++)
  {
    Layer* c=layers.at(i);
    printf("Layer %lu biases:",i);
    c->printBiases();
  }
}

void Network::printWeightedInputs() const
{
  for (size_t i = 1; i < layers.size(); i++)
  {
    Layer* c=layers.at(i);
    printf("Layer %lu weighted sums:\n",i);
    c->printWeightedInputs();
  }
}

void Network::printWeights() const
{
  for (size_t i = 1; i < layers.size(); i++)
  {
    printf("Layer %lu weights:\n",i);
    Layer* c=layers.at(i);
    c->printWeights();
  }
  puts("");

}

Matrix Network::feedForward(Matrix& input)
{
  //puts("Feeding forward...");
  Matrix cInput(input);
  for (size_t i = 1; i < layers.size(); i++)
  {
    Layer* cL=layers.at(i);
    cInput = cL->feedForward(cInput);
  }
  return cInput;
}

Matrix Network::backpropagateError(Matrix& input, Matrix& desiredOut)
{
  Layer* cLayer=layers[layers.size() - 1];

  /*
    The error for the last layer is calculated as the hadamard product
    of the folowing vectors:
      1. (a(L) - y)
      2. SigmoidPrime(z)
    For the CrossEntropy cost function the error is just 1.
  */
  //1.
  Matrix first=feedForward(input) - desiredOut;
  //2.
  Matrix second=cLayer->getWeightedInputs();
  if(config.costFn == QuadraticCostFunction)
  {
    second.map(sigmoidPrime);
    //hadamard
    first = first.hadamard(second);  
  }
  cLayer->setErrors(first);
  for (size_t i = layers.size() - 2; i >0; --i)
  {
    /* For backpropagation calculate the error in the current layer
      as the hadamard product of the vectors:
        1. Transposed matrix of next layer * next layer error vector
        2. SigmoidPrime(z) for current layer
    */
    //1.
    Layer* prevLayer=layers[i+1];
    first = prevLayer->getWeights();
    first=(first.transpose() * prevLayer->getErrors());
    //2.
    cLayer = layers[i];
    second=cLayer->getWeightedInputs();
    second.map(sigmoidPrime);
    //hadamard
    first = first.hadamard(second);
    cLayer->setErrors(first);
  }



  return first;
}

void Network::gradientDescent(Matrix& input,
                              Matrix& desired)
{
  backpropagateError(input,desired);  
  
  Matrix prevAct(input);
  for (size_t i = 1; i < layers.size(); ++i)
  {
    Layer* cLayer=layers[i];
    cLayer->gradientDescent(prevAct);
    prevAct=cLayer->getActivations();
  }
}

void Network::trainNetwork(std::vector<Matrix>& inputs,
                           std::vector<Matrix>& outputs,
                           float eta, size_t batchSz, float reg_factor)
{
  if(config.regType == NoRegularization)
  {
    reg_factor=1.0;
  }
  for (size_t i = 0; i < inputs.size(); i+=batchSz)
  {
    for (size_t j = 0; j < batchSz; ++j)
    {
      gradientDescent(inputs[i+j], outputs[i+j]);
    }
    for (size_t j = 1; j < layers.size(); ++j)
    {
      Layer* cLayer=layers[j];
      float fact=eta/((float) batchSz);
      cLayer->updateParams(fact, reg_factor);
    }
  }
}

size_t Network::predict(Matrix& input)
{
  Matrix out = feedForward(input);
  size_t idx = 0;
  float max = out.getAt(0,0);
  for (size_t i = 0; i < out.getNumRows(); ++i)
  {
    float c = out.getAt(i,0);
    if(c>max)
    {
      max=c;
      idx=i;
    }
  }
  return idx;
}

size_t Network::evaluate(std::vector<Matrix>& inputs,
                       std::vector<Matrix>& outputs)
{
  size_t numIn=inputs.size();
  size_t correct=0;
  printf("Evaluating %lu inputs....\n",numIn);
  for (size_t i = 0; i < numIn; ++i)
  {
    size_t cPred=predict(inputs[i]);
    if(outputs[i].getAt(cPred,0) ==1.0f)
    {
      correct++;
    }
    //printf("Finished at %lu\n",i);
  }
  float percent=(((float)correct * 100) / ((float)numIn));
  printf("\t\t%.1f%% predicted correctly.\n",percent);
  return correct;
}


void Network::save(const std::string& filename) const
{
  NetworkFileHeader fh;
  fh.magicNum = 0xAFFE;
  fh.numLayers=layerSizes.size();
  std::unique_ptr<FILE, FileDeleter> fNet(fopen(filename.c_str(),"wb"));
  size_t nElem=fwrite(&fh, sizeof(fh),1,fNet.get());
  if(nElem!=1) puts("Error saving network!");
  //Saving layer sizes
  for (int i = 0; i < layerSizes.size(); ++i)
  {
    size_t sz=layerSizes[i];
    nElem=fwrite(&sz, sizeof(sz),1,fNet.get());
    if(nElem!=1) puts("Error saving network!");
  }
  //Saving layers
  //Skipping first layer as it doesn't have weights or biases
  for (int i = 1; i < layerSizes.size(); ++i)
  {
    Layer* cLayer=layers[i];
    std::vector<float> weights=cLayer->getWeights().getMatrix();
    std::vector<float> biases =cLayer->getBiases().getMatrix();
    nElem = fwrite(weights.data(), sizeof(float),weights.size(),fNet.get());
    if(nElem!=weights.size()) puts("Error saving weights!");
    nElem = fwrite(biases.data(), sizeof(float),biases.size(),fNet.get());
    if(nElem!=biases.size()) puts("Error saving biases!");
  }
  printf("Network successfully saved to \"%s\"!\n", filename.c_str());
}

void Network::load(const std::string& filename)
{
  NetworkFileHeader fh;
  std::unique_ptr<FILE, FileDeleter> fNet(fopen(filename.c_str(),"rb"));
  size_t nElem=fread(&fh, sizeof(fh),1,fNet.get());
  if(nElem!=1)puts("Error reading network save file!");
  else
  {
    if(fh.magicNum!=0xAFFE) puts("The file does not apear to be a network save file!");
    else
    {
      layerSizes.clear();
      size_t data[fh.numLayers];
      nElem=fread(data, sizeof(data),1,fNet.get());
      if(nElem!=1)puts("Error reading layer sizes!");
      layerSizes.assign(data, data+fh.numLayers);
      layers.clear();

      for (int i = 0; i < layerSizes.size(); ++i)
      {
        if(i==0) 
        { 
          Matrix weightsM(layerSizes[i],1);
          Matrix biasesM(layerSizes[i],1);
          Layer* current = new Layer(weightsM, biasesM);
          layers.push_back(current);
        }
        else
        {
          //number of weights + number of biases
          size_t numParams= layerSizes[i]*layerSizes[i-1] + layerSizes[i];
          float paramData[numParams];
          nElem=fread(paramData, sizeof(paramData),1,fNet.get());
          if(nElem!=1) printf("Error reading layer %i data!\n",i);
          Matrix weightsM(layerSizes[i],layerSizes[i-1],paramData);
          Matrix biasesM(layerSizes[i],1,paramData + layerSizes[i]*layerSizes[i-1]);
          Layer* current = new Layer(weightsM, biasesM, (i==layerSizes.size() - 1));
          layers.push_back(current);
        }
        
      }
      puts("Network loaded successfully!");
    }

  }
}