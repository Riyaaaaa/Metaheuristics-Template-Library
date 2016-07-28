
![MTL](https://github.com/Riyaaaaa/Metaheuristic-Template-Library/blob/master/MTL_LOGO.png)

# Metaheuristics-Template-Library
Metaheuristics Template Library / supported Genetic Algorithm, Simulated Annealing, and Neural Network. 

waring: It is under development.

##Overview
MTL is a header-only library for meta-heuristics programming.

When you wanted to use the meta-heuristics, or when you need abstract and highly reusable architecture than, this library would be useful always.

## Description
support the following algorithm

1.Genetic Algorithm

2.Simulated Annealing

3.Neural Network

This framework is a static implementation, and works faster than usual. Instead, the compilation will take some time.

Anyone who is able to use moddern C++, would can handle this easily.

## Requirement
C++14 Compiler

OpenCV 2.49 (TSP of GA/SA sample)

## Install
 Please place the file of purpose in the same directory, and compile.
 
 compile option: `-std=c++1y`
 
## Usage

### NeuralNetwork

 ```cpp
#include"NNSolver.hpp"
#include<fstream>
#include<vector>
#include<utility>

int main() {
/* Neural Network Learning Solver(NNSolver) require template arguments            */
/* First template arg: network structure                  */
/* Second template arg: activation function            */

/* sample: static network                              */
/* mtl::FeedForward construct multilayer perceptron network as statically       */
/* mtl::FeedForward require variadic template arguments */
/* variadic template arguments :  number of units on each layer. order: input-output-hide(multi) */
/* below sample construct 2 unit input layers / 1 unit output layers */
  mtl::NNSolver< mtl::FeedForward<2, 1>, mtl::tanh_af > solver(0.05);
 
 /* training samples */
 /* pair.first : input, pair.second : output */
  std::vector< std::pair< std::array<double,2>, std::array<double,1> > > list;
  list.push_back(std::make_pair( std::array<double,2>{1,1}, std::array<double,1>{1} ));
  list.push_back(std::make_pair( std::array<double,2>{1,-1}, std::array<double,1>{-1} ));
  list.push_back(std::make_pair( std::array<double,2>{-1,1}, std::array<double,1>{-1} ));
  list.push_back(std::make_pair( std::array<double,2>{-1,-1}, std::array<double,1>{-1} ));
  
  /* Exec learning */
  /* training function require learning algorithm */
  solver.training<mtl::Backpropagation>(list);
  
  /* if training succeeded, solveAnswer function returns correctly value...  */
  auto ans = solver.solveAnswer({0, 0});
  
  /* dynamic network version */
  std::vector<mtl::FeedForward_Dy::size_t> network_struct = {2, 4, 1}; // input: 2 hide: 4 output: 1
  mtl::NNSolver< mtl::FeedForward_Dy, mtl::tanh_af > solver(network_struct);
  ...
  solver.training<mtl::Backpropagation>(0.15, list);
  
  /* gpu-accelerated version */
 std::vector<mtl::FeedForward_Dy::size_t> network_struct = {2, 4, 1};
 mtl::FeedForward_Amp<4> network; //template argument require max unit size
	network.setStruct(network_struct);

	mtl::NNSolver< mtl::FeedForward_Amp_View<4>, mtl::tanh_af_gpu_accel > solver(network);

	std::vector< std::pair< std::vector<float>, std::vector<float> > > list;
	list.push_back(std::make_pair(std::vector<float>{1, 1}, std::vector<float>{-1}));
	...
	
	solver.training<mtl::Backpropagation_Gpu_Accel>(0.15, list); //gpe acceleration algorithm
  
  return 0;
}
 ```
 
## Licence
[MIT](https://github.com/Riyaaaaa/Metaheuristic-Template-Library/blob/master/LICENSE)  

## Link
### My thesis
[Development of OCR system by Gpu-Accelerated Deep-Learning](http://202.231.11.56/~ATSUMU/docs/study.pdf)  
Original(ja): GPUによって加速化されたディープラーニングによる光学文字認識システムの開発  

## Author
 りやさん(@Riyaaaa_a)
 
