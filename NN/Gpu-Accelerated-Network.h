#pragma once

#include"NNSolver.hpp"
#include"modules.h"
#include<fstream>
#include<vector>
#include<utility>

/* Each module output the CSV file as result.*/

/* CSV format

x,y,z
INPUT1_1,INPUT1_2,OUTPUT1
INPUT2_1,INPUT2_2,OUTPUT2
...
INPUTn_1,INPUTn_2,OUTPUTn

*/

/* default input value step is 2 / 0.02 (100STEPS).*/

void xor_nn_amp() {
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 2;
	network_struct[1] = 4;
	network_struct[2] = 1;

	mtl::FeedForward_Amp<4> network;
	network.setStruct(network_struct);

	mtl::NNSolver< mtl::FeedForward_Amp_View<4>, mtl::tanh_af_gpu_accel > solver(network);

	std::vector< std::pair< std::vector<float>, std::vector<float> > > list;
	list.push_back(std::make_pair(std::vector<float>{1, 1}, std::vector<float>{-1}));
	list.push_back(std::make_pair(std::vector<float>{-1, 1}, std::vector<float>{1}));
	list.push_back(std::make_pair(std::vector<float>{1, -1}, std::vector<float>{1}));
	list.push_back(std::make_pair(std::vector<float>{-1, -1}, std::vector<float>{-1}));

	solver.training<mtl::Backpropagation_Gpu_Accel>(0.015,list);

	std::cout << "-------END--------" << std::endl;
	std::ofstream ofs("xor_result.csv");
	ofs << "x," << "y," << "z" << std::endl;
	for (float x = -1.0; x <= 1.0; x += 0.02) {
		for (float y = -1.0; y <= 1.0; y += 0.02) {
			auto output = solver.solveAnswer({ x,y });
			ofs << x << "," << y << "," << output[0].output(mtl::tanh_af::activate) << std::endl;
		}
	}
	std::cout << std::endl;

	if (!solver.exportNetwork("xor_network_parameters.txt"))std::cout << "faild export" << std::endl;
}

void xor_nn_amp_fileio(std::string csv_filename) {
	auto trainig_sample = import_csv(csv_filename, 2, 1);
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 2;
	network_struct[1] = 4;
	network_struct[2] = 1;

	mtl::FeedForward_Amp<4> network;
	network.setStruct(network_struct);
	mtl::NNSolver< mtl::FeedForward_Amp_View<4>, mtl::tanh_af_gpu_accel > solver(network);

	solver.training<mtl::Backpropagation_Gpu_Accel>(0.015,trainig_sample);
	solver.exportNetwork("xor_network.txt");

	std::cout << "-------END--------" << std::endl;
}