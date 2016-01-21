//
//  OCR_Network.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/09/26.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_OCR_Network_h
#define MTL_Development_OCR_Network_h

#include"NNSolver.hpp"
#include"modules.h"
#include"../Configure/opencv_include.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<numeric>
#include<utility>

void ocr_nn_convolution(std::string filename) {
	auto trainig_sample = import_csv(filename, 784, 10);
	mtl::FeedForward_Convolution<5>::struct_t network_struct_cnn;
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	mtl::FeedForward_Amp<784> network;

	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	network.setStruct(network_struct);

	network_struct_cnn.resize(2);
	network_struct_cnn[0].resize(1);
	network_struct_cnn[1].resize(20);

	for (auto&& map : network_struct_cnn[0]) {
		map = mtl::Size(28, 28);
	}
	for (auto&& map : network_struct_cnn[1]) {
		map = mtl::Size(24, 24);
	}

	mtl::NNSolver< mtl::FeedForward_Convolution<2>, mtl::tanh_af_gpu_accel > solver_cnn(network_struct_cnn);
	mtl::NNSolver< mtl::FeedForward_Amp_View<784>, mtl::tanh_af_gpu_accel > solver_perceptron(network);


}

void ocr_nn(std::string filename){
    auto trainig_sample = import_csv(filename,784,10);
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	mtl::FeedForward_Amp<784> network;
	network.setStruct(network_struct);
    mtl::NNSolver< mtl::FeedForward_Amp_View<784>, mtl::tanh_af_gpu_accel > solver(network);
	//solver.setNetworkStruct(network_struct);
    solver.training<mtl::Backpropagation_Gpu_Accel>(0.0003,trainig_sample);
	network.exportNetwork("ocr_network.txt");
	//network.exportNetwork("ocr_network.txt");
    
    std::cout << "-------END--------" << std::endl;
}

void ocr_nn(std::string csv_filename,std::string network_filename) {
	auto trainig_sample = import_csv_from_density(csv_filename, 784, 10);
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	mtl::FeedForward_Amp<784> network;
	network.setStruct(network_struct);
	mtl::NNSolver< mtl::FeedForward_Amp_View<784>, mtl::tanh_af_gpu_accel > solver(network);
	network.importNetwork(network_filename);

	solver.training<mtl::Backpropagation_Gpu_Accel>(0.001,trainig_sample);
	network.exportNetwork("ocr_network.txt");

	std::cout << "-------END--------" << std::endl;
}

void ocr_train_trimmer(int scale) {
	auto training_sample = import_csv_from_density("../../NN/training_sample/ocr_train.csv",784,10);
	std::ofstream ofs("ocr_train_scale_" + std::to_string(scale) + ".csv");

	ofs << "label" << ",";
	for (int i = 0; i < training_sample[0].first.size(); i++) {
		ofs << "pixel" + std::to_string(i);
		if (i != training_sample[0].first.size() - 1)ofs << ",";
	}

	ofs << std::endl;

	for (int i = 0; i < training_sample.size(); i+=scale) {
		for (int j = 0; j < training_sample[i].second.size(); j++) {
			ofs << training_sample[i].second[j] << ",";
		}
		for (int j = 0; j < training_sample[i].first.size(); j++) {
			ofs << training_sample[i].first[j];
			if (j != training_sample[i].first.size() - 1)ofs << ",";
		}
		ofs << std::endl;
	}
}

void ocr_calc_error(std::string csv_filename,std::string network_filename) {
	auto trainig_sample = import_csv(csv_filename, 784, 10);
	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	mtl::FeedForward_Amp<784> network;
	network.setStruct(network_struct);

	mtl::NNSolver< mtl::FeedForward_Amp_View<784>, mtl::tanh_af_gpu_accel > solver(network);
	network.importNetwork(network_filename);

	float error = solver.calcError(trainig_sample);
	std::cout << "RMSerror = " << error << std::endl;

}

void ocr_test_trimmer(int scale) {
	auto training_sample = import_csv_for_test("../../NN/training_sample/ocr_test.csv", 784);
	std::ofstream ofs("ocr_test_scale_" + std::to_string(scale) + ".csv");

	for (int i = 0; i < training_sample[0].size(); i++) {
		ofs << "pixel" + std::to_string(i);
		if (i != training_sample[0].size() - 1)ofs << ",";
	}

	ofs << std::endl;

	for (int i = 0; i < training_sample.size(); i += scale) {
		for (int j = 0; j < training_sample[i].size(); j++) {
			ofs << training_sample[i][j];
			if (j != training_sample[i].size() - 1)ofs << ",";
		}
		ofs << std::endl;
	}
}

void ocr_tester(std::string csv_filename,std::string network_filename) {
	std::vector<int> problem_num{ 0,0,0,0,0,0,0,0,0,0 }, correct_ans{ 0,0,0,0,0,0,0,0,0,0 };
	std::vector< std::vector<int> > ans_mat(10);
	for (auto&& v: ans_mat) {
		v.resize(10);
		std::fill(v.begin(), v.end(), 0);
	}

	float percent;
	const int cols = 28, rows = 28;
	auto ocr_test = import_csv(csv_filename,784,10);

	cv::Mat charactor_img(rows, cols, CV_8UC1);
	cv::Mat view;

	std::vector<float> percentages(10);
	std::ofstream ofs("ocr_result_percentages_corrent.csv");

	std::vector<mtl::FeedForward_Dy::size_t> network_struct(3);
	network_struct[0] = 784;
	network_struct[1] = 784;
	network_struct[2] = 10;

	mtl::FeedForward_Amp<784> network;
	network.setStruct(network_struct);

	mtl::NNSolver< mtl::FeedForward_Amp_View<784>, mtl::tanh_af_gpu_accel > solver(network);
	network.importNetwork(network_filename);

	for (auto&& test : ocr_test) {
		/*for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				charactor_img.at<unsigned char>(i, j) = (test.first[i*cols + j] + 1) * 128;
			}
		}
		cv::resize(charactor_img, view, cv::Size(rows * 5, cols * 5));
		cv::imshow("charactor", view);*/

		float result = mtl::elite_principle<concurrency::array_view<mtl::Unit_Dy_Amp<784>>, mtl::tanh_af>(solver.solveAnswer(test.first)).idx;
		float ans = std::max_element(test.second.begin(), test.second.end()) - test.second.begin();
		problem_num[ans]++;
		std::cout << result;

		if (ans == result) {
			correct_ans[ans]++;
			std::cout << " correct!";
		}
		else std::cout << "wrong.";
		std::cout << " percent: " << correct_ans[ans] / (float)problem_num[ans] * 100.f
			<< "%" << std::endl;

		percentages[ans] = correct_ans[ans] / (float)problem_num[ans] * 100.f;
		ans_mat[ans][result]++;

		/*cv::waitKey(-1);
		cv::destroyWindow("charactor");*/
	}

	for (int i = 0; i < 10; i++) {
		ofs << i << "," << percentages[i] << "," << problem_num[i] << "," << correct_ans[i] << std::endl;
	}
	ofs << std::endl;

	for (int j = 0; j < 10; j++) ofs << "," << j;
	ofs << std::endl;
	for (int i = 0; i < 10; i++) {
		ofs << i;
		for (int j = 0; j < 10; j++) {
			ofs << "," << ans_mat[i][j];
		}
		ofs << std::endl;
	}

	ofs << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) ofs << i << "," << j << "," << ans_mat[i][j] / std::accumulate(ans_mat[i].begin(),ans_mat[i].end(),0.0f) << std::endl;
	}
}

#endif
