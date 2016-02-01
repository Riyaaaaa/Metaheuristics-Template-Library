//
//  test_nn.h
//  MTL_Development
//
//  Created by Riya.Liel on 2015/07/14.
//  Copyright (c) 2015年 Riya.Liel. All rights reserved.
//

#ifndef MTL_Development_test_nn_h
#define MTL_Development_test_nn_h

#include"Boolean_Operation_Network.h"
#include"Boolean_Operation_Network_Dynamic.h"
#include"Gpu-Accelerated-Network.h"
#include"OCR_Network.h"

//void unit_test(){
//    mtl::NNSolver< mtl::FeedForward<1,1>, mtl::sigmoid_af > solver(0.05);
//    
//    std::vector< std::pair< std::array<double,1>, std::array<double,1> > > list;
//    list.push_back(std::make_pair( std::array<double,1>{1}, std::array<double,1>{0} ));
//    
//    solver.training<mtl::ErrorCorrection>(list);
//    
//    std::cout << "-------END--------" << std::endl;
//}

void test_nn(){
    //import_network_and_plot("xor_network_parameters.txt");

    //xor_nn_amp();
	//xor_nn_dy();
	//xor_nn_amp_fileio("../../NN/training_sample/xor_train.csv");
    //unit_test();
	//ocr_train_trimmer(101);
	//ocr_nn("../../NN/training_sample/ocr_train.csv");
	//ocr_nn("ocr_test_scale_101.csv","ocr_network_testcase01.txt");
	//ocr_test_trimmer(100);
	//ocr_nn("ocr_train_scale_101.csv");
	//ocr_calc_error("ocr_train_scale_101.csv", "ocr_network.txt");
	//ocr_tester("ocr_test_scale_100.csv","ocr_network_0114_1110.txt");

	/*std::vector<std::string> filenames;
	for (int i = 0; i < 10; i++) {
		filenames.push_back("../../NN/training_sample_image/ono2_" + std::to_string(i) + ".png");
	}

	auto samples = import_csv_from_image(filenames);

	export_csv("ono.csv",samples,std::ios::app);*/
	//ocr_tester("minst.csv", "ocr_network_0114_1110.txt");

	//auto samples = import_csv("minst.csv",784,10);
	//outputFeature("ocr_network_0114_1110.txt", samples[0].first);

	/*auto sample = import_csv_for_test("ocr_test.csv",784);

	std::vector< std::pair< std::vector<float>, std::vector<float> > > csv_data;
	const int rows = 28, cols = 28;
	cv::Mat charactor_img(rows, cols, CV_8UC1);
	cv::Mat view;

	for (int idx = 3; idx < sample.size(); idx+=500) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				charactor_img.at<unsigned char>(i, j) = (sample[idx][i*cols + j] + 1) * 128.f;
			}
		}
		cv::resize(charactor_img, view, cv::Size(rows * 5, cols * 5));
		cv::imshow("charactor", view);

		int num;
		std::cout << "enter answer" << std::endl;
		cv::waitKey(-1);
		cv::destroyWindow("charactor");

		std::cin >> num;

		std::vector<float> target(10);
		std::fill(target.begin(), target.end(), -1);
		target[num] = 1;

		csv_data.push_back(std::make_pair(sample[idx], target));
	}
	export_csv("minst.csv", csv_data, std::ios::app);*/

	//mtl::FeedForward_Convolution<5>::struct_t network_struct;

	//network_struct.resize(2);

	//network_struct[0].resize(1);
	//network_struct[1].resize(5);

	//for (auto&& map : network_struct[0]) {
	//	map = mtl::Size(28, 28);
	//}

	//for (auto&& map : network_struct[1]) {
	//	map = mtl::Size(24, 24);
	//}

	//mtl::NNSolver< mtl::FeedForward_Convolution<2>, mtl::tanh_af_gpu_accel > solver(network_struct);

	//std::vector< std::vector<float> > sensory(1 , std::vector<float>(28*28));
	//std::fill(sensory[0].begin(), sensory[0].end(), 0.1f);
	//auto& output = solver.solveAnswer(sensory);

	//for (auto&& map: output) {
	//	for (auto&& status: map.map) {
	//		std::cout << status << ",";
	//	}
	//	std::cout << std::endl;
	//}

	//std::cout << "-----------p0oling----------" << std::endl;

	//mtl::max_pooling< std::vector< typename mtl::FeedForward_Convolution<2>::Unit_t >, 2 > pooling_layer(solver.neural.network[1]);

	//for (int i = 0; i < solver.neural.network[1].size(); i++) {
	//	for (int j = 0; j < pooling_layer[i].size(); j++) {
	//		std::cout << pooling_layer[i][j] << ",";
	//	}
	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;
	return;
}

#endif
