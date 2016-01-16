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

    xor_nn_amp();
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
		filenames.push_back("../../NN/training_sample_image/micha_" + std::to_string(i) + ".png");
	}

	auto samples = import_csv_from_image(filenames);

	export_csv("micachan.csv",samples);
	ocr_tester("micachan.csv", "ocr_network_0114_1110.txt");*/
}

#endif
