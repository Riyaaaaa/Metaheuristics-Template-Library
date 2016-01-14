#include "modules.h"
#include "../Configure/opencv_include.h"

std::vector< std::pair< std::vector<float>, std::vector<float> > > import_csv_from_density(std::string filename, const std::size_t InputVectorSize, const std::size_t OutputVectorSize) {
	std::ifstream filestream(filename);
	std::vector< std::pair< std::vector<float>, std::vector<float> > > list;
	std::vector<std::vector<std::string>> table;
	const char delimiter = ',';
	std::string str;

	if (!filestream.is_open())
	{
		return list;
	}

	getline(filestream, str);

	while (!filestream.eof())
	{
		std::string buffer;
		getline(filestream, buffer);

		std::istringstream streambuffer(buffer);
		std::string token;
		std::vector<float> input(InputVectorSize);
		std::vector<float> output(OutputVectorSize);

		if (buffer.empty())break;

		std::fill(output.begin(), output.end(), -1.f);

		getline(streambuffer, token, delimiter);
		output[std::stoi(token)] = 1;

		for (int i = 0; getline(streambuffer, token, delimiter); i++) {

			input[i] = std::stoi(token) / 128.f - 1;
		}
		list.push_back(std::make_pair(input, output));
	}

	return list;
}

std::vector< std::pair< std::vector<float>, std::vector<float> > > import_csv(std::string filename, const std::size_t InputVectorSize, const std::size_t OutputVectorSize) {
	std::ifstream filestream(filename);
	std::vector< std::pair< std::vector<float>, std::vector<float> > > list;
	std::vector<std::vector<std::string>> table;
	const char delimiter = ',';
	std::string str;

	if (!filestream.is_open())
	{
		return list;
	}

	getline(filestream, str);

	while (!filestream.eof())
	{
		std::string buffer;
		getline(filestream, buffer);

		std::istringstream streambuffer(buffer);
		std::string token;
		std::vector<float> input(InputVectorSize);
		std::vector<float> output(OutputVectorSize);

		if (buffer.empty())break;

		std::fill(output.begin(), output.end(), -1.f);

		for (int i = 0; i < OutputVectorSize; i++) {
			getline(streambuffer, token, delimiter);
			output[i] = std::stoi(token);
		}

		for (int i = 0; i < InputVectorSize; i++) {
			getline(streambuffer, token, delimiter);
			input[i] = std::stoi(token);
		}
		list.push_back(std::make_pair(input, output));
	}

	return list;
}

std::vector< std::vector<float> > import_csv_for_test(std::string filename, const std::size_t InputVectorSize) {
	std::ifstream filestream(filename);
	std::vector< std::vector<float> > list;
	std::vector<std::vector<std::string>> table;
	const char delimiter = ',';
	std::string str;

	if (!filestream.is_open())
	{
		return list;
	}

	getline(filestream, str);

	while (!filestream.eof())
	{
		std::string buffer;
		getline(filestream, buffer);

		std::istringstream streambuffer(buffer);
		std::string token;
		std::vector<float> input(InputVectorSize);

		if (buffer.empty())break;

		for (int i = 0; getline(streambuffer, token, delimiter); i++) {

			input[i] = std::stof(token);
		}

		list.push_back(input);
	}

	return list;
}

std::vector< std::pair< std::vector<float>, std::vector<float> > > import_csv_from_image(std::vector<std::string> filenames) {
	std::vector<std::pair<std::vector<float>,std::vector<float>>> data;
	cv::Mat all_image(28 * (filenames.size() / 11 + 1), 28 * 10, CV_8UC1),view,charactor_img(28,28,CV_8UC1);

	for (int idx = 0; idx < filenames.size(); idx++) {
		cv::Mat chara = cv::imread(filenames[idx], cv::IMREAD_GRAYSCALE);
		chara.convertTo(chara, CV_8UC1);
		cv::resize(chara, view, cv::Size(chara.rows * 5, chara.cols * 5));
		int num;
		cv::imshow("charactor",view);
		std::cout << "enter answer " + filenames[idx] << std::endl;
		//cv::waitKey(-1);
		std::cin >> num;

		std::vector<float> input, target(10);
		std::fill(target.begin(), target.end(), -1);
		target[num] = 1;
		for (int i = 0; i < chara.rows; i++) {
			for (int j = 0; j < chara.cols; j++) {
				input.push_back(chara.at<unsigned char>(i,j) / 128.f - 1);
			}
		}

		data.push_back( std::make_pair(input,target) );
		chara.copyTo(all_image(cv::Rect(idx % 10 * 28, idx / 10 * 28 , 28, 28)));
	}

	cv::imshow("window", all_image);
	cv::waitKey(-1);

	return data;
}

bool export_csv(std::string csv_filename,std::vector< std::pair< std::vector<float>, std::vector<float> > > training_sample) {
	std::ofstream ofs(csv_filename);

	ofs << "label" << ",";
	for (int i = 0; i < training_sample[0].first.size(); i++) {
		ofs << "pixel" + std::to_string(i);
		if (i != training_sample[0].first.size() - 1)ofs << ",";
	}

	ofs << std::endl;

	for (int i = 0; i < training_sample.size(); i++) {
		for (int j = 0; j < training_sample[i].second.size(); j++) {
			ofs << training_sample[i].second[j] << ",";
		}
		for (int j = 0; j < training_sample[i].first.size(); j++) {
			ofs << training_sample[i].first[j];
			if (j != training_sample[i].first.size() - 1)ofs << ",";
		}
		ofs << std::endl;
	}

	return true;
}