#include "modules.h"

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