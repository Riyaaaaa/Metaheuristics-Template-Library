#pragma once

#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<utility>

template<std::size_t InputVectorSize, std::size_t OutputVectorSize>
std::vector< std::pair< std::array<double, InputVectorSize>, std::array<double, OutputVectorSize> > > import_csv(std::string filename) {
	std::fstream filestream(filename);
	std::vector< std::pair< std::array<double, InputVectorSize>, std::array<double, OutputVectorSize> > > list;
	std::vector<std::vector<std::string>> table;
	const char delimiter = ',';
	std::string str;

	if (!filestream.is_open())
	{
		return list;
	}

	filestream >> str;

	while (!filestream.eof())
	{
		std::string buffer;
		filestream >> buffer;

		std::istringstream streambuffer(buffer);
		std::string token;
		std::array<double, InputVectorSize> input;
		std::array<double, OutputVectorSize> output;

		if (buffer.empty())break;

		std::fill(output.begin(), output.end(), 0);

		getline(streambuffer, token, delimiter);
		output[std::stoi(token)] = 1;

		for (int i = 0; i<InputVectorSize; i++) {
			getline(streambuffer, token, delimiter);
			input[i] = std::stoi(token);
		}
		list.push_back(std::make_pair(input, output));
	}

	for (int row = 0; row < list.size(); row++)
	{
		for (int column = 0; column < list[row].first.size(); column++)
		{
			if (column < list[row].first.size() - 1)
			{
			}
		}
		for (int column = 0; column < list[row].second.size(); column++)
		{
			if (column < list[row].second.size() - 1)
			{
			}
		}
	}

	return list;
}

std::vector< std::pair< std::vector<float>, std::vector<float> > > import_csv_from_density(std::string filename, const std::size_t InputVectorSize, const std::size_t OutputVectorSize);
std::vector< std::pair< std::vector<float>, std::vector<float> > > import_csv(std::string filename, const std::size_t InputVectorSize, const std::size_t OutputVectorSize);
std::vector< std::vector<float> > import_csv_for_test(std::string filename, const std::size_t InputVectorSize);
std::vector< std::pair< std::vector<float>, std::vector<float> > > import_csv_from_image(std::vector<std::string>);
bool export_csv(std::string,std::vector< std::pair< std::vector<float>, std::vector<float> > >);