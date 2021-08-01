#pragma once

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

class VoxelShape {
public:
	VoxelShape(string path);

	void readFile(string path);

	int* getVoxelsArray()
	{
		return m_voxels;
	}

	int getSize()
	{
		return size;
	}

private:
	int* m_voxels;
	int size;
};
VoxelShape::VoxelShape(string path) {
	readFile(path);
}

void VoxelShape::readFile(string path) {

	int number_of_lines = 0;
	string line;
	ifstream myfile(path);

	while (std::getline(myfile, line))
		++number_of_lines;
	myfile.close();

	size = number_of_lines;
	// Create a text string, which is used to output the text file
	string myText;
	m_voxels = new int[size*3];
	int c = 0;
	// Read from the text file
	ifstream MyReadFile(path);
	// Use a while loop together with the getline() function to read the file line by line
	while (getline(MyReadFile, myText)) {
		// Output the text from the file
		istringstream ss(myText);
		string token;

		while (getline(ss, token, ',')) {
			ss.ignore();
			m_voxels[c] = stoi(token);
			c++;
		}

	}
	// Close the file
	MyReadFile.close();
}
