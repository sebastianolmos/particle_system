#pragma once

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

class VoxelShape {
public:
	VoxelShape(string path);
	VoxelShape() {}

	void readFile(string path);

	int* getVoxelsArray()
	{
		return m_voxels;
	}

	int getSize()
	{
		return size;
	}

	int getWidthX()
	{
		return m_widthX;
	}

	int getWidthY()
	{
		return m_widthY;
	}

	int getWidthZ()
	{
		return m_widthZ;
	}

private:
	int* m_voxels;
	int size;
	int m_widthX;
	int m_widthY;
	int m_widthZ;
};
VoxelShape::VoxelShape(string path) {
	readFile(path);
}

void VoxelShape::readFile(string path) {
	int minx = 0;
	int maxx = 0;
	int miny = 0;
	int maxy = 0;
	int minz = 0;
	int maxz = 0;

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
			switch (c % 3)
			{
			case 0: // coord z
				if (m_voxels[c] > maxx)
				{
					maxx = m_voxels[c];
				}

				if (m_voxels[c] < minx)
				{
					minx = m_voxels[c];
				}
				break;
			case 1: //coord x
				if (m_voxels[c] > maxy)
				{
					maxy = m_voxels[c];
				}

				if (m_voxels[c] < miny)
				{
					miny = m_voxels[c];
				}
				break;
			case 2: //coord y
				if (m_voxels[c] > maxz)
				{
					maxz = m_voxels[c];
				}

				if (m_voxels[c] < minz)
				{
					minz = m_voxels[c];
				}
				break;
			}
			c++;
		}

	}

	m_widthX = maxx - minx;
	m_widthY = maxy - miny;
	m_widthZ = maxz - minz;
	// Close the file
	MyReadFile.close();
}
