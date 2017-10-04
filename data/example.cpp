#include <iostream>
#include <string>
#include <vector>

typedef std::vector<float> FLTARY;

template<typename _IS>
void LoadData(_IS &inStream, uint32_t *pImgRows, uint32_t *pImgCols,
	std::vector<FLTARY> &trainImages, FLTARY &trainLabels,
	std::vector<FLTARY> &testImages) {
	uint32_t nTrainCnt, nTestCnt;
	inStream >> nTrainCnt >> nTestCnt >> *pImgRows >> *pImgCols;
	uint32_t nImgArea = *pImgRows * *pImgCols, n = 41;
	trainImages.resize(nTrainCnt);
	trainLabels.resize(nTrainCnt);
	testImages.resize(nTestCnt);
	for (uint32_t i = 0; i < nTrainCnt + nTestCnt; ++i) {
		std::string strLine;
		inStream >> strLine;
		std::vector<float> fltBuf(nImgArea);
		for (uint32_t j = 0; j < nImgArea / 2; ++j)
		{
			const char *p = strLine.c_str() + j * 3;
			uint16_t rawCode = (uint16_t)(p[0] - '0') * n * n;
			rawCode += (uint16_t)(p[1] - '0') * n;
			rawCode += (uint16_t)(p[2] - '0');
			fltBuf[j * 2 + 0] = ((rawCode & 0xFF) - 128.0f) / 255.0f;
			fltBuf[j * 2 + 1] = ((rawCode >> 8) - 128.0f) / 255.0f;
		}
		if (i < nTrainCnt) {
			fltBuf.swap(trainImages[i]);
			inStream >> trainLabels[i];
		}
		else fltBuf.swap(testImages[i - nTrainCnt]);
	}
}


//

int main(int nArgCnt, char *ppArgs[])
{
    uint32_t nImgRows, nImgCols;
	std::vector<FLTARY> trainImgs, testImgs;
	FLTARY trainLabels;
	//if you want to load from a file, use std::ifstream to open your file
	//	and replace the std::cin with the file object.
	LoadData(std::cin, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
	std::cout << nImgCols << std::endl;
    
    //....
    //Do your homework here
}


