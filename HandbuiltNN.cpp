#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <functional>

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

class FCLayer
{
public:
    FLTARY _weights;
    FLTARY _bias;
    FLTARY _inData;

    template<class _Gen>
    void Initialize(uint32_t nInSize, uint32_t nOutSize, _Gen gen)
    {
        _weights.resize(nInSize * nOutSize);
        _bias.resize(nOutSize);

        std::generate(_weights.begin(), _weights.end(), gen);
        std::generate(_bias.begin(), _bias.end(), gen);
    }

    void Update(const FLTARY &weightsGrad, const FLTARY &biasGrad, float lr)
    {
        uint32_t nInSize = _weights.size() / _bias.size();
        for (uint32_t j = 0; j < _bias.size(); ++j)
        {
            for (uint32_t i = 0; i < _weights.size(); ++i)
            {
                _weights[j * nInSize + i] = lr * weightsGrad[j * nInSize + i];
            }
            _bias[j] = lr * biasGrad[j];
        }
    }

    void Forward(const FLTARY &inData, FLTARY &outData)
    {
        uint32_t nInSize = inData.size();
        _inData = inData;
        outData = _bias;

        for (uint32_t j = 0; j < _bias.size(); ++j)
        {
            for(uint32_t i = 0; i < nInSize; ++i)
            {
                outData[j] += _weights[j * nInSize + i] * inData[i];
            }
        }
    }

    void Backward(const FLTARY &topGrads, FLTARY &bottomGrads,
        FLTARY &weightsGrads, FLTARY &biaGrads)
    {
        uint32_t nInSize = _inData.size();

        for (uint32_t j = 0; j < _bias.size(); ++j)
        {
            for (uint32_t i = 0; i < nInSize; ++i)
            {
                bottomGrads[i] += topGrads[j] * _weights[j * nInSize + i];
                weightsGrads[j * nInSize + i] += topGrads[j] * _inData[i];
            }
        }

        for (uint32_t j = 0; j < _bias.size(); ++j)
        {
            biaGrads[j] += topGrads[j];
        }
    }
};

class SoftmaxLoss
{
public:
    FLTARY _grads;

    uint32_t Forward(const FLTARY &inData, float fLabel, float* pLoss)
    {
        auto fMax = *std::max_element(inData.begin(), inData.end());
        FLTARY exps;
        for (auto v : inData)
        {
            exps.push_back(std::exp(v - fMax));
        }
        auto fSum = std::accumulate(exps.begin(), exps.end(), 0.0f);
        for (auto &v : exps)
        {
            v /= fSum;
        }

        uint32_t nLabel = (uint32_t)(fLabel + 0.5f);
        *pLoss = -std::log(exps[nLabel]);

        _grads.resize(inData.size());
        for (uint32_t i = 0; i < inData.size(); ++i)
        {
            _grads[i] = exps[i] - (i == nLabel);
        }

        return std::max_element(exps.begin(), exps.end()) - exps.begin();
    }

    void Backward(FLTARY &bottomGrads)
    {
        bottomGrads = _grads;
    }
};

class Sigmoid
{
public:
    FLTARY _grads;

    void Forward(const FLTARY &inData, FLTARY &outData)
    {
        outData.resize(inData.size());
        _grads.resize(inData.size());
        for (uint32_t i = 0; i < inData.size(); i++)
        {
            if (inData[i] < -80)
            {
                outData[i] = 0.0f;
            }
            else
            {
                auto fExp = std::exp(-inData[i]);
                auto fTmp = 1.0 + fExp;
                outData[i] = 1.0f / fTmp;
                _grads[i] = fExp / (fTmp * fTmp);
            }
        }
    }

    void Backward(const FLTARY &topGrads, FLTARY &bottomGrads)
    {
        bottomGrads.resize(topGrads.size());
        for(uint32_t i = 0; i < topGrads.size(); i++)
        {
            bottomGrads[i] = _grads[i];
        }
    }
};


//

int main(int nArgCnt, char *ppArgs[])
{
    uint32_t nImgRows, nImgCols;
    uint32_t nImgArea = nImgCols * nImgRows;
    std::vector<FLTARY> trainImgs, testImgs;
    FLTARY trainLabels;
    //if you want to load from a file, use std::ifstream to open your file
    //	and replace the std::cin with the file object.

    std::ifstream train_file("/Users/fengchi/Github/HandbuiltNN/data/train_2000a.txt");
//    char test[100];

//    if(train_file.is_open())
//    {
//        std::cout<<"Open Success!";
//    }
//    else
//    {
//        std::cout<<"Open Fail!";
//    }

    LoadData(train_file, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
//    LoadData(std::cin, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
    std::cout << nImgCols << " " << nImgRows << std::endl;

    //....
    //Do your homework here

    FCLayer fc1, fc2;
    Sigmoid act;
    SoftmaxLoss loss;

    std::random_device rd; // random seed generator
    std::mt19937 rg(rd()); // random generator
    std::normal_distribution<> normDist(0, 0.1); // normal distribution
    std::cout << normDist(rg) << std::endl;
    auto genNormRand = [&](){return normDist(rg);};

    FLTARY fc1Res, actRes, fc2Res;
    FLTARY lossGrads, fc2Grads, actGrads, fc1Grads;


    uint32_t fc1In = 16;
    uint32_t fc2In = 10;
    float lr = 0.1;

    fc1.Initialize(nImgArea, fc1In, genNormRand);
    fc2.Initialize(fc1In, fc2In, genNormRand);

    uint32_t nBatchSize = 64;
    FLTARY fc1Weights(fc1._weights.size()), fc1Bias(fc1._bias.size());
    FLTARY fc2Weights(fc2._weights.size()), fc2Bias(fc2._bias.size());

    for (uint32_t iter = 0; iter < 10; ++iter)
    {
        std::cout << "here" << std::endl;
        for (uint32_t iImg = 0; iImg + nBatchSize < trainImgs.size(); iImg += nBatchSize)
        {
            float fLossSum = 0.0f, fLoss;
            uint32_t nCorrected = 0;
            for (uint32_t iBatch = 0; iBatch < nBatchSize; ++iBatch)
            {
                iImg = (iImg + 1) % trainImgs.size();
                fc1.Forward(trainImgs[iImg], fc1Res);
                act.Forward(fc1Res, actRes);
                fc2.Forward(actRes, fc2Res);

                uint32_t nPred = loss.Forward(fc2Res, trainLabels[iImg], &fLoss);
                loss.Backward(lossGrads);
                fc2.Backward(lossGrads, fc2Grads, fc2Weights, fc2Bias);
                act.Backward(fc2Grads, actGrads);
                fc1.Backward(actGrads, fc1Grads, fc1Weights, fc1Bias);

                fLossSum += fLoss;
                nCorrected += (nPred == trainLabels[iImg]);

            }
            std::cout << "loss = " << fLossSum / nBatchSize \
                      << "\tprecision = " << nCorrected / nBatchSize;

            fc1.Update(fc1Grads, fc1Bias, lr);
            fc2.Update(fc2Grads, fc2Bias, lr);
        }
    }


    return 0;
}


