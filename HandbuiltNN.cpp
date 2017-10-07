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

const int MAXEXP = 80;


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
        for (uint32_t j = 0; j < nImgArea / 2; ++j) {
            const char *p = strLine.c_str() + j * 3;
            uint16_t rawCode = (uint16_t) (p[0] - '0') * n * n;
            rawCode += (uint16_t) (p[1] - '0') * n;
            rawCode += (uint16_t) (p[2] - '0');
            fltBuf[j * 2 + 0] = ((rawCode & 0xFF) - 128.0f) / 255.0f;
            fltBuf[j * 2 + 1] = ((rawCode >> 8) - 128.0f) / 255.0f;
        }
        if (i < nTrainCnt) {
            fltBuf.swap(trainImages[i]);
            inStream >> trainLabels[i];
        } else fltBuf.swap(testImages[i - nTrainCnt]);
    }
}


class FCLayer {
public:
    FLTARY _weights;
    FLTARY _bias;
    FLTARY _inData;

    template<typename _Gen>
    void Initialize(uint32_t nInSize, uint32_t nOutSize, _Gen gen) {
        _weights.resize(nInSize * nOutSize);
        _bias.resize(nOutSize);

        std::generate(_weights.begin(), _weights.end(), gen);
        std::generate(_bias.begin(), _bias.end(), gen);
    }

    // update in SGD
    void Update(const FLTARY &weightsGrads, const FLTARY &biasGrads, float lr) {
        uint32_t nInSize = _weights.size() / _bias.size();
        for (uint32_t j = 0; j < _bias.size(); ++j) {
            for (uint32_t i = 0; i < nInSize; ++i) {
                _weights[j * nInSize + i] -= lr * weightsGrads[j * nInSize + i];
            }
            _bias[j] -= lr * biasGrads[j];
        }
    }

    // Update in SGD with momentum
    void Update(const FLTARY &weightsGrads, const FLTARY &biasGrads,
                FLTARY &weightsMomentum, FLTARY &biasMomentum, float lr, float momentum) {
        uint32_t nInSize = _weights.size() / _bias.size();

        for (uint32_t j = 0; j < _bias.size(); ++j) {
            for (uint32_t i = 0; i < nInSize; ++i) {
                weightsMomentum[j * nInSize + i] = momentum * weightsMomentum[j * nInSize + i] + lr * weightsGrads[j * nInSize + i];
                _weights[j * nInSize + i] -= weightsMomentum[j * nInSize + i];
            }
//            biasMomentum[j] = momentum * biasMomentum[j] + lr * biasGrads[j];
//            _bias[j] -= biasMomentum[j];
            _bias[j] -= lr * biasGrads[j];
        }
    }

    void Forward(const FLTARY &inData, FLTARY &outData) {
        uint32_t nInSize = inData.size();
        _inData = inData;
        outData = _bias;

        for (uint32_t j = 0; j < _bias.size(); ++j) {
            for (uint32_t i = 0; i < nInSize; ++i) {
                outData[j] += _weights[j * nInSize + i] * inData[i];
            }
        }
    }

    void Backward(const FLTARY &topGrads, FLTARY &bottomGrads,
                  FLTARY &weightsGrads, FLTARY &biaGrads) {
        uint32_t nInSize = _inData.size();
        bottomGrads.clear();
        bottomGrads.resize(nInSize);
        for (uint32_t i = 0; i < nInSize; ++i) {
            for (uint32_t j = 0; j < _bias.size(); ++j) {
                bottomGrads[i] += topGrads[j] * _weights[j * nInSize + i];
                weightsGrads[j * nInSize + i] += topGrads[j] * _inData[i];
            }
        }

        for (uint32_t j = 0; j < _bias.size(); ++j) {
            biaGrads[j] += topGrads[j];
        }
    }
};

class SoftmaxLoss {
public:
    FLTARY _grads;

    uint32_t Forward(const FLTARY &inData, float fLabel, float *pLoss) {
        auto fMax = *std::max_element(inData.begin(), inData.end());
        FLTARY exps(inData.size());
        for (uint32_t i = 0; i < inData.size(); i++) {
            exps[i] = std::exp(inData[i] - fMax);
        }

        auto fSum = std::accumulate(exps.begin(), exps.end(), 0.0f);
        for (auto &v : exps) {
            v /= fSum;
        }

        uint32_t nLabel = (uint32_t) (fLabel + 0.5f);
        *pLoss = -std::log(exps[nLabel]);

        _grads.resize(inData.size(), 0);
        for (uint32_t i = 0; i < inData.size(); ++i) {
            _grads[i] = exps[i] - (i == nLabel);
        }

        return (std::max_element(exps.begin(), exps.end()) - exps.begin());
    }

    void Backward(FLTARY &bottomGrads) {
        bottomGrads = _grads;
    }
};

class ActiveLayer {
public:
    FLTARY _grads;

    virtual void Forward(const FLTARY &inData, FLTARY &outData) = 0;

    void Backward(const FLTARY &topGrads, FLTARY &bottomGrads) {
        bottomGrads.resize(topGrads.size());
        for (uint32_t i = 0; i < topGrads.size(); i++) {
            bottomGrads[i] = topGrads[i] * _grads[i];
        }
    }
};

class Sigmoid : public ActiveLayer {
public:
    void Forward(const FLTARY &inData, FLTARY &outData) {
        outData.resize(inData.size());
        _grads.resize(inData.size());
        for (uint32_t i = 0; i < inData.size(); i++) {
            if (inData[i] < -MAXEXP) {
                outData[i] = 0.0f;
            } else {
                auto fExp = std::exp(-inData[i]);
                auto fTmp = 1.0 + fExp;
                outData[i] = 1.0f / fTmp;
                _grads[i] = fExp / (fTmp * fTmp);
            }
        }
    }
};

class Tanh : public ActiveLayer {
public:
    void Forward(const FLTARY &inData, FLTARY &outData)
    {
        outData.resize(inData.size(), 0);
        _grads.resize(inData.size(), 0);
        for (uint32_t i = 0; i < inData.size(); i++) {
            if (inData[i] < -(MAXEXP / 2.0f)) {
                outData[i] = 0.0f;
            } else {
                auto fExp = std::exp(-2.0f * inData[i]);
                auto fTmp = 1.0f + fExp;
                outData[i] = 2.0f / fTmp - 1;
                _grads[i] = 4.0f * fExp / (fTmp * fTmp);
            }
        }
    }
};

class ReLU : public ActiveLayer{
public:
    void Forward(const FLTARY &inData, FLTARY &outData)
    {
        outData.resize(inData.size());
        _grads.resize(inData.size());
        for (uint32_t i = 0; i < inData.size(); i++) {
            if (inData[i] <= 0) {
                outData[i] = 0.0f;
            } else {
                outData[i] = inData[i];
                _grads[i] = 1.0f;
            }
        }
    }
};

int main(int nArgCnt, char *ppArgs[]) {
    clock_t start = clock();

    uint32_t nImgRows, nImgCols;

    std::vector<FLTARY> trainImgs, testImgs;
    FLTARY trainLabels;
    //if you want to load from a file, use std::ifstream to open your file
    //	and replace the std::cin with the file object.

//    std::ifstream trainFileStram("/Users/fengchi/Github/HandbuiltNN/data/train_2000a.txt");

//    LoadData(trainFileStram, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
//    LoadData(std::cin, &nImgRows, &nImgCols, trainImgs, trainLabels, testImgs);
    uint32_t nImgArea = nImgCols * nImgRows;


    FCLayer fc1, fc2;
    Tanh act;
    SoftmaxLoss loss;

    std::random_device rd; // random seed generator
    std::mt19937 rg(rd()); // random generator
    std::normal_distribution<> normDist(0, 0.1); // normal distribution
    auto genNormRand = [&]() { return normDist(rg); };

    FLTARY fc1Res, actRes, fc2Res;
    FLTARY lossGrads, fc2Grads, actGrads, fc1Grads;


    uint32_t fc1In = 28;
    uint32_t fc2In = 10;
    uint32_t maxIter = 4;
    float lr = 0.03;
    float momentum = 0.9;
    uint32_t nBatchSize = 64;
    bool timeFlag = true;

    fc1.Initialize(nImgArea, fc1In, genNormRand);
    fc2.Initialize(fc1In, fc2In, genNormRand);

    FLTARY fc1WeightsGrads(fc1._weights.size()), fc1BiasGrads(fc1._bias.size());
    FLTARY fc2WeightsGrads(fc2._weights.size()), fc2BiasGrads(fc2._bias.size());
    FLTARY fc1WeightsMomentum(fc1._weights.size(), 0), fc1BiasMomentum(fc1._bias.size(), 0);
    FLTARY fc2WeightsMomentum(fc2._weights.size(), 0), fc2BiasMomentum(fc2._bias.size(), 0);

    fc1Grads.resize(fc1In, 0);
    fc2Grads.resize(fc2In, 0);
    actGrads.resize(fc2In, 0);
    uint32_t iter;

    for (iter = 0; iter < maxIter && timeFlag; ++iter) {
        for (uint32_t iImgdx = 0; iImgdx < trainImgs.size() && timeFlag; iImgdx += nBatchSize) {

            std::fill(fc1WeightsGrads.begin(), fc1WeightsGrads.end(), 0.0f);
            std::fill(fc2WeightsGrads.begin(), fc2WeightsGrads.end(), 0.0f);
            std::fill(fc1BiasGrads.begin(), fc1BiasGrads.end(), 0.0f);
            std::fill(fc1BiasGrads.begin(), fc1BiasGrads.end(), 0.0f);

            float fLossSum = 0.0f, fLoss;
            uint32_t nCorrected = 0;
            for (uint32_t iBatch = 0; iBatch < nBatchSize && timeFlag; ++iBatch) {

                // Random SGD
                uint32_t iImg = rand() % trainImgs.size();
                // forward
                fc1.Forward(trainImgs[iImg], fc1Res);
                act.Forward(fc1Res, actRes);
                fc2.Forward(actRes, fc2Res);
                uint32_t nPred = loss.Forward(fc2Res, trainLabels[iImg], &fLoss);

                // backward
                loss.Backward(lossGrads);
                fc2.Backward(lossGrads, fc2Grads, fc2WeightsGrads, fc2BiasGrads);
                act.Backward(fc2Grads, actGrads);
                fc1.Backward(actGrads, fc1Grads, fc1WeightsGrads, fc1BiasGrads);

                fLossSum += fLoss;
                nCorrected += (nPred == trainLabels[iImg]);
//                iImg++;
//                if (iImg >= trainImgs.size()) break;
            }

            std::cout << "loss = " << fLossSum / (float)nBatchSize << "\tprecision = " << nCorrected / (float) nBatchSize << std::endl;

//            fc1.Update(fc1WeightsGrads, fc1BiasGrads, lr / nBatchSize);
//            fc2.Update(fc2WeightsGrads, fc2BiasGrads, lr / nBatchSize);
            fc1.Update(fc1WeightsGrads, fc1BiasGrads, fc1WeightsMomentum, fc1BiasMomentum, lr / nBatchSize, momentum);
            fc2.Update(fc2WeightsGrads, fc2BiasGrads, fc2WeightsMomentum, fc2BiasMomentum, lr / nBatchSize, momentum);
            if ((clock() - start) / (double) CLOCKS_PER_SEC > 3.91) {
                timeFlag = false;
            }
        }
    }

    // Test
    std::ifstream testLabelFileStream("/Users/fengchi/Github/HandbuiltNN/data/label_2000a.txt");
    FLTARY testLabel;
    testLabel.resize(500);
    for (uint32_t i = 0; i < 500; i++) {
        testLabelFileStream >> testLabel[i];
    }

    float fLossSum = 0.0f, fLoss;
    uint32_t nCorrected = 0;
    for (uint32_t i = 0; i < testImgs.size(); i++) {
        fc1.Forward(testImgs[i], fc1Res);
        act.Forward(fc1Res, actRes);
        fc2.Forward(actRes, fc2Res);

        uint32_t nPred = loss.Forward(fc2Res, testLabel[i], &fLoss);
        fLossSum += fLoss;
        uint32_t label = (uint32_t)(testLabel[i] + 0.5f);
        nCorrected += (nPred == label);
//        std::cout << nPred << " " << label << std::endl;
    }
    std::cout << "[test] " << "loss = " << fLossSum / testImgs.size() << " accuracy = "
         << (float) nCorrected / testImgs.size() << std::endl;

    std::cout << "Duration: " << (clock() - start) / (double) CLOCKS_PER_SEC << "s " << "for " << iter
         << " times durations" << std::endl;

    // submit
//    float fLoss;
//    for(uint32_t i = 0; i < testImgs.size(); i++) {
//        fc1.Forward(testImgs[i], fc1Res);
//        act.Forward(fc1Res, actRes);
//        fc2.Forward(actRes, fc2Res);
//
//        uint32_t nPred = loss.Forward(fc2Res, 0, &fLoss);
//        std::cout << nPred << std::endl;
//    }
    return 0;
}


