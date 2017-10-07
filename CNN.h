//
// Created by 峰池 on 2017/10/7.
//
#ifndef HANDBUILTNN_CNN_H
#define HANDBUILTNN_CNN_H

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


class ConvLayer{
public:
    FLTARY _weights;
    FLTARY _bias;
    FLTARY _inData;
    uint32_t _inSize, _outSize;
    uint32_t _inWidth, _outWidth;
    uint32_t _filterWidth;


    template<typename _Gen>
    uint32_t Initialize(uint32_t nInSize, uint32_t nOutSize, uint32_t nInWidth,
                        _Gen gen, uint32_t filterWidth = 3) {
        _inSize = nInSize;
        _outSize = nOutSize;
        _filterWidth = filterWidth;
        _weights.resize(_inSize * _filterWidth * _filterWidth * _outSize);
        _bias.resize(_outSize);
        _inWidth = nInWidth;
        _outWidth = _inWidth - _filterWidth + 1;

        std::generate(_weights.begin(), _weights.end(), gen);
        std::generate(_bias.begin(), _bias.end(), gen);

        return _outWidth;
    }


    void Update(const FLTARY &weightsGrads, const FLTARY &biasGrads, float lr) {
        // i : input size
        // m : filter width
        // n : filter height
        // j : output size
        uint32_t jCount = _filterWidth * _filterWidth * _outSize;
        uint32_t mCount = _filterWidth * _outSize;
        uint32_t nCount = _outSize;
        for (uint32_t i = 0; i < _inSize; ++i) {
            for (uint32_t m = 0; m < _filterWidth; ++m) {
                for (uint32_t n = 0; n < _filterWidth; ++n) {
                    for (uint32_t j = 0; j < _outSize; ++j) {
                        _weights[j * jCount + m * mCount + n * nCount + i] -= lr * weightsGrads[j * jCount + m * mCount + n * nCount + i];
                    }
                }
            }
        }

        for (uint32_t j = 0; j < _outSize; ++j) {
            _bias[j] -= lr * biasGrads[j];
        }
    }

    void Forward(const FLTARY &inData, FLTARY &outData) {
        _inData = inData;
        // j : output size
        outData.resize(_outSize * _outWidth * _outWidth);

        uint32_t jOutCount = _outWidth * _outWidth;
        uint32_t iInCount = _inWidth * _inWidth;

        uint32_t jWeightCount = _filterWidth * _filterWidth * _outSize;
        uint32_t mWeightCount = _filterWidth * _outSize;
        uint32_t nWeightCount = _outSize;


        // 使用非数组的方式太难写了。。。。
        for (uint32_t j = 0; j < _outSize; j++) {
            for (uint32_t p = 0; p < _outWidth; p++){
                for (uint32_t q = 0; q < _outWidth; q++){
                    // 三层数据迭代
                    for (uint32_t m = 0; m < _filterWidth; m++){
                        for (uint32_t n = 0; n < _filterWidth; n++){
                            for (uint32_t i = 0; i < _inSize; i++)
                            outData[j * jOutCount + p * _outWidth + q] += _weights[j * jWeightCount + m * mWeightCount + n * nWeightCount + i] \
                                    * _inData[i * iInCount + (p + m) * _inWidth  + (n + q)];

                        }
                    }
                }
            }
        }
    }

    void Backward(const FLTARY &topGrads, FLTARY &bottomGrads,
                  FLTARY &weightsGrads, FLTARY &biaGrads) {

    }
};


#endif //HANDBUILTNN_CNN_H
