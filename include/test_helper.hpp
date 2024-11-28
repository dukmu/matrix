#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "matrix.h"
#include "timeit.h"
#include <string>

template <typename T>
cv::Mat toCvMat(const fkZQ::Matrix<T> &mat)
{
    int type = CV_32F;
    if (std::is_same<T, uchar>::value)
    {
        type = CV_8U;
    }
    else if (std::is_same<T, ushort>::value)
    {
        type = CV_16U;
    }
    else if (std::is_same<T, int>::value)
    {
        type = CV_32S;
    }
    else if (std::is_same<T, float>::value)
    {
        type = CV_32F;
    }
    else if (std::is_same<T, double>::value)
    {
        type = CV_64F;
    }
    else
    {
        std::cerr << "Unsupported type" << std::endl;
        return cv::Mat();
    }
    cv::Mat ret(mat.rows, mat.cols, type);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            ret.at<T>(i, j) = mat.at(i, j);
        }
    }
    return ret;
}

template <typename T>
float compare(const cv::Mat &cvMat, const fkZQ::Matrix<T> &mat)
{
    cv::Mat cvMat2 = toCvMat(mat);
    cv::Mat diff;
    cv::absdiff(cvMat, cvMat2, diff);
    cv::Scalar s = cv::sum(diff);
    return s[0] / (cvMat.rows * cvMat.cols);
}

#define assert_eq(cvMat, mat)                                                                                \
    {                                                                                                        \
        float diff = compare(cvMat, mat);                                                                    \
        if (diff > 1e-3)                                                                                     \
        {                                                                                                    \
            std::cerr << "Assertion failed: " << #cvMat << " != " << #mat << " diff: " << diff << std::endl; \
            cv::Mat _m1##cvMat, _m2##mat;                                                                    \
            cv::normalize(cvMat, _m1##cvMat, 0, 255, cv::NORM_MINMAX, CV_8U);                               \
            cv::normalize(toCvMat(mat), _m2##mat, 0, 255, cv::NORM_MINMAX, CV_8U);                          \
            cv::imwrite(std::string(#cvMat) + ".png", _m1##cvMat);                                           \
            cv::imwrite(std::string(#mat) + ".png", _m2##mat);                                               \
        }                                                                                                    \
    }
