#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include "matrix.h"
#include "timeit.h"
#include <string>

#define IS_EQUAL(lhs, rhs) (std::abs((lhs) - (rhs)) < 1e-3)

template <typename T>
void assert_eq(const fkZQ::Matrix<T> &lhs, const fkZQ::Matrix<T> &rhs)
{
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols)
    {
        std::string msg = "Matrix size not equal: ";
        msg += std::to_string(lhs.rows) + "x" + std::to_string(lhs.cols) + " vs ";
        msg += std::to_string(rhs.rows) + "x" + std::to_string(rhs.cols);
        std::cerr << msg << std::endl;
        return;
    }

    for (int i = 0; i < lhs.rows; ++i)
    {
        for (int j = 0; j < lhs.cols; ++j)
        {
            if (!IS_EQUAL(lhs.at(i, j), rhs.at(i, j)))
            {
                std::string msg = "Matrix element not equal at (" + std::to_string(i) + ", " + std::to_string(j) + "): ";
                msg += std::to_string(lhs.at(i, j)) + " vs " + std::to_string(rhs.at(i, j));
                std::cerr << msg << std::endl;
                return;
            }
        }
    }
    std::cout << "Matrix equal" << std::endl;
}

template <typename T>
void assert_eq(const cv::Mat &lhs, const fkZQ::Matrix<T> &rhs)
{
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols)
    {
        std::string msg = "Matrix size not equal: ";
        msg += std::to_string(lhs.rows) + "x" + std::to_string(lhs.cols) + " vs ";
        msg += std::to_string(rhs.rows) + "x" + std::to_string(rhs.cols);
        std::cerr << msg << std::endl;
        return;
    }

    for (int i = 0; i < lhs.rows; ++i)
    {
        for (int j = 0; j < lhs.cols; ++j)
        {
            if (!IS_EQUAL(lhs.at<T>(i, j), rhs.at(i, j)))
            {
                std::string msg = "Matrix element not equal at (" + std::to_string(i) + ", " + std::to_string(j) + "): ";
                msg += std::to_string(lhs.at<T>(i, j)) + " vs " + std::to_string(rhs.at(i, j));
                std::cerr << msg << std::endl;
                return;
            }
        }
    }
    std::cout << "Matrix equal" << std::endl;
}

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