

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "matrix.h"
#include "boxfilter.hpp"

#include "timeit.h"
#include "test_helper.hpp"
#include <vector>
#define ROWS 1024
#define COLS 1280

template <size_t _rows, size_t _cols, typename T>
fkZQ::Matrix<T> FixedMatrix()
{
    return fkZQ::Matrix<T>(_rows, _cols);
}

int main()
{
    cv::Mat cvmatab(ROWS, COLS, CV_32F);
    cv::Mat cvmataa(ROWS, ROWS, CV_32F);
    cv::Mat cvmatba(COLS, ROWS, CV_32F);
    fkZQ::Matrix<float> pmatab(ROWS, COLS);
    fkZQ::Matrix<float> pmataa(ROWS, ROWS);
    fkZQ::Matrix<float> pmatba(COLS, ROWS);
    cv::randu(cvmatab, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(cvmataa, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(cvmatba, cv::Scalar::all(0), cv::Scalar::all(255));

    for (int i = 0; i < ROWS; ++i)
    {
        for (int j = 0; j < COLS; ++j)
        {
            pmatab.at(i, j) = cvmatab.at<float>(i, j);
            pmatba.at(j, i) = cvmatba.at<float>(j, i);
        }
    }
    for (int i = 0; i < ROWS; ++i)
    {
        for (int j = 0; j < ROWS; ++j)
        {
            pmataa.at(i, j) = cvmataa.at<float>(i, j);
        }
    }

    assert_eq(cvmatab, pmatab);
    assert_eq(cvmataa, pmataa);
    assert_eq(cvmatba, pmatba);

    TIMEIT_BEGIN(cv_add);
    cv::Mat cvadd = cvmatab + cvmatab;
    TIMEIT_END(cv_add);
    TIMEIT_PRINT(cv_add, 0, 0);

    TIMEIT_BEGIN(fkZQ_add);
    fkZQ::Matrix<float> padd = pmatab + pmatab;
    TIMEIT_END(fkZQ_add);
    TIMEIT_PRINT(fkZQ_add, 0, 0);

    assert_eq(cvadd, padd);

    TIMEIT_BEGIN(cv_mul);
    cv::Mat cvmul = cvmataa.mul(cvmataa);
    TIMEIT_END(cv_mul);
    TIMEIT_PRINT(cv_mul, 0, 0);

    TIMEIT_BEGIN(fkZQ_mul);
    fkZQ::Matrix<float> pmul = pmataa.mul(pmataa);
    TIMEIT_END(fkZQ_mul);
    TIMEIT_PRINT(fkZQ_mul, 0, 0);

    assert_eq(cvmul, pmul);

    TIMEIT_BEGIN(cv_matmul);
    cv::Mat cvmatmul = cvmatab * cvmatba;
    TIMEIT_END(cv_matmul);
    TIMEIT_PRINT(cv_matmul, 0, 0);

    TIMEIT_BEGIN(fkZQ_matmul);
    fkZQ::Matrix<float> pmatmul = pmatab * pmatba;
    TIMEIT_END(fkZQ_matmul);
    TIMEIT_PRINT(fkZQ_matmul, 0, 0);

    assert_eq(cvmatmul, pmatmul);

    TIMEIT_BEGIN(cv_div);
    cv::Mat cvdiv = cvmatab / (cvmatab + 1);
    TIMEIT_END(cv_div);
    TIMEIT_PRINT(cv_div, 0, 0);

    TIMEIT_BEGIN(fkZQ_div);
    fkZQ::Matrix<float> pdiv = pmatab / (pmatab + 1);
    TIMEIT_END(fkZQ_div);
    TIMEIT_PRINT(fkZQ_div, 0, 0);

    assert_eq(cvdiv, pdiv);

    TIMEIT_BEGIN(cv_transpose);
    cv::Mat cvtrans = cvmatab.t();
    TIMEIT_END(cv_transpose);
    TIMEIT_PRINT(cv_transpose, 0, 0);

    TIMEIT_BEGIN(fkZQ_transpose);
    fkZQ::Matrix<float> ptrans = pmatab.transpose();
    TIMEIT_END(fkZQ_transpose);
    TIMEIT_PRINT(fkZQ_transpose, 0, 0);

    assert_eq(cvtrans, ptrans);

    cv::Mat cvbox;
    cvbox.create(ROWS, COLS, CV_32F);
    TIMEIT_BEGIN(cv_boxfilter);
    cv::boxFilter(cvmatab, cvbox, -1, cv::Size(5, 5), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    TIMEIT_END(cv_boxfilter);
    TIMEIT_PRINT(cv_boxfilter, 0, 0);

    fkZQ::Matrix<float> pbox;
    pbox.create(ROWS, COLS);
    TIMEIT_BEGIN(fkZQ_boxfilter);
    box_filter_s(pmatab, pbox, 5);
    TIMEIT_END(fkZQ_boxfilter);
    TIMEIT_PRINT(fkZQ_boxfilter, 0, 0);

    assert_eq(cvbox, pbox);

    return 0;
}