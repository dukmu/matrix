#pragma once
#include "matrix.h"

using fkZQ::Matrix;
using fkZQ::AlignedMalloc;
using fkZQ::AlignedFree;
using uchar = unsigned char;
int *get_pos(int L, int k)
{
    int *pos = (int *)malloc((L + 2 * k) * sizeof(int));
    for (int i = 0; i < k; i++)
        pos[i] = k - 1 - i;
    for (int i = k; i < L + k; i++)
        pos[i] = i - k;
    for (int i = L + k; i < L + 2 * k; i++)
        pos[i] = 2 * L + k - 1 - i;
    return pos;
}

template <typename IT, typename ST>
void box_filter_s(Matrix<IT> &img_, Matrix<ST> &result, int k_size)
{
    assert(img_.isContinuous());
    int width_ = img_.cols;
    int height_ = img_.rows;

    int k = k_size / 2;
    k_size = 2 * k + 1;
    float ks = 1.0 / ((2 * k + 1) * (2 * k + 1));
    int width = width_ + 2 * k;
    int height = height_ + 2 * k;
    int *pos_row = get_pos(width_, k);
    int *pos_col = get_pos(height_, k);

    uchar *buffer = nullptr;
    uchar *data = nullptr;
    uchar *diff = nullptr;
    Matrix<ST> sum;
    if (result.empty())
        result.create(height_, width_);
    else
    {
        if (result.row() != height_ || result.col() != width_)
            result.create(height_, width_);
    }
    sum.create(height, width);

    buffer = (uchar *)AlignedMalloc<ST>(width_ * sizeof(ST));
    data = (uchar *)AlignedMalloc<IT>(width * sizeof(IT));
    diff = (uchar *)AlignedMalloc<ST>((width_ - 1) * sizeof(ST));
    ST *buffer_ = (ST *)buffer;
    IT *data_ = (IT *)data;
    ST *diff_ = (ST *)diff;
    ST tmp;
    // #pragma omp parallel for
    for (int j = 0; j < height_; j++)
    {
        IT *LinePS = img_.ptr(j);
        ST *LinePD = sum.ptr(j);
        // copy data
        for (int i = 0; i < k; i++)
            data_[i] = LinePS[pos_row[i]];
        for (int i = width_ + k; i < width; i++)
            data_[i] = LinePS[pos_row[i]];
        memcpy(data_ + k, LinePS, width_ * sizeof(IT));

        // diff along current row
        for (int i = 0; i < width_ - 1; i++)
            diff_[i] = (ST)data_[i + k_size] - (ST)data_[i];

        // sum along current row, with windows_size = k_size
        tmp = 0;
        for (int i = 0; i < k_size; i++)
            tmp += (ST)data_[i];
        LinePD[0] = tmp;
        for (int i = 1; i < width; i++)
        {
            tmp += diff_[i - 1];
            LinePD[i] = tmp;
        }
    }
    memset(buffer_, 0, sizeof(ST) * width_);
    // sum along col, first k_size-1 rows
    for (int j = 0; j < k_size - 1; j++)
    {
        ST *LinePS = sum.ptr(pos_col[j]);
        for (int i = 0; i < width_; i++)
            buffer_[i] += LinePS[i];
    }

    for (int j = 0; j < height_; j++)
    {
        ST *LinePD = result.ptr(j);
        ST *LineADD = sum.ptr(pos_col[j + k_size - 1]);
        ST *LineSUB = sum.ptr(pos_col[j]);
        for (int i = 0; i < width_; i++)
        {
            tmp = buffer_[i] + LineADD[i];
            LinePD[i] = tmp;
            buffer_[i] = tmp - LineSUB[i];
        }
    }
    result *= ks;
    free(pos_row);
    free(pos_col);
    AlignedFree(buffer);
    AlignedFree(data);
    AlignedFree(diff);
}