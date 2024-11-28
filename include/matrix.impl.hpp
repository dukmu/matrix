#pragma once
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <string>
#include <omp.h>
#include "matrix.h"

#include <xmmintrin.h>

namespace fkZQ
{
    template <typename T>
    Matrix<T>::~Matrix()
    {
        this->clear();
    }
    template <typename T>
    Matrix<T>::Matrix() : _data(nullptr), rows(0), cols(0), step(0), size(0) {}
    template <typename T>
    Matrix<T>::Matrix(Matrix<T> &&other) // move constructor
    {
        this->_data = nullptr;
        this->rows = other.rows;
        this->cols = other.cols;
        this->step = other.step;
        this->size = other.size;
        std::swap(this->_data, other._data);
    }
    template <typename T>
    Matrix<T>::Matrix(const Matrix<T> &other) // copy constructor
    {
        FKZQ_NEW
        this->rows = other.rows;
        this->cols = other.cols;
        this->_data = (T *)AlignedMalloc<T>(this->rows, this->cols, this->step, this->size);
        memcpy(this->_data, other._data, this->size);
    }
    template <typename T>
    Matrix<T>::Matrix(size_t _rows, size_t _cols)
    {
        FKZQ_NEW
        this->rows = _rows;
        this->cols = _cols;
        this->_data = (T *)AlignedMalloc<T>(this->rows, this->cols, this->step, this->size);
    }
    template <typename T>
    Matrix<T>::Matrix(size_t _rows, size_t _cols, const T *_data, bool aligned)
    {
        FKZQ_NEW
        this->rows = _rows;
        this->cols = _cols;
        this->_data = (T *)AlignedMalloc<T>(this->rows, this->cols, this->step, this->size);
        if (aligned)
        {
            memcpy(this->_data, _data, this->size);
        }
        else
        {
            for (size_t i = 0; i < this->rows; i++)
            {
                memcpy(this->ptr(i), _data + i * this->cols, this->cols * sizeof(T));
            }
        }
    }
    template <typename T>
    void Matrix<T>::create(size_t _rows, size_t _cols)
    {
        FKZQ_NEW
        this->clear();
        this->rows = _rows;
        this->cols = _cols;
        this->_data = (T *)AlignedMalloc<T>(this->rows, this->cols, this->step, this->size);
    }
    template <typename T>
    inline bool Matrix<T>::isContinuous() { return true; }

    template <typename T>
    inline void Matrix<T>::clear()
    {
        if (this->_data)
        {
            FKZQ_DELETE
            AlignedFree(this->_data);
        }
        this->_data = nullptr;
        this->rows = 0;
        this->cols = 0;
    }
    template <typename T>
    inline void Matrix<T>::setZero()
    {
        memset(this->_data, 0, this->size);
    }

    template <typename T>
    inline T *Matrix<T>::data() { return this->_data; }

    template <typename T>
    inline size_t Matrix<T>::col() { return cols; }
    template <typename T>
    inline size_t Matrix<T>::row() { return rows; }
    template <typename T>
    inline size_t Matrix<T>::elements() { return rows * step; }

    template <typename T>
    inline T Matrix<T>::operator[](size_t index) { return this->at(index); }
    template <typename T>
    inline T Matrix<T>::operator()(size_t row, size_t col) { return this->at(row, col); }

    template <typename T>
    inline T &Matrix<T>::at(size_t index) const { return this->_data[index]; }
    template <typename T>
    inline T &Matrix<T>::at(size_t row, size_t col) const { return this->_data[row * this->step + col]; }

    template <typename T>
    inline T *Matrix<T>::ptr(size_t row) { return this->_data + row * this->step; }
    template <typename T>
    inline T *Matrix<T>::ptr(size_t row, size_t col) { return this->_data + row * this->step + col; }

    template <typename T>
    bool inline Matrix<T>::empty() { return this->_data == nullptr || this->rows == 0 || this->cols == 0; }

    template <typename U>
    std::ostream &operator<<(std::ostream &o, const Matrix<U> &mat)
    {
        bool dFlag = std::is_same<U, float>::value || std::is_same<U, double>::value;

        if (dFlag)
        {
            o << std::setiosflags(std::ios::right) << std::setiosflags(std::ios::scientific) << std::setprecision(4);
        }

        for (size_t i = 0; i != mat.rows(); ++i)
        {
            for (size_t j = 0; j != mat.cols(); ++j)
            {
                if (dFlag)
                {
                    o << std::setw(12) << mat.at(i, j) << ' ';
                }
                else
                {
                    o << mat.at(i, j) << ' ';
                }
            }

            if (i < mat.rows() - 1)
            {
                o << '\n';
            }
        }

        o << std::endl;
        return o;
    }

    template <typename T>
    void Matrix<T>::add(Matrix<T> &ret, const Matrix<T> &other)
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] + other._data[i];
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *b = (simd<T> *)other._data;
        simd<T> *c = (simd<T> *)ret._data;
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] + b[i];
        }
#endif
    }

    template <typename T>
    void Matrix<T>::add(Matrix<T> &ret, const T &other)
    {
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] + other;
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *c = (simd<T> *)ret._data;
        simd<T> b(other);
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] + b;
        }
#endif
    }

    template <typename T>
    void Matrix<T>::sub(Matrix<T> &ret, const Matrix<T> &other)
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] - other._data[i];
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *b = (simd<T> *)other._data;
        simd<T> *c = (simd<T> *)ret._data;
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] - b[i];
        }
#endif
    }

    template <typename T>
    void Matrix<T>::sub(Matrix<T> &ret, const T &other)
    {
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] - other;
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *c = (simd<T> *)ret._data;
        simd<T> b(other);
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] - b;
        }
#endif
    }

    template <typename T>
    void Matrix<T>::multiply(Matrix<T> &ret, const Matrix<T> &other)
    {
        assert(this->cols == other.rows);
        assert(this->rows == ret.rows);
        assert(other.cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->rows; ++i)
        {
            for (size_t j = 0; j < other.cols; ++j)
            {
                T sum = 0;
                for (size_t k = 0; k < this->cols; ++k)
                {
                    sum += this->_data[i * this->cols + k] * other._data[k * other.cols + j];
                }
                ret.at(i, j) = sum;
            }
        }
#else
        Matrix<T> other_T = other.transpose();
        for (size_t i = 0; i < this->rows; ++i)
        {
            for (size_t j = 0; j < other.cols; ++j)
            {
                simd<T> *a = (simd<T> *)(this->_data + i * this->step);
                simd<T> *b = (simd<T> *)(other_T._data + j * other_T.step);
                simd<T> sum(0);
                int n = this->step / simd<T>::size();
                for (int k = 0; k < n; ++k)
                {
                    sum += a[k] * b[k];
                }
                T sum_ = 0;
                for (int k = 0; k < simd<T>::size(); ++k)
                {
                    sum_ += sum[k];
                }
                ret.at(i, j) = sum_;
            }
        }
#endif
    }

    template <typename T>
    void Matrix<T>::multiply(Matrix<T> &ret, const T &other)
    {
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] * other;
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *c = (simd<T> *)ret._data;
        simd<T> b(other);
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] * b;
        }
#endif
    }

    template <typename T>
    void Matrix<T>::mul(Matrix<T> &ret, const Matrix<T> &other)
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] * other._data[i];
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *b = (simd<T> *)other._data;
        simd<T> *c = (simd<T> *)ret._data;
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] * b[i];
        }
#endif
    }

    template <typename T>
    void Matrix<T>::div(Matrix<T> &ret, const Matrix<T> &other)
    {
        assert(this->rows == other.rows);
        assert(this->cols == other.cols);
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] / other._data[i];
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *b = (simd<T> *)other._data;
        simd<T> *c = (simd<T> *)ret._data;
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] / b[i];
        }
#endif
    }

    template <typename T>
    void Matrix<T>::div(Matrix<T> &ret, const T &other)
    {
        assert(this->rows == ret.rows);
        assert(this->cols == ret.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < this->elements(); ++i)
        {
            ret._data[i] = this->_data[i] / other;
        }
#else
        simd<T> *a = (simd<T> *)this->_data;
        simd<T> *c = (simd<T> *)ret._data;
        simd<T> b(other);
        size_t n = this->elements() / simd<T>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = a[i] / b;
        }
#endif
    }

    template <typename T>
    Matrix<T> Matrix<T>::transpose() const
    {
        Matrix<T> ret(this->cols, this->rows);
        for (size_t i = 0; i < this->rows; ++i)
        {
            for (size_t j = 0; j < this->cols; ++j)
            {
                ret._data[j * ret.step + i] = this->_data[i * this->step + j];
            }
        }
        return ret;
    }

    template <typename T>
    void Matrix<T>::operator=(const Matrix<T> &other)
    {
        FKZQ_NEW
        this->clear();
        this->rows = other.rows;
        this->cols = other.cols;
        this->_data = (T *)AlignedMalloc<T>(this->rows, this->cols, this->step, this->size);
        memcpy(this->_data, other._data, this->size);
    }

    template <typename T>
    void Matrix<T>::operator=(Matrix<T> &&other)
    {
        this->clear();
        this->rows = other.rows;
        this->cols = other.cols;
        this->_data = other._data;
        this->step = other.step;
        this->size = other.size;
        other._data = nullptr;
        other.rows = 0;
        other.cols = 0;
        other.step = 0;
        other.size = 0;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T> &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->add(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const T &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->add(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T> &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->sub(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-(const T &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->sub(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T> &other)
    {
        Matrix<T> ret(this->rows, other.cols);
        this->multiply(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const T &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->multiply(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::mul(const Matrix<T> &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->mul(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator/(const Matrix<T> &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->div(ret, other);
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator/(const T &other)
    {
        Matrix<T> ret(this->rows, this->cols);
        this->div(ret, other);
        return ret;
    }

    template <typename U>
    Matrix<U> operator+(const U &other, const Matrix<U> &mat)
    {
        Matrix<U> ret(mat.rows, mat.cols);
        mat.add(ret, other);
        return ret;
    }

    template <typename U>
    Matrix<U> operator-(const U &other, const Matrix<U> &mat)
    {
        Matrix<U> ret(mat.rows, mat.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < mat.elements(); ++i)
        {
            ret._data[i] = other - mat._data[i];
        }
#else
        simd<U> *a = (simd<U> *)mat._data;
        simd<U> *c = (simd<U> *)ret._data;
        simd<U> b(other);
        size_t n = mat.elements() / simd<U>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = b - a[i];
        }
#endif
        return ret;
    }

    template <typename U>
    Matrix<U> operator*(const U &other, const Matrix<U> &mat)
    {
        Matrix<U> ret(mat.rows, mat.cols);
        mat.multiply(ret, other);
        return ret;
    }

    template <typename U>
    Matrix<U> operator/(const U &other, const Matrix<U> &mat)
    {
        Matrix<U> ret(mat.rows, mat.cols);
#ifndef _USE_SIMD
        for (size_t i = 0; i < mat.elements(); ++i)
        {
            ret._data[i] = other / mat._data[i];
        }
#else
        simd<U> *a = (simd<U> *)mat._data;
        simd<U> *c = (simd<U> *)ret._data;
        simd<U> b(other);
        size_t n = mat.elements() / simd<U>::size();
        for (size_t i = 0; i < n; ++i)
        {
            c[i] = b / a[i];
        }
#endif
        return ret;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+=(const Matrix<T> &other)
    {
        this->add(*this, other);
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+=(const T &other)
    {
        this->add(*this, other);
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-=(const Matrix<T> &other)
    {
        this->sub(*this, other);
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator-=(const T &other)
    {
        this->sub(*this, other);
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*=(const Matrix<T> &other)
    {
        this->mul(*this, other);
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*=(const T &other)
    {
        this->multiply(*this, other);
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator/=(const Matrix<T> &other)
    {
        this->div(*this, other);
        return *this;
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator/=(const T &other)
    {
        this->div(*this, other);
        return *this;
    }
}