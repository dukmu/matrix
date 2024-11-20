#pragma once

#define _USE_SIMD

#include <iostream>
#include <cstdlib>
#include <cstring>

#ifdef FKZQ_DEBUG
#define FKZQ_NEW std::cout << "new matrix at" << __FILE__ << " " << __LINE__ << "@" << __FUNCTION__ << ", addr: " << this << std::endl;
#define FKZQ_DELETE std::cout << "delete matrix at" << __FILE__ << " " << __LINE__ << "@" << __FUNCTION__ << ", addr: " << this << std::endl;
#else
#define FKZQ_NEW
#define FKZQ_DELETE
#endif

#ifdef _USE_SIMD
#include <experimental/simd>
namespace stdx = std::experimental;
#define simd stdx::native_simd
#endif

namespace fkZQ
{
    template <typename T>
    class Matrix;

    template <typename T>
    void *AlignedMalloc(size_t size)
    {
        void *ptr = nullptr;
#ifndef _USE_SIMD
        ptr = malloc(size);
#else
        size_t align = sizeof(simd<T>);
        ptr = _aligned_malloc(size, align);
#endif
        if (ptr == nullptr)
        {
            std::cerr << "AlignedMalloc failed" << std::endl;
            exit(1);
        }
        return ptr;
    }
    inline void AlignedFree(void *ptr)
    {
#ifndef _USE_SIMD
        free(ptr);
#else
        _aligned_free(ptr);
#endif
    }

    template <typename U, typename T>
    Matrix<U> inline toType(const Matrix<T> &mat)
    {
        auto ptr = mat.ptr(0);
        Matrix<U> ret(mat.rows, mat.cols);
        for (size_t i = 0; i < mat.size(); ++i)
        {
            ret._data[i] = static_cast<U>(ptr[i]);
        }
        return ret;
    }

    template <typename T>
    class Matrix
    {
    private:
        T *_data;

    public:
        using _T = T;
        size_t rows, cols;
        constexpr static size_t step = sizeof(T);
        ~Matrix();
        Matrix();
        Matrix(Matrix<T> &&other);      // move constructor
        Matrix(const Matrix<T> &other); // copy constructor
        Matrix(size_t _rows, size_t _cols);
        Matrix(size_t _rows, size_t _cols, const T *_data);
        void create(size_t _rows, size_t _cols);
        bool isContinuous();

        void clear();
        void setZero();

        T *data();

        size_t col();
        size_t row();
        size_t size();

        T operator[](size_t index);
        T operator()(size_t row, size_t col);

        T &at(size_t index) const;
        T &at(size_t row, size_t col) const;
        T *ptr(size_t row);
        T *ptr(size_t row, size_t col);

        bool empty();

        template <typename U>
        friend std::ostream &operator<<(std::ostream &o, const Matrix<U> &mat);

    private:
        void add(Matrix<T> &ret, const Matrix<T> &other);
        void add(Matrix<T> &ret, const T &other);
        void sub(Matrix<T> &ret, const Matrix<T> &other);
        void sub(Matrix<T> &ret, const T &other);
        void multiply(Matrix<T> &ret, const Matrix<T> &other);
        void multiply(Matrix<T> &ret, const T &other);
        void mul(Matrix<T> &ret, const Matrix<T> &other);
        void div(Matrix<T> &ret, const Matrix<T> &other);
        void div(Matrix<T> &ret, const T &other);

    public:
        Matrix<T> transpose();
        void operator=(const Matrix<T> &other); // copy assignment
        void operator=(Matrix<T> &&other);      // move assignment
        Matrix<T> operator+(const Matrix<T> &other);
        Matrix<T> operator+(const T &other);
        Matrix<T> operator-(const Matrix<T> &other);
        Matrix<T> operator-(const T &other);
        Matrix<T> operator*(const Matrix<T> &other);
        Matrix<T> operator*(const T &other);
        Matrix<T> mul(const Matrix<T> &other);
        Matrix<T> operator/(const Matrix<T> &other);
        Matrix<T> operator/(const T &other);

        template <typename U>
        friend Matrix<U> operator+(const U &other, const Matrix<U> &mat);
        template <typename U>
        friend Matrix<U> operator-(const U &other, const Matrix<U> &mat);
        template <typename U>
        friend Matrix<U> operator*(const U &other, const Matrix<U> &mat);
        template <typename U>
        friend Matrix<U> operator/(const U &other, const Matrix<U> &mat);

        Matrix<T> operator+=(const Matrix<T> &other);
        Matrix<T> operator+=(const T &other);
        Matrix<T> operator-=(const Matrix<T> &other);
        Matrix<T> operator-=(const T &other);
        Matrix<T> operator*=(const Matrix<T> &other);
        Matrix<T> operator*=(const T &other);
        Matrix<T> operator/=(const Matrix<T> &other);
        Matrix<T> operator/=(const T &other);
    };
}