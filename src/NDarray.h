#ifndef NDARRAY_H
#define NDARRAY_H

#include<iostream>
#include<vector>
#include<numeric>
#include<stdexcept>

template <typename T>
class NDarray
{
private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<T> data_; 

    //計算strides
    //strides可以事先記錄在找data時 橫跨多少維度時應該跳過幾筆資料
    void compute_strides()
    {
        size_t dim = shape_.size();
        strides_.resize(dim);

        if (dim == 0) return;

        strides_[dim - 1] = 1;
        
        for (int i = dim - 2; i >= 0; --i)
        {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

public:
    //建構子
    //這種寫法的目的在於當shape_一出生就會被馬上賦予shape的值
    NDarray(const std::vector<size_t>& shape) : shape_(shape)
    {
        size_t size = 1;
        for (size_t dim : shape_)
        {
            size *= dim;
        }

        data_.resize(size, 0);
        compute_strides();
    }

    //建構子2
    //處理接收data時的情況
    NDarray(const std::vector<size_t>& shape, const std::vector<T>& data) : shape_(shape), data_(data)
    {
        size_t size = 1;

        for (size_t dim : shape_)
        {
            size *= dim;
        }

        if (size != data_.size()) 
        {
            throw std::invalid_argument("data size do not match shape.");
        }

        compute_strides();
    }

    //建構子3
    //將多維陣列統一填入特定數值
    NDarray(const std::vector<size_t>& shape, T initial_value) : shape_(shape)
    {
        size_t size = 1;

        for (size_t dim : shape_)
        {
            size *= dim;
        }

        data_.resize(size, initial_value);
        compute_strides();
    }

    //建構子4
    //手動宣告內部數值建立多維陣列
    //待做
    //

    //存取器
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }

    //讀取或修改多維陣列的資料
    T& operator()(const std::vector<size_t>& indices)
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("Indices dimensions do not match array dimensions.");
        }
        
        size_t flat_index = 0;

        for (size_t i = 0; i < shape_.size(); i++)
        {
            if (indices[i] >= shape_[i])
            {
                throw std::out_of_range("Index out of bounds.");
            }
            
            flat_index += indices[i] * strides_[i];
        }

        return data_[flat_index];
    }

    //僅讀取多維陣列的資料 
    const T& operator()(const std::vector<size_t>& indices) const
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("Indices dimensions do not match array dimensions.");
        }
        
        size_t flat_index = 0;

        for (size_t i = 0; i < shape_.size(); i++)
        {
            if (indices[i] >= shape_[i])
            {
                throw std::invalid_argument("Index out of bounds.");
            }

            flat_index += indices[i] * strides_[i];
        }

        return data_[flat_index];
    }

    //給以後的AVX加速用
    T* data() { return data_.data(); }

    //同上 唯獨版本
    const T* data() const { return data_.data(); }
};

#endif