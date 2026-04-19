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

    // 計算strides
    // strides可以事先記錄在找data時 橫跨多少維度時應該跳過幾筆資料
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

    template <typename Op>
    NDarray<T> broadcast_operation(const NDarray<T>& other, Op operation) const
    {
        // 如果形狀相同 不用broadcast 直接逐位元運算
        if (shape_ == other.shape_) 
        {
            NDarray<T> result(shape_);
            for (size_t i = 0; i < data_.size(); ++i) 
            {
                result.data_[i] = operation(data_[i], other.data_[i]);
            }
            return result;
        }

        // 形狀不同 進行broadcast
        std::vector<size_t> result_shape;
        std::vector<size_t> strides_A;
        std::vector<size_t> strides_B;

        int ndim_A = shape_.size();
        int ndim_B = other.shape_.size();
        int max_ndim = std::max(ndim_A, ndim_B);

        result_shape.resize(max_ndim);
        strides_A.resize(max_ndim, 0); 
        strides_B.resize(max_ndim, 0);

        for (int i = 0; i < max_ndim; ++i) 
        {
            int idx_A = ndim_A - 1 - i;
            int idx_B = ndim_B - 1 - i;
            int res_idx = max_ndim - 1 - i;

            size_t dim_A = (idx_A >= 0) ? shape_[idx_A] : 1;
            size_t dim_B = (idx_B >= 0) ? other.shape_[idx_B] : 1;

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1) 
            {
                throw std::invalid_argument("Shapes cannot be broadcasted together.");
            }

            result_shape[res_idx] = std::max(dim_A, dim_B);
            strides_A[res_idx] = (dim_A == 1) ? 0 : strides_[idx_A];
            strides_B[res_idx] = (dim_B == 1) ? 0 : other.strides_[idx_B];
        }

        NDarray<T> result(result_shape);

        for (size_t i = 0; i < result.data_.size(); ++i) 
        {
            size_t temp_idx = i;
            size_t flat_idx_A = 0;
            size_t flat_idx_B = 0;

            for (int d = 0; d < max_ndim; ++d) 
            {
                size_t res_stride = result.strides_[d];
                size_t coord = temp_idx / res_stride;
                temp_idx %= res_stride;

                flat_idx_A += coord * strides_A[d];
                flat_idx_B += coord * strides_B[d];
            }

            result.data_[i] = operation(data_[flat_idx_A], other.data_[flat_idx_B]);
        }

        return result;
    }

public:
    // 建構子
    // 這種寫法的目的在於當shape_一出生就會被馬上賦予shape的值
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

    // 建構子2
    // 處理接收data時的情況
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

    // 建構子3
    // 將多維陣列統一填入特定數值
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

    // 建構子4
    // 手動宣告內部數值建立多維陣列
    NDarray(const std::vector<size_t>& shape, std::initializer_list<T> init_list) : shape_(shape), data_(init_list)
    {
        size_t size = 1;

        for (size_t dim : shape_)
        {
            size *= dim;
        }

        if (size != data_.size()) 
        {
            throw std::invalid_argument("Initializer list size does not match shape.");
        }

        compute_strides();
    }

    // 存取器
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }

    // 讀取或修改多維陣列的資料
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

    // 僅讀取多維陣列的資料 
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

    // 給以後的AVX加速用
    T* data() { return data_.data(); }

    // 同上 唯獨版本
    const T* data() const { return data_.data(); }

    // NDarray by NDarray 逐元素相加
    NDarray<T> operator+(const NDarray<T>& other) const
    {
        return broadcast_operation(other, [](T a, T b) { return a + b; });
    }

    // NDarray by NDarray 逐元素相減
    NDarray<T> operator-(const NDarray<T>& other) const
    {
        return broadcast_operation(other, [](T a, T b) { return a - b; });
    }

    // NDarray by NDarray 逐元素相乘
    NDarray<T> operator*(const NDarray<T>& other) const
    {
        return broadcast_operation(other, [](T a, T b) { return a * b; });
    }

    // NDarray by NDarray 逐元素相除
    NDarray<T> operator/(const NDarray<T>& other) const
    {
        return broadcast_operation(other, [](T a, T b) { return a / b; });
    }

    // NDarray by Scalar 逐元素相加
    NDarray<T> operator+(T scalar) const
    {
        NDarray<T> result(shape_); 

        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] + scalar;
        }

        return result;
    }

    // NDarray by Scalar 逐元素相減
    NDarray<T> operator-(T scalar) const
    {
        NDarray<T> result(shape_);

        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] - scalar;
        }

        return result;
    }

    // NDarray by Scalar 逐元素相乘
    NDarray<T> operator*(T scalar) const
    {
        NDarray<T> result(shape_); 

        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] * scalar;
        }

        return result;
    }

    // NDarray by Scalar 逐元素相除
    NDarray<T> operator/(T scalar) const
    {
        NDarray<T> result(shape_); 

        for (size_t i = 0; i < data_.size(); ++i)
        {
            result.data_[i] = data_[i] / scalar;
        }

        return result;
    }

    // NDarray by Scalar 反向逐元素相加
    friend NDarray<T> operator+(T scalar, const NDarray<T>& arr)
    {
        return arr + scalar; 
    }

    // NDarray by Scalar 反向逐元素相減
    friend NDarray<T> operator-(T scalar, const NDarray<T>& arr)
    {
        NDarray<T> result(arr.shape_); 
        
        for (size_t i = 0; i < arr.data_.size(); ++i)
        {
            result.data_[i] = scalar - arr.data_[i];
        }

        return result;
    }

    // NDarray by Scalar 反向逐元素相乘
    friend NDarray<T> operator*(T scalar, const NDarray<T>& arr)
    {
        return arr * scalar;
    }

    // NDarray by Scalar 反向逐元素相除
    friend NDarray<T> operator/(T scalar, const NDarray<T>& arr)
    {
        NDarray<T> result(arr.shape_); 

        for (size_t i = 0; i < arr.data_.size(); ++i)
        {
            result.data_[i] = scalar / arr.data_[i];
        }

        return result;
    }

    // Reduction 沿著特定軸將NDarray壓扁
    // 舉例： 一個形狀為[A, B, C, D]的NDarray
    // 以axis=-1壓扁： 形狀為[1]
    // 以axis=0壓扁： 形狀為[B, C, D]
    // 以axis=1壓扁： 形狀為[A, C, D]
    // 以axis=2壓扁： 形狀為[A, B, D]
    // 以axis=3壓扁： 形狀為[A, B, C]
    NDarray<T> sum(int axis = -1) const
    {
        // 軸為-1時 設定成將NDarray壓扁成一格 (形狀為[1])
        if (axis == -1) 
        {
            NDarray<T> result({1});
            T total = 0;

            for (size_t i = 0; i < data_.size(); ++i) 
            {
                total += data_[i];
            }

            result.data_[0] = total;
            return result;
        }

        // 若軸不對 拋出例外
        if (axis < 0 || axis >= static_cast<int>(shape_.size())) 
        {
            throw std::invalid_argument("Invalid axis for reduction.");
        }

        // 沿著特定軸將NDarray壓扁
        std::vector<size_t> res_shape;
        for (int i = 0; i < static_cast<int>(shape_.size()); ++i) 
        {
            if (i != axis)
            {
                res_shape.push_back(shape_[i]);
            }
        }
        
        // 一維NDarray以axis=0壓扁 會變Scalar 形狀為[1]
        if (res_shape.empty()) 
        {
            res_shape.push_back(1);
        }

        NDarray<T> result(res_shape);

        for (size_t i = 0; i < data_.size(); ++i) 
        {
            size_t temp_idx = i;
            size_t res_flat_idx = 0;
            int res_d = 0;

            // 算出這個元素對應到result的哪個位置
            for (int d = 0; d < static_cast<int>(shape_.size()); ++d) 
            {
                size_t coord = temp_idx / strides_[d];
                temp_idx %= strides_[d];
                
                if (d != axis) 
                {
                    res_flat_idx += coord * result.strides_[res_d];
                    res_d++;
                }
            }

            result.data_[res_flat_idx] += data_[i];
        }

        return result;
    }
};

#endif