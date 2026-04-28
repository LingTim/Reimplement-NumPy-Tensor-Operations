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
            int result_idx = max_ndim - 1 - i;

            size_t dim_A = (idx_A >= 0) ? shape_[idx_A] : 1;
            size_t dim_B = (idx_B >= 0) ? other.shape_[idx_B] : 1;

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1) 
            {
                throw std::invalid_argument("Shapes cannot be broadcasted together.");
            }

            result_shape[result_idx] = std::max(dim_A, dim_B);
            strides_A[result_idx] = (dim_A == 1) ? 0 : strides_[idx_A];
            strides_B[result_idx] = (dim_B == 1) ? 0 : other.strides_[idx_B];
        }

        NDarray<T> result(result_shape);

        for (size_t i = 0; i < result.data_.size(); ++i) 
        {
            size_t temp_idx = i;
            size_t flat_idx_A = 0;
            size_t flat_idx_B = 0;

            for (int d = 0; d < max_ndim; ++d) 
            {
                size_t result_stride = result.strides_[d];
                size_t coord = temp_idx / result_stride;
                temp_idx %= result_stride;

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

    // Reduction： 沿著特定軸將NDarray壓扁
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
        std::vector<size_t> result_shape;
        for (int i = 0; i < static_cast<int>(shape_.size()); ++i) 
        {
            if (i != axis)
            {
                result_shape.push_back(shape_[i]);
            }
        }
        
        // 一維NDarray以axis=0壓扁 會變Scalar 形狀為[1]
        if (result_shape.empty()) 
        {
            result_shape.push_back(1);
        }

        NDarray<T> result(result_shape);

        for (size_t i = 0; i < data_.size(); ++i) 
        {
            size_t temp_idx = i;
            size_t result_flat_idx = 0;
            int result_d = 0;

            // 算出這個元素對應到result的哪個位置
            for (int d = 0; d < static_cast<int>(shape_.size()); ++d) 
            {
                size_t coord = temp_idx / strides_[d];
                temp_idx %= strides_[d];
                
                if (d != axis) 
                {
                    result_flat_idx += coord * result.strides_[result_d];
                    result_d++;
                }
            }

            result.data_[result_flat_idx] += data_[i];
        }

        return result;
    }

    // Tensor Contraction: 支援Broadcasting的N維矩陣乘法
    // 矩陣形狀的最後兩位才是真正要做矩陣乘法的形狀 前面的都是Batch
    // 最後兩位不可以不滿足矩陣乘法的限制：a*b和c*d的矩陣要做乘法的話 b與c必須相同
    // 除最後兩位外的形狀只要能滿足broadcasting的話 就可以相同
    NDarray<T> matmul(const NDarray<T>& other) const
    {
        int ndim_A = shape_.size();
        int ndim_B = other.shape_.size();

        // 檢查矩陣的維度至少是2
        if (ndim_A < 2 || ndim_B < 2)
        {
            throw std::invalid_argument("matmul requires arrays to be at least 2D.");
        }

        // 檢查形狀的最後兩位是否符合矩陣乘法的規則
        size_t M = shape_[ndim_A - 2];
        size_t K = shape_[ndim_A - 1];
        size_t other_K = other.shape_[ndim_B - 2];
        size_t N = other.shape_[ndim_B - 1];

        if (K != other_K)
        {
            throw std::invalid_argument("Matrix inner dimensions do not align for multiplication.");
        }

        // 處理Batch的Broadcasting
        int batch_ndim_A = ndim_A - 2;
        int batch_ndim_B = ndim_B - 2;
        int max_batch_ndim = std::max(batch_ndim_A, batch_ndim_B);

        std::vector<size_t> result_batch_shape(max_batch_ndim);
        std::vector<size_t> batch_strides_A(max_batch_ndim, 0);
        std::vector<size_t> batch_strides_B(max_batch_ndim, 0);

        for (int i = 0; i < max_batch_ndim; ++i)
        {
            int idx_A = batch_ndim_A - 1 - i;
            int idx_B = batch_ndim_B - 1 - i;
            int result_idx = max_batch_ndim - 1 - i;

            size_t dim_A = (idx_A >= 0) ? shape_[idx_A] : 1;
            size_t dim_B = (idx_B >= 0) ? other.shape_[idx_B] : 1;

            if (dim_A != dim_B && dim_A != 1 && dim_B != 1)
            {
                throw std::invalid_argument("Batch shapes cannot be broadcasted together.");
            }

            result_batch_shape[result_idx] = std::max(dim_A, dim_B);

            batch_strides_A[result_idx] = (dim_A == 1) ? 0 : strides_[idx_A];
            batch_strides_B[result_idx] = (dim_B == 1) ? 0 : other.strides_[idx_B];
        }

        std::vector<size_t> result_shape = result_batch_shape;
        result_shape.push_back(M);
        result_shape.push_back(N);

        NDarray<T> result(result_shape);

        // 預先提取Matrix的strides 提升迴圈效能
        size_t stride_A_M = strides_[ndim_A - 2];
        size_t stride_A_K = strides_[ndim_A - 1];
        size_t stride_B_K = other.strides_[ndim_B - 2];
        size_t stride_B_N = other.strides_[ndim_B - 1];
        
        int result_ndim = result.shape_.size();
        size_t stride_C_M = result.strides_[result_ndim - 2];
        size_t stride_C_N = result.strides_[result_ndim - 1];

        size_t num_batches = 1;
        for (size_t dim : result_batch_shape)
        {
            num_batches *= dim;
        }

        // 走訪所有廣播後的Batch並相乘
        for (size_t b = 0; b < num_batches; ++b)
        {
            size_t temp_b = b;
            size_t batch_offset_A = 0;
            size_t batch_offset_B = 0;
            size_t batch_offset_C = 0;

            for (int d = max_batch_ndim - 1; d >= 0; --d)
            {
                size_t coord = temp_b % result_batch_shape[d];
                temp_b /= result_batch_shape[d];

                batch_offset_A += coord * batch_strides_A[d];
                batch_offset_B += coord * batch_strides_B[d];
                batch_offset_C += coord * result.strides_[d];
            }

            for (size_t i = 0; i < M; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    size_t idx_A = batch_offset_A + i * stride_A_M + k * stride_A_K;
                    T a_val = data_[idx_A];

                    for (size_t j = 0; j < N; ++j)
                    {
                        size_t idx_B = batch_offset_B + k * stride_B_K + j * stride_B_N;
                        size_t idx_C = batch_offset_C + i * stride_C_M + j * stride_C_N;

                        result.data_[idx_C] += a_val * other.data_[idx_B];
                    }
                }
            }
        }

        return result;
    }
};

#endif