import pytest
import numpy as np
import my_tensor_module

def test_tensor_creation():
    # 測試是否能成功建立，並且 shape 相符
    np_array = np.zeros((2, 3))
    my_tensor = my_tensor_module.Tensor([2, 3])
    assert my_tensor.shape == list(np_array.shape)

def test_array_with_array():
    # 建立兩個 2x2 的矩陣，全部填滿特定數值
    shape = [2, 2]
    a = my_tensor_module.Tensor(shape, 6.0) # 全為 6.0
    b = my_tensor_module.Tensor(shape, 2.0) # 全為 2.0

    # 測試 Array + Array
    c_add = a + b
    assert c_add[[0, 0]] == 8.0
    assert c_add[[1, 1]] == 8.0

    # 測試 Array - Array
    c_sub = a - b
    assert c_sub[[0, 1]] == 4.0

    # 測試 Array * Array
    c_mul = a * b
    assert c_mul[[1, 0]] == 12.0

    # 測試 Array / Array
    c_div = a / b
    assert c_div[[1, 1]] == 3.0

def test_array_with_scalar():
    a = my_tensor_module.Tensor([3], 10.0) # 1D 向量: [10.0, 10.0, 10.0]
    
    c_add = a + 5.0
    assert c_add[[0]] == 15.0

    c_sub = a - 2.0
    assert c_sub[[1]] == 8.0

    c_mul = a * 3.0
    assert c_mul[[2]] == 30.0

    c_div = a / 2.0
    assert c_div[[0]] == 5.0

def test_scalar_with_array():
    a = my_tensor_module.Tensor([2, 2], 4.0) # 全為 4.0 的矩陣
    
    # 測試反向運算
    c_add = 10.0 + a
    assert c_add[[0, 0]] == 14.0

    c_sub = 10.0 - a  # 重頭戲：這應該等於 6.0
    assert c_sub[[0, 1]] == 6.0

    c_mul = 2.0 * a
    assert c_mul[[1, 0]] == 8.0

    c_div = 20.0 / a  # 重頭戲：這應該等於 5.0
    assert c_div[[1, 1]] == 5.0