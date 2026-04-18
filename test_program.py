import pytest
import numpy as np
import my_NDarray_module

# 測試NDarray的建立
def test_NDarray_creation():
    np_array = np.zeros((2, 3, 4))
    my_NDarray = my_NDarray_module.Tensor([2, 3, 4])
    assert my_NDarray.shape == list(np_array.shape)

    np_array2 = np.zeros((2, 2))
    my_NDarray2 = my_NDarray_module.Tensor([2, 2], [1, 2, 3, 4])
    assert my_NDarray2.shape == list(np_array2.shape)

# 測試NDarray和NDarray的逐位元運算
def test_NDarray_by_NDarray():
    shape = [2, 2]
    a = my_NDarray_module.Tensor(shape, 6.0)
    b = my_NDarray_module.Tensor(shape, [1.0, 2.0, 3.0, 4.0])

    # 加法
    c_add = a + b
    assert c_add[[0, 0]] == 7.0

    # 減法
    c_sub = a - b
    assert c_sub[[0, 1]] == 4.0

    # 乘法
    c_mul = a * b
    assert c_mul[[1, 0]] == 18.0

    # 除法
    c_div = a / b
    assert c_div[[1, 1]] == 1.5

# 測試NDarray和Scalar的逐位元運算
def test_NDarray_with_Scalar():
    shape = [2, 2]
    a = my_NDarray_module.Tensor(shape, 10.0)
    
    # 加法
    c_add = a + 5.0
    assert c_add[[0, 0]] == 15.0

    # 減法
    c_sub = a - 2.0
    assert c_sub[[0, 1]] == 8.0

    # 乘法
    c_mul = a * 3.0
    assert c_mul[[1, 0]] == 30.0

    # 除法
    c_div = a / 2.0
    assert c_div[[1, 1]] == 5.0

# 測試NDarray和Scalar的反向逐位元運算
def test_Scalar_with_NDarray():
    shape = [2, 2]
    a = my_NDarray_module.Tensor(shape, [8.0, 7.0, 6.0, 5.0])
    
    # 加法
    c_add = 10.0 + a
    assert c_add[[0, 0]] == 18.0

    # 減法
    c_sub = 10.0 - a
    assert c_sub[[0, 1]] == 3.0

    # 乘法
    c_mul = 2.0 * a
    assert c_mul[[1, 0]] == 12.0

    # 除法
    c_div = 20.0 / a
    assert c_div[[1, 1]] == 4.0