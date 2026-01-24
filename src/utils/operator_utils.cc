#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

/**
 * infer_broadcast: 双向广播形状推导
 * 按照 ONNX/NumPy 标准执行广播逻辑：
 * 1. 如果两个维度不相等且其中一个不为 1，则无法广播。
 * 2. 输出形状的每一维是两个输入维度中较大的那一个。
 */
Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // 提示：可以先反转 Shape A 和 B，从低维向高维处理。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================

    int rankA = A.size();
    int rankB = B.size();
    int maxRank = std::max(rankA, rankB);

    Shape out(maxRank);

    for (int i = 0; i < maxRank; ++i) {
       int dimA = (i < rankA) ? A[rankA - i - 1] : 1;
       int dimB = (i < rankB) ? B[rankB - i - 1] : 1;
       if (dimA == dimB) {
        out[maxRank - 1 - i] = dimA;
       } else if (dimA == 1) {
        out[maxRank - 1 - i] = dimB;
       } else if (dimB == 1) {
        out[maxRank - 1 - i] = dimA;
       } else {
        IT_TODO_HALT();
       }  
    }
    
    return out;
}

/**
 * get_real_axis: 将可能的负数轴索引映射为正确的正数索引
 * @param axis 原始轴索引（如 -1）
 * @param rank 张量的维度总数
 * @return 映射后的正数索引
 */
int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    // 检查范围是否有效，例如 rank=3，axis 必须在 [-3, 2] 之间
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis; // 例如 -1 变为 rank-1
    } else {
        newAxis = axis;
    }
    return newAxis;
}

/**
 * locate_index: 将线性索引转换回多维坐标
 * @param inputN 线性索引（一维偏移量）
 * @param shape  张量的形状
 * @return 对应的多维坐标向量
 * 逻辑：通过不断进行除法和取余运算分解索引
 */
Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem; // 余数为该维度的坐标
        inputN = div.quot; // 商继续进行下一维的分解
    }
    return ans;
}

/**
 * delocate_index: 将多维坐标转换为线性索引
 * @param shapeIndex 多维坐标向量
 * @param shape      张量形状
 * @param stride     张量每一维的步长
 * @return 对应的一维内存偏移量
 */
size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        // 先对坐标取余（处理广播情况），然后累加各维度的偏移
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

/**
 * device_to_str: 将 Device 枚举转为字符串
 */
std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT(); // 目前仅支持 CPU
    }
}

/**
 * get_kernel_attrs_str: 获取内核属性的字符串描述
 */
std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini