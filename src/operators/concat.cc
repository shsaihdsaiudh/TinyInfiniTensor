#include "operators/concat.h"
#include "utils/operator_utils.h"

namespace infini {
/**
 * ConcatObj 构造函数
 * @param graph   所属的计算图
 * @param inputs  输入张量列表，Concat 算子支持连接多个输入
 * @param output  输出张量
 * @param _dim    指定的连接维度（轴）
 */
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
    : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    // get_real_axis 用于处理负值索引（如 -1 映射到最后一个维度）
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

/**
 * inferShape：实现算子的形状推导逻辑
 * 拼接规则：除了指定的 dim 维度外，所有输入张量的其他维度大小必须一致。
 * 输出张量在 dim 维度的大小等于所有输入在该维度大小的累加。
 */
optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // 实现思路：
    // 1. 遍历 inputs 中除了第一个之外的其他输入
    // 2. 检查除 dim 外的维度是否匹配（可选，IT_ASSERT)
    // 3. 将每个输入在 dims[dim] 上的长度累加起来
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================

    for (size_t i = 1; i < inputs.size(); i++) {
        IT_ASSERT(inputs[i]->getRank() == rank);
        for (size_t j = 0; j < rank; j++) {
            if (j == static_cast<size_t>(dim)) {
                dims[j] += inputs[i]->getDims()[j];
                // 不属于拼接的维度
            } else {
                IT_ASSERT(inputs[i]->getDims()[j] == dims[j]);
            }
        }
    }

    return {{dims}};
}

/**
 * toString：将算子转换为可读字符串，方便调试打印
 */
std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]"; // 打印全局唯一 ID
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ","; // 打印每个输入的形状
    os << "dim=" << dim << ","; // 打印连接轴
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ","; // 打印输入张量的 ID
    os << "output=" << outputs[0]->getGuid() << ")"; // 打印输出张量的 ID
    return os.str();
}

} // namespace infini
