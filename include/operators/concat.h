#pragma once
#include "core/operator.h"

namespace infini {
/**
 * @brief ConcatObj 类定义了连接（Concatenate）算子。
 * 连接算子的作用是在指定的维度上将多个张量拼接在一起。
 * 要求：除拼接维度外，所有输入张量的形状必须一致。
 *
 */
class ConcatObj : public OperatorObj {
    int dim; // 拼接轴所在的维度索引

  public:
    /**
     * @brief 构造函数，创建一个 Concat 算子实例。
     *
     * @param graph  该算子所属的计算图。
     * @param inputs 连接的输入张量列表。
     * @param output 拼接后的结果张量。
     * @param dim    执行拼接的维度索引。
     */
    ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int dim);
    
    // OP_CLONE 宏用于生成该算子的克隆方法，支持深度拷贝。
    OP_CLONE(ConcatObj);

    /**
     * @brief 根据输入张量推导输出张量的形状。
     */
    optional<vector<Shape>> inferShape(const TensorVec &inputs) override;

    /**
     * @brief 获取算子的字符串描述，便于调试和日志记录。
     */
    std::string toString() const override;

    /**
     * @brief 返回输入张量的个数。
     */
    int numInputs() const override { return inputs.size(); }

    /**
     * @brief 返回输出张量的个数（固定为 1）。
     */
    int numOutputs() const override { return 1; }

    /**
     * @brief 获取当前算子配置的拼接维度。
     */
    int getDim() const { return dim; }
};
} // namespace infini
