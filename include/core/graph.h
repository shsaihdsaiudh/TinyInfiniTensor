#pragma once
#include "core/allocator.h"
#include "core/operator.h"
#include "core/tensor.h"
#include <algorithm>
#include <cstdint>

namespace infini
{

    // GraphObj 类：计算图的核心容器 (The Core Container of the Computational Graph)
    //
    // 在深度学习框架中，“图 (Graph)” 是用来描述整个神经网络结构的蓝图。
    // 它由两个核心元素组成：
    // 1. Tensors (数据/张量): 就像工厂传送带上的零件。
    // 2. Operators (算子/操作): 就像加工零件的机床。
    //
    // 这个类负责把所有的 Tensor 和 Op 组织起来，告诉它们谁连着谁，
    // 并且负责指挥它们什么时候运行 (topo_sort)、分配内存 (dataMalloc) 以及优化结构 (optimize)。
    class GraphObj : public Object
    {
    protected:
        Runtime runtime; // 整个图运行在哪？ (e.g., CPU, GPU)
        TensorVec tensors; // 所有的 Tensor 清单
        OpVec ops; // 所有的 Op 清单
        Allocator allocator; // 专属的内存分配器

    public:
        explicit GraphObj(Runtime runtime)
            : runtime(runtime), allocator(runtime), sorted(false){};
        string toString() const override;
        Runtime getRuntime() const { return runtime; }

        Tensor addTensor(Shape dim, DataType dtype = DataType::Float32);
        Tensor addTensor(const Tensor &tensor);
        TensorVec addTensor(const TensorVec &tensors);
        
        // 从图中移除一个 Operator
        void removeOperator(Operator op)
        {
            auto it = std::find(ops.begin(), ops.end(), op);
            if (it != ops.end())
                ops.erase(it);
        }

        // 从图中移除一个 Tensor
        void removeTensor(Tensor tensor)
        {
            auto it = std::find(tensors.begin(), tensors.end(), tensor);
            if (it != tensors.end())
                tensors.erase(it);
        }

        const TensorVec &getTensors() const { return tensors; }
        const OpVec &getOperators() const { return ops; }
        Tensor getTensor(int) const;

        /**
         * @brief Sort the nodes in topological order.
         * It returns true if the sorting is successful.
         * Otherwise false is returned, means that there are rings in the graph,
         * so the topological sorting fails.
         */
        // 核心功能：拓扑排序
        // 将所有 Op 按照依赖关系排队。如果 A 的输出是 B 的输入，那么 A 必须排在 B 前面。
        bool topo_sort();

        // 核心功能：图优化 (作业重点)
        // 自动识别低效的计算模式（如 redundant transpose）并消除它们。
        void optimize();

        // 核心功能：形状推导
        // 自动计算网络中每一层中间结果 Tensor 的形状。
        void shape_infer();

        // 核心功能：内存分配 (作业重点)
        // 使用 Allocator 为图中所有的 Tensor 申请实际的物理内存地址。
        void dataMalloc();

        /**
         * @brief Add an operator and create its outputs. Output tensor arguments
         * should be empty Refs (e.g., nullptr).
         */
        // 模版函数：添加 Op 到图中
        // 这是构建网络最常用的函数。
        // 它会自动创建该 Op 的对象，并调用 addOperatorAndConnect 把它连入图中。
        template <typename T, typename... Args>
        Ref<T> addOp(Args &&...args)
        {
            Ref<T> op = infini::make_ref<T>(this, std::forward<Args>(args)...);
            addOperatorAndConnect(op);
            return op;
        }

        /**
         * @brief Add an operator with its outputs specified.
         */
        // 变体：添加 Op，但输出 Tensor 是已经存在的（用于某些特殊连接情况）
        template <typename T, typename... Args>
        Ref<T> addOpWithOutputs(Args &&...args)
        {
            Ref<T> op = infini::make_ref<T>(nullptr, std::forward<Args>(args)...);
            addOperatorAndConnect(op);
            return op;
        }

        /**
         * @brief Gets input tensors of this graph.
         */
        // 获取整个图的输入 Tensor
        // 怎么判断谁是输入？很简单：如果一个 Tensor 没有来源 (source 为空)，那它一定是外部给进来的。
        inline TensorVec getInputs() const
        {
            TensorVec ret;
            for (const auto &t : tensors)
                if (!t->getSource())
                    ret.emplace_back(t);
            return ret;
        }

        /**
         * @brief Gets output tensors of this graph.
         */
        // 获取整个图的输出 Tensor
        // 怎么判断谁是输出？如果一个 Tensor 没有去向 (targets 为空)，那它就是最终结果，没被别人用掉。
        inline TensorVec getOutputs() const
        {
            TensorVec ret;
            for (const auto &t : tensors)
                if (t->getTargets().empty())
                    ret.emplace_back(t);
            return ret;
        }

        // 检查图是不是坏了（debug 用）
        bool checkValid() const;

    private:
        /**
         * @brief Add reverse connections and Op relationship in ctor.
         */
        // 内部辅助函数：建立连接
        // 负责维护 Tensor 和 Op 之间的双向指针（Predecessors/Successors）。
        void addOperatorAndConnect(const Operator &op);

        /**
         * @brief If the nodes is sorted in topological order.
         */
        bool sorted;
    };

} // namespace infini
