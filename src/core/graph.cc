#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                // 【小白讲解】建立连接关系
                // 1. 让输入 Tensor 知道自己被这个 op 消费了 (addTarget)
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    // 2. 如果输入 Tensor 有来源（由前一个 op 产生），
                    //    那么前一个 op 就是当前 op 的【前驱 (Predecessor)】
                    //    当前 op 就是前一个 op 的【后继 (Successor)】
                    //    建立起 Op 之间的双向链表关系。
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                // 3. 让输出 Tensor 知道自己是由这个 op 产生的 (setSource)
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    // 4. 如果输出 Tensor 已经被后续的 op 消费了（在构建图时可能先加了后续 op），
                    //    也需要完善 Op 之间的前后继关系。
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        // 【小白讲解】拓扑排序 (Kahn 算法变种)
        // 任何有依赖关系的图（比如计算图 A->B->C），执行顺序必须是 A, B, C。
        // 原理：
        // 1. 找到那些所有依赖（Input Tensor）都已经准备好的 Op。
        // 2. 把它们放入 sorted 列表。
        // 3. 标记它们为“已处理”，这样依赖它们的后续 Op 就会变成“依赖已就绪”。
        // 4. 循环直到所有 Op 都排好序。
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        // 【小白讲解】形状推导 (Shape Inference)
        // 在深度学习中，很多时候我们只知道输入数据的形状（比如图片是 224x224）。
        // 中间每一层的输出形状是可以算出来的。
        // 比如：卷积层的输出形状取决于输入大小、卷积核大小、步长(stride)、填充(padding)等。
        // 这个函数就是遍历所有 Op，让每个 Op 算出它输出 Tensor 应该长什么样。
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        // 【小白讲解】内存分配建议：
        // 1. 遍历所有 Tensor:
        //    图里的 tensors 数组存储了所有的 tensor。
        //
        // 2. 申请内存:
        //    对于每个 tensor，调用 allocator.alloc(tensor->getBytes())。
        //    这个 allocator 就是我们在 allocator.cc 里写的那个，它会返回一个【偏移量 (offset)】。
        //
        // 3. 计算真实地址:
        //    void* ptr = allocator.getPtr(); // 拿到整个大内存池的基地址
        //    void* tensor_ptr = (char*)ptr + offset; // 基地址 + 偏移量 = 真实地址
        //
        // 4. 绑定给 Tensor:
        //    tensor->setDataBlob(make_ref<BlobObj>(runtime, tensor_ptr));
        //    这样这个 Tensor 就知道它的数据存在哪里了。
        //
        // *进阶思考*：
        // 虽然这里还没要求，但真正的深度学习框架会做【内存复用】。
        // 比如 Tensor A 用完了，后面的 Op 不再需要它了，那它的空间就可以释放 (allocator.free) 给 Tensor B 用。
        // 简单的实现可以先把所有 tensor 都 alloc 一遍，暂不考虑 free。
        std::unordered_map<int, size_t> tensor_offset_map;
        for (auto &tensor : tensors) {
            size_t offset = allocator.alloc(tensor->getBytes());
            tensor_offset_map[tensor->getFuid()] = offset;
        }

        // 到这一步正式分配了空间
        void *head_ptr = allocator.getPtr();
        
        // 这里要算出物理地址
        for (auto &tensor : tensors) {
            size_t offset = tensor_offset_map[tensor->getFuid()];
            void *ptr = (char*)head_ptr + offset;
            tensor->setDataBlob(make_ref<BlobObj>(runtime, ptr));
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini