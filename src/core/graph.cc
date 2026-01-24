#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
#include "operators/transpose.h"
#include "operators/matmul.h"

namespace infini
{

    /**
     * @brief 将一个算子添加到图中，并建立其与相关 Tensor 以及前后级算子之间的连接关系
     * 
     * 这个函数是构建图的核心。在深度学习框架中，图是由 Op(节点) 和 Tensor(边) 组成的。
     */
    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false; // 图结构发生变化，标记为未排序
        ops.push_back(op);
        
        // 1. 处理输入 Tensor：让 Op 连接到它的“祖先”
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                // Tensor 知道自己被这个新来的 op 使用了
                input->addTarget(op);
                
                // 如果这个 Tensor 是由之前的某个算子产生的 (Source)
                if (auto pred = input->getSource())
                {
                    // 那么那个算子就是当前算子的【前驱 (Predecessor)】
                    // 当前算子就是那个算子的【后继 (Successor)】
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        
        // 2. 处理输出 Tensor：让 Op 连接到它的“子孙”
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                // 标记这个输出 Tensor 的生产者是当前的 op
                output->setSource(op);
                
                // 极端情况：如果输出 Tensor 已经提前被后续 Op 引用了（比如先构建了后面的 Op）
                // 则需要补全它们之间的连接关系
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    /**
     * @brief 打印图的结构，用于调试
     */
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

    /**
     * @brief 拓扑排序：确定算子的执行顺序
     * 
     * 只有当一个算子的所有输入 Tensor 都已经计算出来时，该算子才能运行。
     * 该算法确保 ops 列表中的顺序符合执行依赖。
     */
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

        while (sorted.size() < ops.size())
        {
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

    /**
     * @brief 图优化入口（你的作业所在地）
     */
    void GraphObj::optimize()
    {
        // TODO: 在这里设计算法。
        // 提示：你可能需要多次遍历 ops 列表。
        // 对于每一个 Op，检查它的类型，如果是 Transpose，再看它的前驱或后继是什么。

        bool refined = false;
        do
        {
            refined = false;
            for (int i = 0; i < (int)ops.size(); ++i)
            {
                auto op = ops[i];
                // 这里是开始操作去掉转置的操作
                if (auto transOp2 = std::dynamic_pointer_cast<TransposeObj> (op))
                {
                    auto input = transOp2->getInputs(0);
                    auto prevOp = input->getSource();
                    if (prevOp && prevOp->getOpType() == OpType::Transpose)
                    {
                        auto transOp1 = std::dynamic_pointer_cast<TransposeObj> (prevOp);
                        auto perm1 = transOp1->getPermute();
                        auto perm2 = transOp2->getPermute();

                        if (perm1.size() == perm2.size())
                        {   
                            bool isIdentity = true;
                            for (size_t i = 0; i < perm1.size(); ++i)
                            {
                                // 这里检查是否是正确的置换，也就是转置操作
                                if (perm2[perm1[i]] != static_cast<int>(i))
                                {
                                    isIdentity = false;
                                    break;
                                }
                            }

                            if (isIdentity)
                            {
                                // 如果输入的 Tensor 只被一个算子使用
                                if (input->getTargets().size() == 1) {
                                    auto grandInput = transOp1->getInputs(0);
                                    auto grandoutput = transOp2->getOutputs();
                                    auto grandInputSource = grandInput->getSource();

                                    // Remove transOp1 from P
                                    if(grandInputSource) 
                                        grandInputSource->removeSuccessors(transOp1);

                                    // 这里是去掉了中间的两个转置节点
                                    for (auto nextOp : grandoutput[0]->getTargets())
                                    {
                                        nextOp->replaceInput(grandoutput[0], grandInput);
                                        grandInput->addTarget(nextOp);
                                        grandoutput[0]->removeTarget(nextOp);

                                        // Fix Op-Op connectivity
                                        nextOp->removePredecessors(transOp2);
                                        if (grandInputSource) {
                                            nextOp->addPredecessors(grandInputSource);
                                            grandInputSource->addSuccessors(nextOp);
                                        }
                                    }
                                    grandInput->removeTarget(transOp1);
                                    
                                    this->removeTensor(transOp1->getOutput());
                                    this->removeTensor(transOp2->getOutput());
                                    this->removeOperator(transOp1);
                                    this->removeOperator(transOp2);

                                    refined = true;
                                    break;
                                }
                            }
                        }
                        
                    }
                }

                // 这里开始融合转置
                if (auto matmulOp = std::dynamic_pointer_cast<MatmulObj> (op)) {

                    auto inputs = matmulOp->getInputs();
                    
                    auto prevOpA = inputs[0]->getSource();
                    if (prevOpA && prevOpA->getOpType() == OpType::Transpose)
                    {
                        auto transOpA = std::dynamic_pointer_cast<TransposeObj> (prevOpA);
                        auto permA = transOpA->getPermute();
                        auto rankA = permA.size();

                        // 这里先认为是最后的置换是交换最后两个维度
                        bool isSwapLastTwo = true;
                        if (rankA < 2) isSwapLastTwo = false;
                        else
                        {
                            if (permA[rankA - 1] != static_cast<int>(rankA) - 2 || permA[rankA - 2] != static_cast<int>(rankA) - 1)
                            {
                                isSwapLastTwo = false;
                            }
                            // 检查其他维度有没有被转置
                            for (size_t i = 0; i < rankA - 2; ++i)
                            {
                                if (permA[i] != static_cast<int>(i))
                                {
                                    isSwapLastTwo = false;
                                    break;
                                }
                            }
                        }

                        // 如果这里的Input没有被其他地方使用的话
                        if (isSwapLastTwo && inputs[0]->getTargets().size() == 1) {

                            // 这个地方是负负得正，如果之前有转置的话就去掉，之前没有就加上
                            matmulOp->setTransA(!matmulOp->getTransA());

                            // 接下来开始重新连接节点
                            auto transInput = transOpA->getInputs(0);
                            auto intermediateTensor = inputs[0];
                            auto transInputSource = transInput->getSource();

                            matmulOp->replaceInput(intermediateTensor, transInput);
                            transInput->addTarget(matmulOp);
                            transInput->removeTarget(transOpA);
                            intermediateTensor->removeTarget(matmulOp);

                            // Manage Op-Op connections
                            if(transInputSource) 
                                transInputSource->removeSuccessors(transOpA);
                            
                            matmulOp->removePredecessors(transOpA);

                            if(transInputSource) {
                                transInputSource->addSuccessors(matmulOp);
                                matmulOp->addPredecessors(transInputSource);
                            }

                            this->removeTensor(intermediateTensor);
                            this->removeOperator(transOpA);
                        }
                        refined = true;
                        break;
                    }

                    auto prevOpB = inputs[1]->getSource();
                    if (prevOpB && prevOpB->getOpType() == OpType::Transpose)
                    {
                        auto transOpB = std::dynamic_pointer_cast<TransposeObj> (prevOpB);
                        auto permB = transOpB->getPermute();
                        auto rankB = permB.size();

                        // 这里先认为是最后的置换是交换最后两个维度
                        bool isSwapLastTwo = true;
                        if (rankB < 2) isSwapLastTwo = false;
                        else
                        {
                            if (permB[rankB - 1] != static_cast<int>(rankB) - 2 || permB[rankB - 2] != static_cast<int>(rankB) - 1)
                            {
                                isSwapLastTwo = false;
                            }
                            // 检查其他维度有没有被转置
                            for (size_t i = 0; i < rankB - 2; ++i)
                            {
                                if (permB[i] != static_cast<int>(i))
                                {
                                    isSwapLastTwo = false;
                                    break;
                                }
                            }
                        }

                        // 如果这里的Input没有被其他地方使用的话
                        if (isSwapLastTwo && inputs[1]->getTargets().size() == 1) {

                            // 这个地方是负负得正，如果之前有转置的话就去掉，之前没有就加上
                            matmulOp->setTransB(!matmulOp->getTransB());

                            // 接下来开始重新连接节点
                            auto transInput = transOpB->getInputs(0);
                            auto intermediateTensor = inputs[1];
                            auto transInputSource = transInput->getSource();

                            matmulOp->replaceInput(intermediateTensor, transInput);
                            transInput->addTarget(matmulOp);
                            transInput->removeTarget(transOpB);
                            intermediateTensor->removeTarget(matmulOp);

                            // Manage Op-Op connections
                            if(transInputSource) 
                                transInputSource->removeSuccessors(transOpB);
                            
                            matmulOp->removePredecessors(transOpB);

                            if(transInputSource) {
                                transInputSource->addSuccessors(matmulOp);
                                matmulOp->addPredecessors(transInputSource);
                            }

                            this->removeTensor(intermediateTensor);
                            this->removeOperator(transOpB);
                        }
                        refined = true;
                        break;
                    }
                }
                
            }
        } while (refined);
    }

    /**
     * @brief 根据唯一 ID 获取 Tensor 对象
     */
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

    /**
     * @brief 形状推导：根据算子的输入形状，算出输出 Tensor 的形状
     */
    void GraphObj::shape_infer()
    {
        // 必须按拓扑序推导，因为后面的 Op 依赖前面 Op 的输出形状
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value()); // 确保推导成功
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            
            // 更新输出 Tensor 的 Dims（维度信息）
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

    /**
     * @brief 静态内存分配：为图中的所有 Tensor 分配实际的物理地址
     */
    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        std::unordered_map<int, size_t> tensor_offset_map;
        
        // 1. 调用 Allocator 计算每个 Tensor 应该放在内存池的哪个位置
        for (auto &tensor : tensors) {
            size_t offset = allocator.alloc(tensor->getBytes());
            tensor_offset_map[tensor->getFuid()] = offset;
        }

        // 2. 获得内存池的起始基地址（这通常是一个 malloc 出来的大块内存）
        void *head_ptr = allocator.getPtr();
        
        // 3. 将物理地址绑定到 Tensor 对象上
        for (auto &tensor : tensors) {
            size_t offset = tensor_offset_map[tensor->getFuid()];
            // 物理地址 = 基地址 + 偏移量
            void *ptr = (char*)head_ptr + offset;
            tensor->setDataBlob(make_ref<BlobObj>(runtime, ptr));
        }

        allocator.info(); // 打印内存分配详情
    }

    /**
     * @brief 向图中添加新的 Tensor
     */
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
            // 每个 Tensor 必须要么有生产者，要么有消费者，不能凭空存在
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