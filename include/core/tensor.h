#pragma once
#include "core/blob.h"
#include "core/data_type.h"
#include "core/object.h"
#include "core/runtime.h"
#include <cmath>
#include <cstring>
#include <fstream>

namespace infini
{
    class GraphObj;
    using ShapeElem = int;
    using Shape = vector<ShapeElem>;
    // TensorObj 类：数据载体（张量）
    //
    // 在深度学习中，Tensor 是流动的数据包。它不仅包含数值（data），
    // 还包含元数据（metadata），比如形状 (shape)、数据类型 (dtype)，
    // 以及它在计算图中的位置关系（source 和 targets）。
    class TensorObj : public Object
    {
        friend class GraphObj;

    protected:
        int dim; // 张量的维度数 (Rank)，比如 2D 矩阵 dim=2

        DataType dtype; // 数据类型 (float32, int8 等)
        
        // 计算图连接关系：
        // targets: 谁会用到我？（我是哪些 Op 的输入） -> 消费者列表
        vector<WRef<OperatorObj>> targets;
        
        // source: 我是谁产生的？（我是哪个 Op 的输出） -> 生产者
        WRef<OperatorObj> source;

        // 实际的数据存储
        // Blob 是一个封装了 void* 指针的对象，指向真正的内存地址
        Blob data;
        
        Runtime runtime; // 数据存在哪里？(CPU, CUDA)

    private:
        Shape shape; // 具体的形状，比如 [batch, channel, height, width]
        size_t _size; // 元素总个数缓存 (Cache of Π(shape))，比如 2*3=6
        
        Fuid fuid;    // 全局唯一 ID (Functional Unique ID)
                      // 注意：Cloned tensors share the same id. (浅拷贝共享 ID)
                      // Tensors constructed from scratch have a new id. (新创建的有新 ID)

    public:
        TensorObj(Shape shape, DataType dtype, Runtime runtime);
        virtual ~TensorObj() {}
        string toString() const override;

        // 获取元素总个数 (比如 [2,3] -> 6)
        size_t size() const { return _size; }
        
        // 获取总字节数 (比如 6个float -> 24字节)
        size_t getBytes() const { return _size * dtype.getSize(); }

        Shape getDims() const { return shape; }
        void setShape(Shape shape_);
        size_t getRank() const { return shape.size(); }
        UidBaseType getFuid() const { return fuid; }

        // 设置数据：通过一个生成器函数填充数据（用于 debug 或初始化）
        void setData(
            std::function<void(void *, size_t, DataType)> const &generator) const;

        // 绑定实际的内存块 (Blob)
        // 通常在 dataMalloc 阶段调用，确立物理地址。
        void setDataBlob(const Blob &blob);

        // 打印数据内容 (Debug 用)
        void printData() const;
        
        // 比较两个 Tensor 的数据是否相等（校验结果用）
        bool equalData(const Tensor &rhs, double relativeError = 1e-6) const;

        template <typename T>
        bool equalData(const vector<T> &dataVector)
        {
            IT_ASSERT(size() == dataVector.size());
            IT_ASSERT(DataType::get<T>() == dtype.cpuTypeInt());
            return equalDataImpl(getRawDataPtr<T *>(), dataVector.data(), size());
        }

        // 获取原始数据指针（危险操作，要小心类型转换）
        template <typename T>
        T getRawDataPtr() const
        {
            static_assert(std::is_pointer_v<T>,
                          "Raw data pointer has a type of pointer");
            IT_ASSERT(data != nullptr);
            return data->getPtr<T>();
        }

        DataType getDType() const { return dtype; }
        Runtime getRuntime() const { return runtime; }

        // 获取消费者 Op 列表
        OpVec getTargets() const { return wrefs_to_refs(targets); }
        // 获取生产者 Op
        Operator getSource() const { return source.lock(); }

    private:
        template <class T>
        string dataToString() const
        {
            std::stringstream builder;
            builder << "Tensor: " << guid << std::endl;

            auto numDims = shape.size();
            auto dimSzVec = vector<int>(numDims, 1);
            auto ptr = data->getPtr<T *>();
            dimSzVec[numDims - 1] = shape[numDims - 1];

            for (int i = numDims - 1; i != 0; --i)
                dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];

            for (size_t i = 0, iEnd = size(); i < iEnd; ++i)
            {
                for (size_t j = 0; j < numDims; ++j)
                    if (i % dimSzVec[j] == 0)
                        builder << "[";

                builder << ptr[i];
                for (size_t j = 0; j < numDims; ++j)
                    if ((int)i % dimSzVec[j] == dimSzVec[j] - 1)
                        builder << "]";

                if (i != size() - 1)
                    builder << ", ";

                auto column = (size_t)dimSzVec[numDims - 1];
                if (i % column == column - 1)
                    builder << std::endl;
            }
            return builder.str();
        }

        template <typename T>
        bool equalDataImpl(const T *a, const T *b, size_t size,
                           double relativeError = 1e-6) const
        {
            for (size_t i = 0; i < size; ++i)
            {
                if constexpr (std::is_integral_v<T>)
                {
                    if (a[i] != b[i])
                        return false;
                }
                else if constexpr (std::is_floating_point_v<T>)
                {
                    if (std::min(fabs(a[i]), fabs(b[i])) == 0. &&
                        fabs(a[i] - b[i]) > relativeError)
                    {
                        printf("Error on %lu: %f %f\n", i, a[i], b[i]);
                        return false;
                    }
                    else if (std::min(fabs(a[i]), fabs(b[i])) != 0. &&
                             fabs(a[i] - b[i]) /
                                     std::max(fabs(a[i]), fabs(b[i])) >
                                 relativeError)
                    {
                        printf("Error on %lu: %f %f\n", i, a[i], b[i]);
                        return false;
                    }
                }
                else
                {
                    static_assert(!sizeof(T), "Unsupported data type");
                }
            }
            return true;
        }

        void addTarget(const Operator &op) { targets.emplace_back(op); }
        void setSource(const Operator &op) { source = op; }
        void removeTarget(const Operator &op)
        {
            for (auto itr = targets.begin(); itr != targets.end();)
            {
                if (itr->lock() == op)
                    itr = targets.erase(itr);
                else
                    ++itr;
            }
        }
    };

} // namespace infini
