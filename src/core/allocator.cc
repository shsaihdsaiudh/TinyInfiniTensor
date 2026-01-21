#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        // 【小白讲解】内存对齐 (Alignment)
        // 就像盖房子一样，虽然你可能只需要 1 平米，但为了地基稳固（读取速度快），
        // 系统往往要求按照 8 平米（uint64_t 大小）为单位来划地。
        // 这里默认对齐到 8 字节，保证 CPU/GPU 读取最高效。
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            // 【小白讲解】析构函数
            // 当 Allocator 对象被销毁时（比如程序结束），
            // 必须把向 Runtime (OS/CUDA) 申请的真正的物理内存还回去，防止内存泄漏。
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        for (auto it = free_blocks.begin(); it != free_blocks.end(); ++it)
        {
            size_t block_addr = it->first;
            size_t block_size = it->second;

            // 找到了足够大的块
            if (block_size >= size) {
                free_blocks.erase(it);

                size_t remaining_size = block_size - size;
                free_blocks[block_addr + size] = remaining_size;
                used += size;
                return block_addr;
            }
        }

        // 到这里就是没有发现对应的空闲块，所以只能扩容
        size_t new_addr = this->peak;
        this->peak += size;
        this->used += size;

        return new_addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================

        this->used -= size;
        this->free_blocks[addr] = size;

        auto it = this->free_blocks.find(addr);

        auto next_it = it;
        next_it++;
        if (next_it != this->free_blocks.end() && it->first + it->second == next_it->first) {
            it->second += next_it->second;
            this->free_blocks.erase(next_it);
        }

        // 开始往前找，也就是说它不是第一个。
        if (it != this->free_blocks.begin()) {
            auto prev_it = it;
            prev_it--;
            if (prev_it->first + prev_it->second == it->first) {
                prev_it->second += it->second;
                this->free_blocks.erase(it);
               it = prev_it; // update 'it' to point to the merged block
            }
        }
        
        // 4. 尾部优化检查 (Tail Optimization)
        // 经过上面的合并，'it' 指向的是这一大片连续空闲区域的起始位置
        if (it->first + it->second == this->peak) {
            // 居然连通到了 peak！全部回收！
            this->peak -= it->second;
            this->free_blocks.erase(it);
        }
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            // 【小白讲解】延迟分配 (Lazy Allocation)
            // 这是一个非常关键的设计！
            // 在 alloc() 阶段，我们实际上并没有真的去买内存，只是在纸上（offset）画地盘。
            // 只有当真的需要用到指针时（调用 getPtr），我们才根据之前记录的 peak（峰值大小），
            // 一次性向系统申请一整块足够大的内存。
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        // 【小白讲解】向上取整公式
        // 比如 size=1, alignment=8 -> returns 8
        // 比如 size=8, alignment=8 -> returns 8
        // 比如 size=9, alignment=8 -> returns 16
        // 这个公式 ((size - 1) / alignment + 1) * alignment 是通用的向上取整写法。
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
