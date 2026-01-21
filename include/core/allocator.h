#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
  // Allocator 类：负责管理一块大的连续内存（Memory Pool）。
  // 它的工作不是每次都向操作系统申请内存，而是申请一大块，然后自己切蛋糕分给 Tensor。
  class Allocator
  {
  private:
    Runtime runtime; // 运行时环境（如 CPU 或 CUDA），决定了内存在哪分配

    size_t used; // 当前已经分配出去的内存总大小（字节）

    size_t peak; // 历史最高内存使用量（用来分析内存峰值）

    size_t alignment; // 内存对齐要求（比如为了性能，地址必须是 256 字节的倍数）

    // pointer to the memory actually allocated
    // 指向这块大内存池起始地址的原始指针
    // 真正的物理分配（malloc/cudaMalloc）只发生在这里
    void *ptr;

    // =================================== 作业 ===================================
    // TODO：可能需要设计一个数据结构来存储free block，以便于管理和合并
    // HINT: 可以使用一个 map 来存储 free block，key 为 block 的起始/结尾地址，value 为 block 的大小
    // =================================== 作业 ===================================
    // 关键点讲解：
    // 建议定义：std::map<size_t, size_t> free_blocks;
    // 1. 为什么要存？因为释放内存后，如果不记录哪里空闲，下次就不知道去哪分配了。
    // 2. 为什么用 map？map 是有序的（按地址排序）。
    //    当你在地址 100 释放了 50 字节，你可以立刻检查：
    //    - 地址 150 有没有空闲块？如果有，合并！（向后合并）
    //    - 地址 <100 的前一个空闲块是不是刚好结束于 100？如果是，合并！（向前合并）
    //key: 内存块的起始偏移量 (offset)
    //value: 内存块的大小 (size)
    std::map<size_t, size_t> free_blocks;

  public:
    Allocator(Runtime runtime);

    virtual ~Allocator();

    // function: simulate memory allocation
    // arguments：
    //     size: size of memory block to be allocated
    // return: head address offset of the allocated memory block
    // 核心函数：分配内存
    // 注意：这里返回的 size_t 不是指针，而是相对于 ptr 的【偏移量 (offset)】。
    // 比如返回 100，意味着真实地址是 ptr + 100。
    size_t alloc(size_t size);

    // function: simulate memory free
    // arguments:
    //     addr: head address offset of memory block to be free
    //     size: size of memory block to be freed
    // 核心函数：释放内存
    // 输入的是【偏移量】和【大小】，你需要把这块空间标记为“空闲”，放入上面的 map 中。
    void free(size_t addr, size_t size);

    // function: perform actual memory allocation
    // return: pointer to the head address of the allocated memory
    // 获取真正的物理内存首地址（即 ptr）。
    // 只有在计算实际数据时才需要用到它。
    void *getPtr();

    void info();

  private:
    // function: memory alignment, rouned up
    // return: size of the aligned memory block
    // 辅助函数：计算对齐后的大小。
    // 比如 alignment=4，你申请 3 字节，实际要占用 4 字节。
    size_t getAlignedSize(size_t size);
  };
}
