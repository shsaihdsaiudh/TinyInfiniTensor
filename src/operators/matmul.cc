#include "operators/matmul.h"
#include "utils/operator_utils.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================

        auto A = inputs[0];
        auto B = inputs[1];

        auto shapeA = A->getDims();
        auto shapeB = B->getDims();

        auto rankA = shapeA.size();
        auto rankB = shapeB.size();

        int currnerK_A = transA ? shapeA[rankA - 2] : shapeA[rankA - 1];
        int currentM = transA ? shapeA[rankA - 1] : shapeA[rankA - 2];


        int currnerK_B = transB ? shapeB[rankB - 1] : shapeB[rankB - 2];
        int currentN = transB ? shapeB[rankB - 2] : shapeB[rankB - 1];

        IT_ASSERT(currnerK_A == currnerK_B);

        Shape batchA(shapeA.begin(), shapeA.end() - 2);
        Shape batchB(shapeB.begin(), shapeB.end() - 2);

        Shape outputShape = infer_broadcast(batchA, batchB);

        outputShape.push_back(currentM);
        outputShape.push_back(currentN);

        return {{outputShape}};
    }

} // namespace infini