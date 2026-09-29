import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
diagonal = ms.Tensor([1, 2, 3], ms.float32)
print(ops.MatrixDiagV3()(diagonal, 0, 3, 3, 0))
