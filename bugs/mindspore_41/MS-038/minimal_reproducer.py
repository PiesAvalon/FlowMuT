import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
overlaps = ms.Tensor([[1.0, 0.6, 0.1], [0.6, 1.0, 0.2],
                      [0.1, 0.2, 1.0]], ms.float32)
scores = ms.Tensor([0.9, 0.8, 0.7], ms.float32)
print(ops.NonMaxSuppressionWithOverlaps()(
    overlaps, scores, 2, ms.Tensor(0.5), ms.Tensor(0.0)))
