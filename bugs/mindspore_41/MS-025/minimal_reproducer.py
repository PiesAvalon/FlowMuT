import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[1, 2], [3, 4], [5, 6]], ms.float32)
segment_ids = ms.Tensor([0, 1, 1], ms.int32)
print(ops.unsorted_segment_min(x, segment_ids, ms.Tensor(2, ms.int32)))
