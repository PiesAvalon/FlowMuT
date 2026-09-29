import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[1, 2, 3], [4, 5, 6]], ms.float32)
seq_lengths = ms.Tensor([2, 3], ms.int32)
print(x.reverse_sequence(seq_lengths))
