import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
x = ms.Tensor([[2, 1], [1, 2]], ms.float32)
print(x.eigvals())
