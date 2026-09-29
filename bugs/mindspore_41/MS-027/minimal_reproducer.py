import numpy as np
import mindspore as ms
import mindspore.nn as nn

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

class Net(nn.Cell):
    def construct(self, x):
        return x.col2im(output_size=ms.Tensor([4, 4], ms.int32),
                        kernel_size=[2, 2], dilation=[1, 1],
                        padding_value=[0, 0], stride=[1, 1])

x = ms.Tensor(np.arange(36, dtype=np.float32).reshape(1, 4, 9))
print(Net()(x.expand_dims(1)))
