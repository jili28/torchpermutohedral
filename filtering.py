# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
try:
    import permuto
except ImportError as e:
    raise(e, "Not installed or torch not loaded")

class PHLFilter(torch.autograd.Function):
    """
    Filters input based on arbitrary feature vectors. Uses a permutohedral
    lattice data structure to efficiently approximate n-dimensional gaussian
    filtering. Complexity is broadly independent of kernel size. Most applicable
    to higher filter dimensions and larger kernel sizes.

    See:
        https://graphics.stanford.edu/papers/permutohedral/

    Args:
        input: input tensor to be filtered.

        features: feature tensor used to filter the input.

        sigmas: the standard deviations of each feature in the filter.

    Returns:
        output (torch.Tensor): output tensor.
    """

    @staticmethod
    def forward(ctx, input, features, sigmas=None):

        scaled_features = features
        if sigmas is not None:
            for i in range(features.size(1)):
                scaled_features[:, i, ...] /= sigmas[i]

        ctx.save_for_backward(scaled_features)
        output_data = permuto.phl_filter_forward(input, scaled_features)
        return output_data

    @staticmethod
    def backward(ctx, grad_output):
        #raise NotImplementedError("PHLFilter does not currently support Backpropagation")
        scaled_features, = ctx.saved_variables
        grad_input = permuto.phl_filter_backward(grad_output, scaled_features)
        return grad_input, None
