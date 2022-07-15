from typing import List, Tuple

import torch
from torch.nn import Module

from nebullvm.base import DataType, InputInfo


def get_outputs_sizes_torch(
    torch_model: Module, input_tensors: List[torch.Tensor]
) -> List[Tuple[int, ...]]:
    if torch.cuda.is_available():
        input_tensors = [x.cuda() for x in input_tensors]
        torch_model.cuda()
    with torch.no_grad():
        outputs = torch_model(*input_tensors)
        if isinstance(outputs, torch.Tensor):
            return [tuple(outputs.size())]
        else:
            return [tuple(output.size()) for output in outputs]


def create_model_inputs_torch(
    input_infos: List[InputInfo]
) -> List[torch.Tensor]:
    # Compute random tensor using information contained inside input_infos
    input_tensors = (
        torch.randn(input_info.size)
        if input_info.dtype is DataType.FLOAT
        else torch.randint(
            size=input_info.size,
            low=input_info.min_value or 0,
            high=input_info.max_value or 100,
        )
        for input_info in input_infos
    )
    return list(input_tensors)


def run_torch_model(
    torch_model: torch.nn.Module, input_tensors: List[torch.Tensor]
) -> List[torch.Tensor]:
    if torch.cuda.is_available():
        torch_model.cuda()
        input_tensors = (t.cuda() for t in input_tensors)
    with torch.no_grad():
        pred = torch_model(*input_tensors)
    if isinstance(pred, torch.Tensor):
        pred = [pred.cpu()]
    else:
        pred = [p.cpu() for p in pred]
    return pred
