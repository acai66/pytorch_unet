import torch
from torch import Tensor


def dice_coeff(inputs: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert inputs.size() == target.size()
    if inputs.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {inputs.shape})')

    if inputs.dim() == 2 or reduce_batch_first:
        inter = torch.dot(inputs.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(inputs) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(inputs.shape[0]):
            dice += dice_coeff(inputs[i, ...], target[i, ...])
        return dice / inputs.shape[0]


def multiclass_dice_coeff(inputs: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert inputs.size() == target.size()
    dice = 0
    for channel in range(inputs.shape[1]):
        dice += dice_coeff(inputs[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / inputs.shape[1]


def dice_loss(inputs: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert inputs.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(inputs, target, reduce_batch_first=True)
