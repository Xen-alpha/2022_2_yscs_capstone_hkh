import torch

class add_input_layer(torch.nn.Module):
    '''
    You can use this class when you want fault injection to input tensor itself.    
    '''
    def __init__(self, model, *args):
        super().__init__(*args)
        self.input_layer = torch.nn.Identity()
        self.model = model

    def forward(self, x):
        input = self.input_layer(x)
        output = self.model(input)
        return output

class module_restriction:
    def __init__(self, restriction_max_value=float('inf'), restriction_min_value=-float('inf'), device='cuda'):
        if restriction_max_value < restriction_min_value:
            raise ValueError('restriction_max_value must be greater than or equal to restriction_min_value.')
        self.restriction_max_value = torch.Tensor([restriction_max_value]).to(device)
        self.restriction_min_value = torch.Tensor([restriction_min_value]).to(device)
        print(f'Restriction info: max {restriction_max_value}, min {restriction_min_value}\n')

    def restrict_relu(self, model):
        fhooks = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.ReLU or type(module) == torch.nn.ReLU6:
                fhooks.append(
                    module.register_forward_hook(self._restriction_hook)
                )
        return fhooks

    def restrict_maxpool(self, model):
        fhooks = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.MaxPool2d:
                fhooks.append(
                    module.register_forward_hook(self._restriction_hook)
                )
        return fhooks

    def restrict_InvertedResidual(self, model):
        fhooks = []
        for name, module in model.named_modules():
            if type(module).__name__ == 'InvertedResidual':
                fhooks.append(
                    module.register_forward_hook(self._restriction_hook)
                )
        return fhooks

    def restrict_InvertedResidual_last_bn(self, model):
        fhooks = []
        bn_cnt = 0
        for name, module in model.named_modules():
            if type(module).__name__ == 'InvertedResidual':
                bn_cnt = 0
            if type(module) == torch.nn.BatchNorm2d:
                bn_cnt += 1
            if bn_cnt == 3:
                bn_cnt = 0
                fhooks.append(
                    module.register_forward_hook(self._restriction_hook)
                )
        return fhooks

    def _restriction_hook(self, module, input, output):
        torch.fmin(output, self.restriction_max_value, out=output)
        torch.fmax(output, self.restriction_min_value, out=output)

    def _restrict_second_largest_hook(self, module, input, output):
        topk = torch.topk(output.flatten(), k=2, dim=-1, largest=True)
        second_largest_val = topk.values[1]

        topk = torch.topk(output.flatten(), k=2, dim=-1, largest=False)
        second_smallest_val = topk.values[1]

        torch.fmin(output, second_largest_val, out=output)
        torch.fmax(output, second_smallest_val, out=output)

    def _test_hook(self, module, input, output):
        print(f'max:{torch.max(output)} min:{torch.min(output)}')