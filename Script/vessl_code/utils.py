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
    def __init__(self, restriction_max_value=2147483647, restriction_min_value=-2147483647, device='cuda'):
        if restriction_max_value < restriction_min_value:
            raise ValueError('restriction_max_value must be greater than or equal to restriction_min_value.')
        self.restriction_max_value = torch.Tensor([restriction_max_value]).to(device)
        self.restriction_min_value = torch.Tensor([restriction_min_value]).to(device)
        print(f'restriction info: max {restriction_max_value}, min {restriction_min_value}\n')

    def restrict_relu(self, model):
        fhooks = []
        for name, module in model.named_modules():
            if type(module) == torch.nn.ReLU or type(module) == torch.nn.ReLU6:
                fhooks.append(
                    module.register_forward_hook(self._restriction_hook)
                )
        return fhooks

    def _restriction_hook(self, module, input, output):
        print(f'BEFORE:: max:{torch.max(output)} min:{torch.min(output)}')
        torch.minimum(output, self.restriction_max_value, out=output)
        torch.maximum(output, self.restriction_min_value, out=output)
        print(f'AFTER::  max:{torch.max(output)} min:{torch.min(output)}')