import torch
import copy

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

class activation_restriction_model(torch.nn.Module):
    def __init__(self, model, restriction_max_value=2147483647, restriction_min_value=-2147483647, *args):
        if restriction_max_value < restriction_min_value:
            raise ValueError('restriction_max_value must be greater than or equal to restriction_min_value.')
        super().__init__(*args)
        self.model = copy.deepcopy(model)
        self.restriction_max_value = torch.Tensor([restriction_max_value])
        self.restriction_min_value = torch.Tensor([restriction_min_value])

    def forward(self, input):
        return self.model(input)

    def restrict_relu(self):
        fhooks = []
        for name, module in self.model.named_children():
            if type(module) == torch.nn.ReLU or type(module) == torch.nn.ReLU6:
                fhooks.append(
                    module.register_forward_hook(self.relu_restriction_hook)
                )
        return fhooks

    def relu_restriction_hook(self, module, input, output):
        torch.minimum(output, self.restriction_max_value, out=output)
        torch.maximum(output, self.restriction_min_value, out=output)
        print(f'max:{torch.max(output)} min:{torch.min(output)}')