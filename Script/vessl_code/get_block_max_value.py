import torch
import torchvision
import vessl

from torchvision import transforms

class PeekBlock:
  def __init__(self):
    self.result = [-float('inf') for _ in range(17)]
    self.index = 0

  def register_hook(self, model):
    fhooks = []
    for name, module in model.named_modules():
      if type(module).__name__ == 'InvertedResidual':
          fhooks.append(
              module.register_forward_hook(self._peek_hook)
          )
    return fhooks
  
  def reset_index(self):
    self.index = 0

  def _peek_hook(self, module, input, output):
    self.result[self.index] = max(
      torch.max(output),
      self.result[self.index]
    )
    self.index += 1

if __name__ == '__main__':

  dataset = 'cifar100'
  model_name = 'mobilenetv2_x1_0'
  batch_size = 256

  model = torch.hub.load("chenyaofo/pytorch-cifar-models", dataset + '_' + model_name, pretrained=True)
  model.to('cuda')

  transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761])
    ]
  )
  data = torchvision.datasets.CIFAR100(root='/input', train=True, download=True, transform=transform)
  dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

  tool = PeekBlock()
  tool.register_hook(model)

  batch_idx = -1
  for images, labels in dataloader:
    batch_idx += 1
    images = images.to('cuda')
    
    tool.reset_index()
    with torch.no_grad():
      result = model(images)

    for i, value in enumerate(tool.result):
      vessl.log(step=batch_idx, payload={f'block_{i}': value.item()})

  print(tool.result)