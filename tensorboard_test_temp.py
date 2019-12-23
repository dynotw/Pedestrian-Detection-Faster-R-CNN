import torchvision.models  as models
import torch
from tensorboardX import SummaryWriter


resnet18 = models.resnet18(False)
dummy_input = torch.rand(6, 3, 224, 224)
resnet18 = resnet18.cuda()
dummy_input = dummy_input.cuda()
temp_logger = SummaryWriter("templogs")
temp_logger.add_graph(resnet18, dummy_input)
for name, param in resnet18.named_parameters():
    temp_logger.add_histogram(name, param.clone().cpu().data.numpy(), 10)