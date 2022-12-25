import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np
from abc import ABC
from torch.cuda.amp import custom_fwd, custom_bwd
class HM(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes):
        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, targets, self.features, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)
        targets = self.labels[indexes].clone()
        labels = self.labels.clone()
        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(self.num_samples,1).float().cuda())
        mask = (nums>0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)





class OIM(autograd.Function):
    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               size_average=self.size_average)
        return loss, inputs

class OIMMultiBranch(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, features):
        ctx.features = features
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        return grad_inputs, None

def multibranch(inputs, features):
    return OIMMultiBranch.apply(inputs, features)

class OIMLossMultiBranch(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20, momentum=0.2, 
                 weight=None, size_average=True):
        super(OIMLossMultiBranch, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        loss = 0
        inputs = F.normalize(inputs, dim=1).cuda()
        inputs = multibranch(inputs, self.lut)
        inputs *= self.scalar
        # loss = F.cross_entropy(inputs, targets, weight=self.weight,
        #                        size_average=self.size_average)
        loss += F.cross_entropy(inputs, targets)
        return loss
        # inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        # inputs *= self.scalar
        # loss = F.cross_entropy(inputs, targets, weight=self.weight,
        #                        size_average=self.size_average)
        # return loss, inputs
    def updata_features(self, inputs, targets):
        momentum =self.momentum
        inputs = F.normalize(inputs, dim=1).cuda()
        for x, y in zip(inputs, targets):
            self.lut[y] = momentum * self.lut[y] + (1. - momentum) * x
            self.lut[y] /= self.lut[y].norm()


    







class OIM_ori(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        return outputs
        
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        # for x, y in zip(inputs, targets):
        #     ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
        #     ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None
def oim_ori(inputs, indexes, features, momentum=0.5):
    return OIM_ori.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))
class OIMLoss_ori(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(OIMLoss_ori, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples, ))

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        # print(targets)

        # targets = self.labels[targets]
        # inds = targets >= 0
        # targets = targets[inds]
        # inputs = inputs[inds]
        
        # print(targets, inputs.shape, targets.shape, self.labels.shape, self.features.shape)
        outputs = oim_ori(inputs, targets, self.features, self.momentum)
        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        # print(targets.shape,loss)
        # loss = self.ce(outputs, targets)
        return loss
    def update_features(self, inputs, targets):
        momentum =self.momentum
        inputs = F.normalize(inputs, dim=1).cuda()
        for x, y in zip(inputs, targets):
            self.features[y] = momentum * self.features[y] + (1. - momentum) * x
            self.features[y] /= self.features[y].norm()

