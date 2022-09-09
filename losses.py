import torch
import torch.nn as nn
#from losses.dice_loss import DiceArrayMulticlass, DiceArrayBinary
#from losses.dice_loss_michael import DICE
import torch.nn.functional as F

######################################################
# Home made:
######################################################
def my_L2_loss(output, target):

    loss = torch.mean((output - target)**2)

    return loss


def L2Ignore():

    def loss(output, target, ignore):

        # L2:
        allelements = (output - target) ** 2

        mean_map = torch.mean(allelements, dim=1)
        to_use = 1 - ignore[:, 0, :, :]

        #l2 = torch.mean(allelements)
        l2 = torch.mean(mean_map * to_use)

        return l2

    return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out):
        hinge_loss = 1 - out
        hinge_loss[hinge_loss < 0] = 0
        hinge_loss = 0.5 * (hinge_loss ** 2)
        return hinge_loss


class DrLIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.drlim = HingeLoss()

    def forward(self, emb1, emb2, gt):
        l2part = self.l2(emb1, emb2)
        return gt * l2part + (1 - gt) * self.drlim(l2part)


class CrossEntArray(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, input, target):

        return torch.mean(torch.mean(super().forward(input, target), axis=1), axis=1)


class CrossEntArrayChannelwise(nn.CrossEntropyLoss):
    def __init__(self, num_parts=5):
        super().__init__(reduction='none')
        self.num_parts = num_parts
        self.obj_ids = range(1, self.num_parts, 1)

    def forward(self, input, target):
        masks = target == self.obj_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        for cls_ind in range(self.num_parts):
            ans = torch.mean(torch.mean(super().forward(input, masks[cls_ind]), axis=1), axis=1)

        return ans


######################################################
# Load
######################################################
def load_loss(loss_name, class_weights=None, num_parts=None, printscreen=True, device=None):

    if printscreen:
        print('Using loss : %s..' % loss_name)

    if isinstance(loss_name, list):
        return [load_loss(l, class_weights=class_weights, num_parts=num_parts, printscreen=False, device=device) for l in loss_name]

    if loss_name == 'weighted_categorical_crossentropy':
        print("todo")
        #loss = weighted_categorical_crossentropy(num_classes, coefficients)

    elif loss_name == 'L1':
        loss = nn.L1Loss()

    elif loss_name == 'SmoothL1':
        loss = nn.SmoothL1Loss()

    elif loss_name == 'L2':
        loss = nn.MSELoss()

    elif loss_name == 'BCE':     # 'binary crossentropy':
        loss = nn.BCELoss()

    elif loss_name == 'BCE_sum':
        loss = nn.BCELoss(reduction='sum')

    elif loss_name == 'CrossEnt':  # categorical_crossentropy
        loss = nn.CrossEntropyLoss()

    elif loss_name == 'WeightedCrossEntReduced':  # categorical_crossentropy
        loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    elif loss_name == 'WeightedCrossEnt':  # categorical_crossentropy
        loss = nn.CrossEntropyLoss(weight=class_weights)

    elif loss_name == 'CrossEntArray':  # categorical_crossentropy
        #loss = nn.CrossEntropyLoss(reduction='none')
        loss = CrossEntArray()

    elif loss_name == 'CrossEntArrayChannelwise':  # categorical_crossentropy
        #loss = nn.CrossEntropyLoss(reduction='none')
        loss = CrossEntArrayChannelwise()


    elif loss_name == 'L2Ignore':

        loss = L2Ignore()

    elif loss_name  == 'L2Consistency':
        loss = nn.ModuleList()
        loss.append(nn.MSELoss())
        loss.append(nn.MSELoss())
        loss.append(nn.MSELoss())

    elif loss_name == 'L2Seq':
        loss = nn.ModuleList()
        loss.append(nn.MSELoss())
        loss.append(nn.MSELoss())
        loss.append(nn.MSELoss())
        loss.append(nn.MSELoss())
        loss.append(nn.MSELoss())

    elif loss_name == 'NLLLoss':
        loss = torch.nn.NLLLoss()

    elif loss_name == 'drlim':  # DRLIM http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        loss = DrLIM()

    else:
        print('ERROR: loss name does not exist..')
        return

    return loss


if __name__ == '__main__':
    # (assuming batch size 10)
    num_parts = 5
    dim_size = 3
    batch_size = 10
    #interp_pred = torch.rand(size=(10, 2, 108, 108)) # torch.rand(1, 1) )     #low=0, high=6,
    #interp_gt = torch.rand(size=(10, 2, 108, 108))
    #interp_gt = torch.randint(low=0, high=6, size=(1, 3, 3)).float() # torch.rand(1, 1)
    interp_pred = torch.rand(size=(batch_size, num_parts, dim_size, dim_size)) # torch.rand(1, 1) )     #low=0, high=6,
    #interp_gt = torch.rand(size=(batch_size, num_parts, dim_size, dim_size))  # torch.rand(1, 1) )     #low=0, high=6,
    interp_gt = torch.randint(low=0, high=num_parts+1, size=(batch_size, dim_size, dim_size)).long()  # torch.rand(1, 1)

    #interp_pred = torch.rand(10, 108, 108)
    #interp_gt = torch.rand(10, 108, 108)

    #criterion = load_loss('CrossEntReduced')
    #criterion = load_loss('CrossEnt')

    criterion = load_loss('Dice', num_parts=num_parts)

    #A = torch.argmax(interp_pred, dim=1)

    # loss1 = criterion(interp_pred, torch.argmax(interp_pred, dim=1))
    # loss2 = criterion(input=interp_pred, target=interp_gt)
    #
    # Z1 = interp_pred[1, :, 1, 1]
    # Z2 = A[1,1,1]

    #interp_pred_gt = torch.argmax(interp_pred, dim=1)
    #interp_pred_gt_onehot = torch.transpose(F.one_hot(interp_pred_gt, num_classes=num_parts), 1, 3)
    #interp_gt_onehot = torch.transpose(F.one_hot(interp_gt, num_classes=num_parts+1), 1, 3)[:, 1:, :, :]
    loss1 = criterion(interp_pred, interp_gt)
    #loss1 = criterion(interp_pred, interp_gt)
    #loss2 = criterion(input=interp_pred, target=interp_pred_gt_onehot)
    print('done')
    pass
