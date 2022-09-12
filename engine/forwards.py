import torch
from typing import List, Dict, Tuple

def seq2ef(model: torch.nn, data: List, criterion: torch.nn, device: torch.device) -> Tuple[Dict, Dict]:
    imgs, ef = data[0].to(device), data[3].to(device)

    loss, reported_loss = None, 0.0
    ef_pred = torch.squeeze(model(imgs), 1)
    if criterion is not None:
        loss = criterion(ef_pred, ef)
        reported_loss = loss.item()

    outputs = {"ef_pred": ef_pred, "ef_gt": ef, "imgs": imgs}
    losses = {"loss": loss, "reported_loss": reported_loss}

    return losses, outputs

def img2kpts(model: torch.nn, data: List, criterion: torch.nn, device: torch.device) -> Tuple[Dict, Dict]:
    imgs, kpts = data[0].to(device),  data[1].to(device)

    loss, reported_loss = None, 0.0
    kpts_pred = model(imgs)
    if criterion is not None:
        loss = criterion(kpts_pred, kpts)
        reported_loss = loss.item()

    outputs = {"kpts_pred": kpts_pred, "kpts_gt": kpts, "imgs": imgs}
    losses = {"loss": loss, "reported_loss": reported_loss}

    return losses, outputs

def seq2ef_kpts(model: torch.nn, data: List, criterion: torch.nn, device: torch.device) -> Tuple[Dict, Dict]:
    imgs, kpts, ef = data[0].to(device), data[1].to(device), data[3].to(device)

    loss, reported_loss, reported_ef_loss, reported_kpts_loss = None, 0.0, 0.0, 0.0
    ef_pred, kpts_pred = model(imgs)
    ef_pred = torch.squeeze(ef_pred, 1)
    batch_size = kpts_pred.shape[0]
    kpts_pred = torch.reshape(kpts_pred, (batch_size, 40, 2, 2))

    if criterion is not None:
        loss_kpts = criterion(kpts_pred, kpts)
        loss_ef = criterion(ef_pred, ef)
        reported_ef_loss, reported_kpts_loss = loss_ef.item(), loss_kpts.item()
        loss = loss_ef + loss_kpts
        reported_loss = loss.item()

    outputs = {"kpts_pred": kpts_pred, "kpts_gt": kpts, "ef_pred": ef_pred, "ef_gt": ef, "imgs": imgs}
    losses = {"loss": loss, "reported_loss": reported_loss, "reported_ef_loss": reported_ef_loss, "reported_kpts_loss": reported_kpts_loss}

    return losses, outputs

def seq2ef_kpts_sd(model: torch.nn, data: List, criterion: torch.nn, device: torch.device) -> Tuple[Dict, Dict]:
    imgs, kpts, ef = data[0].to(device), data[1].to(device), data[3].to(device)
    index_frame1, index_frame2 = data[6], data[7]   # normalized to [0,..,16] where 0 is a transition (non ES or ED) frame.
    sd = torch.stack((index_frame1, index_frame2), dim=1)
    sd = sd.to(device)

    loss, reported_loss, reported_ef_loss, reported_kpts_loss, reported_sd_loss = None, 0.0, 0.0, 0.0, 0.0
    ef_pred, kpts_pred, sd_pred = model(imgs)
    ef_pred = torch.squeeze(ef_pred, 1)
    batch_size = kpts_pred.shape[0]
    kpts_pred = torch.reshape(kpts_pred, (batch_size, 40, 2, 2))

    if criterion is not None:
        loss_ef = criterion[0](ef_pred, ef)
        loss_sd = criterion[1](sd_pred, sd)
        reported_ef_loss, reported_sd_loss = loss_ef.item(), loss_sd.item()
        loss = loss_ef + loss_sd

        # init kpts loss
        loss_kpts = 0

        # Select SD frames (Non-transition frames). Then apply kpts loss only on SD frames, if exist
        transition_index = -1
        SD_indices_frame1 = torch.where(index_frame1 > transition_index)[0].to(device)
        SD_indices_frame2 = torch.where(index_frame2 > transition_index)[0].to(device)

        # add kpts loss, assuming kpts[:,:,:,0] belong to frame_index1, kpts[:,:,:,1] belong to frame_index2
        if SD_indices_frame1.shape[0] > 0:
            loss_kpts += criterion[0](torch.index_select(kpts_pred[:, :, :, 0], 0, SD_indices_frame1),
                                      torch.index_select(kpts[:, :, :, 0], 0, SD_indices_frame1))
        if SD_indices_frame2.shape[0] > 0:
            loss_kpts += criterion[0](torch.index_select(kpts_pred[:, :, :, 1], 0, SD_indices_frame2),
                                      torch.index_select(kpts[:, :, :, 1], 0, SD_indices_frame2))
        # clean
        SD_indices_frame1.detach().cpu(), SD_indices_frame2.detach().cpu()

        if loss_kpts != 0:
            reported_kpts_loss = loss_kpts.item()
            loss += loss_kpts

        reported_loss = loss.item()

    outputs = {"imgs": imgs, "kpts_pred": kpts_pred, "kpts_gt": kpts, "ef_pred": ef_pred, "ef_gt": ef, "sd_pred": sd_pred, "sd_gt": sd}
    losses = {"loss": loss, "reported_loss": reported_loss, "reported_ef_loss": reported_ef_loss,
              "reported_kpts_loss": reported_kpts_loss, "reported_sd_loss": reported_sd_loss}

    return losses, outputs

def run_forward(model: torch.nn, data: List, criterion: torch.nn, device: torch.device) -> Dict:
    if model.output_type == 'seq2ef':
        losses, outputs = seq2ef(model, data, criterion, device)
    elif model.output_type == 'img2kpts':
        losses, outputs = img2kpts(model, data, criterion, device)
    elif model.output_type == 'seq2ef&kpts':
        losses, outputs = seq2ef_kpts(model, data, criterion, device)
    elif model.output_type == 'seq2ef&kpts&sd':
        losses, outputs = seq2ef_kpts_sd(model, data, criterion, device)

    else:
        raise NotImplementedError("Forward method to model type {} is not supported..".format(model.output_type))

    return losses, outputs
