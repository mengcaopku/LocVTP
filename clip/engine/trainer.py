import datetime
import logging
import os
import time

import torch
import torch.distributed as dist

from clip.data import make_data_loader
from clip.utils.comm import get_world_size, synchronize
from clip.utils.metric_logger import MetricLogger
from clip.engine.inference import inference

def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss

def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    writer,
):
    logger = logging.getLogger("tan.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH
    shiftweight = cfg.MODEL.CLIP.LOSS.SHIFTWEIGHT
    fgweight = cfg.MODEL.CLIP.LOSS.FGWEIGHT

    model.train()
    start_training_time = time.time()
    end = time.time()

    for epoch in range(arguments["epoch"], max_epoch + 1):
        max_iteration = len(data_loader)
        last_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch

        for iteration, (clip, shiftClip, capTokens, vidnames, normSlens, captions) in enumerate(data_loader):
            iteration += 1
            data_time = time.time() - end

            clip = clip.to(device)
            shiftClip = shiftClip.to(device)
            capTokens = capTokens.to(device)
            normSlens = normSlens.to(device)
            capTokens = {_key: _val.to(device) for _key, _val in capTokens.items()}
            
            def closure():
                #print("iteration: {}".format(iteration))
                optimizer.zero_grad()
                loss_clip, loss_shift, loss_fgVid, loss_fgWord, loss_fgVidPos, loss_fgWordPos = model(clip, shiftClip, capTokens, normSlens, captions)
                loss = loss_clip + shiftweight * loss_shift + fgweight * loss_fgVid + fgweight * loss_fgWord \
                     + fgweight * loss_fgVidPos + fgweight * loss_fgWordPos
                #loss = loss_shift # For debug
                if iteration % 20 == 0 or iteration == max_iteration:
                    meters.update(loss=reduce_loss(loss.detach()))
                    meters.update(loss_clip=reduce_loss(loss_clip.detach()))
                    meters.update(loss_shift=reduce_loss(loss_shift.detach()))
                    meters.update(loss_fgVid=reduce_loss(loss_fgVid.detach()))
                    meters.update(loss_fgWord=reduce_loss(loss_fgWord.detach()))
                    meters.update(loss_fgVidPos=reduce_loss(loss_fgVidPos.detach()))
                    meters.update(loss_fgWordPos=reduce_loss(loss_fgWordPos.detach()))
                    if writer is not None:
                        writer.add_data('Loss/loss', reduce_loss(loss.detach()), epoch * max_iteration + iteration)
                        writer.add_data('Loss/loss_clip', reduce_loss(loss_clip.detach()), epoch * max_iteration + iteration)
                        writer.add_data('Loss/loss_shift', reduce_loss(loss_shift.detach()), epoch * max_iteration + iteration)
                        writer.add_data('Loss/loss_fgVid', reduce_loss(loss_fgVid.detach()), epoch * max_iteration + iteration)
                        writer.add_data('Loss/loss_fgWord', reduce_loss(loss_fgWord.detach()), epoch * max_iteration + iteration)
                        writer.add_data('Loss/loss_fgVidPos', reduce_loss(loss_fgVidPos.detach()), epoch * max_iteration + iteration)
                        writer.add_data('Loss/loss_fgWordPos', reduce_loss(loss_fgWordPos.detach()), epoch * max_iteration + iteration)
                loss.backward()
                return loss

            optimizer.step(closure)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iteration - iteration + last_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 20 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

        scheduler.step()

        if epoch % checkpoint_period == 0:
            checkpointer.save(f"model_{epoch}e", **arguments)

        if data_loader_val is not None and test_period > 0 and \
            epoch % test_period == 0:
            synchronize()
            inference(
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            synchronize()
            model.train()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
