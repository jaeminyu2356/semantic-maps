import os
from datetime import datetime
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from src.models.model_factory import build_model, build_criterion
from src.data.data_factory import build_dataloaders
from src.utils.configs import get_default_configuration, load_config
from src.utils.confusion import BinaryConfusionMatrix
from src.data.nuscenes.utils import NUSCENES_CLASS_NAMES
# from src.data.argoverse.utils import ARGOVERSE_CLASS_NAMES
from src.utils.visualise import colorise


def train(dataloader, model, criterion, optimiser, summary, config, epoch):
    model.train()

    # Compute prior probability of occupancy
    prior = torch.tensor(config.prior).cuda()
    prior_log_odds = torch.log(prior / (1 - prior))

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(config.num_class)

    # Iterate over dataloader
    iteration = (epoch - 1) * len(dataloader)
    for i, batch in enumerate(tqdm(dataloader)):

        # Move tensors to GPU
        batch = [t.cuda() for t in batch]

        # Predict class occupancy scores and compute loss
        image, calib, labels, mask = batch
        if config.model == 'ved':
            logits, mu, logvar = model(image)
            loss = criterion(logits, labels, mask, mu, logvar)
        else:
            logits = model(image, calib)
            loss = criterion(logits, labels, mask)

        # Compute gradients and update parameters
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update confusion matrix
        scores = logits.cpu().sigmoid()
        confusion.update(scores > config.score_thresh, labels, mask)

        # Update tensorboard
        if i % config.log_interval == 0:
            summary.add_scalar('train/loss', float(loss), iteration)

        # Visualise
        if i % config.vis_interval == 0:
            visualise(summary, image, scores, labels, mask, iteration, config.train_dataset, split='train')

        iteration += 1

    # Print and record results
    display_results(confusion, config.train_dataset)
    log_results(confusion, config.train_dataset, summary, 'train', epoch)


def evaluate(dataloader, model, criterion, summary, config, epoch):
    model.eval()

    # Compute prior probability of occupancy
    prior = torch.tensor(config.prior).cuda()
    prior_log_odds = torch.log(prior / (1 - prior))

    # Initialise confusion matrix
    confusion = BinaryConfusionMatrix(config.num_class)

    # Iterate over dataset
    for i, batch in enumerate(tqdm(dataloader)):

        # Move tensors to GPU
        batch = [t.cuda() for t in batch]

        # Predict class occupancy scores and compute loss
        image, calib, labels, mask = batch
        with torch.no_grad():
            if config.model == 'ved':
                logits, mu, logvar = model(image)
                loss = criterion(logits, labels, mask, mu, logvar)
            else:
                logits = model(image, calib)
                loss = criterion(logits, labels, mask)

        # Update confusion matrix
        scores = logits.cpu().sigmoid()
        confusion.update(scores > config.score_thresh, labels, mask)

        # Update tensorboard
        if i % config.log_interval == 0:
            summary.add_scalar('val/loss', float(loss), epoch)

        # Visualise
        if i % config.vis_interval == 0:
            visualise(summary, image, scores, labels, mask, epoch, config.train_dataset, split='val')

    # Print and record results
    display_results(confusion, config.train_dataset)
    log_results(confusion, config.train_dataset, summary, 'val', epoch)

    return confusion.mean_iou


def visualise(summary, image, scores, labels, mask, step, dataset, split):
    class_names = NUSCENES_CLASS_NAMES if dataset == 'nuscenes' else ARGOVERSE_CLASS_NAMES

    summary.add_image(split + '/image', image[0], step, dataformats='CHW')
    summary.add_image(split + '/pred', colorise(scores[0], 'coolwarm', 0, 1), step, dataformats='NHWC')
    summary.add_image(split + '/gt', colorise(labels[0], 'coolwarm', 0, 1), step, dataformats='NHWC')


def display_results(confusion, dataset):
    class_names = NUSCENES_CLASS_NAMES if dataset == 'nuscenes' else ARGOVERSE_CLASS_NAMES

    print('\nResults:')
    for name, iou_score in zip(class_names, confusion.iou):
        print('{:20s} {:.3f}'.format(name, iou_score))
    print('{:20s} {:.3f}'.format('MEAN', confusion.mean_iou))


def log_results(confusion, dataset, summary, split, epoch):
    class_names = NUSCENES_CLASS_NAMES if dataset == 'nuscenes' else ARGOVERSE_CLASS_NAMES

    for name, iou_score in zip(class_names, confusion.iou):
        summary.add_scalar(f'{split}/iou/{name}', iou_score, epoch)
    summary.add_scalar(f'{split}/iou/MEAN', confusion.mean_iou, epoch)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_iou):
    if isinstance(model, nn.DataParallel):
        model = model.module

    ckpt = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best_iou': best_iou
    }

    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path)

    # Load model weights
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.load_state_dict(ckpt['model'])

    # Load optimiser state
    optimizer.load_state_dict(ckpt['optimizer'])

    # Load scheduler state
    scheduler.load_state_dict(ckpt['scheduler'])

    return ckpt['epoch'], ckpt['best_iou']


# Load the configuration for this experiment
def get_configuration(args):
    # Load config defaults
    config = get_default_configuration()

    # Load dataset options
    config.merge_from_file(f'configs/datasets/{args.dataset}.yml')

    # Load model options
    config.merge_from_file(f'configs/models/{args.model}.yml')

    # Load experiment options
    config.merge_from_file(f'configs/experiments/{args.experiment}.yml')

    # Restore config from an existing experiment
    if args.resume is not None:
        config.merge_from_file(os.path.join(args.resume, 'config.yml'))

    # Override with command line options
    config.merge_from_list(args.options)

    # Finalise config
    config.freeze()

    return config


def create_experiment(config, tag, resume=None):
    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
    else:
        # Otherwise, generate a run directory based on the current time
        name = datetime.now().strftime('{}_%y-%m-%d--%H-%M-%S').format(tag)
        logdir = os.path.join(os.path.expandvars(config.logdir), name)
        print("\n==> Creating new experiment in directory:\n" + logdir)
        os.makedirs(logdir)

    # Display the config options on-screen
    print(config.dump())

    # Save the current config
    with open(os.path.join(logdir, 'config.yml'), 'w') as f:
        f.write(config.dump())

    return logdir


def main():
    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, default='run', help='optional tag to identify the run')
    parser.add_argument('--dataset', choices=['nuscenes', 'argoverse'], default='nuscenes', help='dataset to train on')
    parser.add_argument('--model', choices=['pyramid', 'vpn', 'ved'], default='pyramid', help='model to train')
    parser.add_argument('--experiment', default='test', help='name of experiment config to load')
    parser.add_argument('--resume', default=None, help='path to an experiment to resume')
    parser.add_argument('--options', nargs='*', default=[], help='list of additional config options as key-val pairs')
    args = parser.parse_args()

    # Load configuration
    config = get_configuration(args)

    # GPU 설정을 config.freeze() 이전에 수정
    if torch.cuda.is_available():
        available_gpus = list(range(torch.cuda.device_count()))  # Get available GPU IDs
        if len(config.gpus) > len(available_gpus):
            print(f"Warning: Only {len(available_gpus)} GPUs are available. Adjusting config.gpus to available GPUs.")
            config.defrost()  # Allow changes to the configuration
            config.gpus = available_gpus[:3]  # Limit to 3 GPUs if more are requested
            config.freeze()  # Refreeze the configuration
        print(f"Using GPUs: {config.gpus}")

        # Define device_ids for DataParallel
        device_ids = config.gpus

        # Setup experiment with DataParallel for multi-GPU
        model = build_model(config.model, config)
        model = nn.DataParallel(model.cuda(), device_ids=device_ids)
    else:
        raise RuntimeError("CUDA is not available.")

    # Create a directory for the experiment
    logdir = create_experiment(config, args.tag, args.resume)  # logdir 생성

    # Create tensorboard summary writer
    summary = SummaryWriter(logdir)  # logdir을 전달

    criterion = build_criterion(config.model, config)
    train_loader, val_loader = build_dataloaders(config.train_dataset, config)

    # Build optimiser and learning rate scheduler
    optimiser = SGD(model.parameters(), config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = MultiStepLR(optimiser, config.lr_milestones, 0.1)

    # Load checkpoint
    if args.resume:
        epoch, best_iou = load_checkpoint(os.path.join(logdir, 'latest.pth'), model, optimiser, lr_scheduler)
    else:
        epoch, best_iou = 1, 0

    # Main training loop
    while epoch <= config.num_epochs:
        print('\n\n=== Beginning epoch {} of {} ==='.format(epoch, config.num_epochs))

        # Train model for one epoch
        train(train_loader, model, criterion, optimiser, summary, config, epoch)

        # Evaluate on the validation set
        val_iou = evaluate(val_loader, model, criterion, summary, config, epoch)

        # Update learning rate
        lr_scheduler.step()

        # Save checkpoints
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(os.path.join(logdir, 'best.pth'), model, optimiser, lr_scheduler, epoch, best_iou)

        save_checkpoint(os.path.join(logdir, 'latest.pth'), model, optimiser, lr_scheduler, epoch, best_iou)

        epoch += 1

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
