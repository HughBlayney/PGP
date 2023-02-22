import argparse
import yaml
from train_eval.trainer import Trainer
from train_eval.evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", help="Config file with dataset parameters", required=True
)
parser.add_argument("-r", "--data_root", help="Root directory with data", required=True)
parser.add_argument("-d", "--data_dir", help="Directory to extract data", required=True)
parser.add_argument(
    "-o",
    "--output_root_dir",
    help="Directory to save checkpoints and logs",
    required=True,
)
parser.add_argument(
    "-n", "--num_epochs", help="Number of epochs to run training for", required=True
)
parser.add_argument(
    "--dataset_fractions",
    nargs="*",
    type=float,
    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
args = parser.parse_args()


# Make directories
if not os.path.isdir(args.output_root_dir):
    os.mkdir(args.output_root_dir)

for dataset_fraction in args.dataset_fractions:
    print(f"COMPUTING {dataset_fraction} of the dataset")
    output_dir = os.path.join(
        args.output_root_dir, str(dataset_fraction).replace(".", "_")
    )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if not os.path.isdir(os.path.join(output_dir, "checkpoints")):
        os.mkdir(os.path.join(output_dir, "checkpoints"))
    if not os.path.isdir(os.path.join(output_dir, "tensorboard_logs")):
        os.mkdir(os.path.join(output_dir, "tensorboard_logs"))
    if not os.path.isdir(os.path.join(output_dir, "results")):
        os.mkdir(os.path.join(output_dir, "results"))

    # Load config
    with open(args.config, "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard_logs"))

    # Train
    trainer = Trainer(
        cfg,
        args.data_root,
        args.data_dir,
        writer=writer,
        train_data_fraction=dataset_fraction,
    )
    trainer.train(num_epochs=int(args.num_epochs), output_dir=output_dir)

    # Close tensorboard writer
    writer.close()

    # Evaluate
    evaluator = Evaluator(
        cfg,
        args.data_root,
        args.data_dir,
        os.path.join(output_dir, "checkpoints", "best.tar"),
    )
    evaluator.evaluate(output_dir=output_dir)
