import torch
import torch.nn as nn
from models.aggregators.global_attention import GlobalAttention
from typing import Dict
from torch.distributions import Categorical
import os

# Initialize device:
device = torch.device(
    os.environ.get("GPU", "cuda:0") if torch.cuda.is_available() else "cpu"
)


class GoalConditioned(GlobalAttention):
    """
    Goal conditioned aggregator with following functionality.
    1) Predicts goal probabilities over lane nodes
    2) Samples goals
    3) Outputs goal conditioned encodings for N samples to pass on to the trajectory decoder
    """

    def __init__(self, args):
        """
        args to include

        for aggregating map and agent context
        enc_size: int Dimension of encodings generated by encoder
        emb_size: int Size of embeddings used for queries, keys and values
        num_heads: int Number of attention heads

        for goal prediction
        'pre_train': bool, whether the model is being pre-trained using ground truth goals.
        'context_enc_size': int, size of node encoding
        'target_agent_enc_size': int, size of target agent encoding
        'goal_h1_size': int, size of first layer of goal prediction header
        'goal_h2_size': int, size of second layer of goal prediction header
        'num_samples': int, number of goals to sample
        """
        super(GoalConditioned, self).__init__(args)

        # Goal prediction header
        self.goal_h1 = nn.Linear(
            args["context_enc_size"] + args["target_agent_enc_size"],
            args["goal_h1_size"],
        )
        self.goal_h2 = nn.Linear(args["goal_h1_size"], args["goal_h2_size"])
        self.goal_op = nn.Linear(args["goal_h2_size"], 1)
        self.num_samples = args["num_samples"]
        self.leaky_relu = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Pretraining
        self.pre_train = args["pre_train"]

    def forward(self, encodings: Dict) -> Dict:
        """
        Forward pass for goal conditioned aggregator
        :param encodings: dictionary with encoder outputs
        :return: outputs, dictionary with
            'agg_encoding': aggregated encodings
            'goal_log_probs':  log probabilities over nodes corresponding to predicted goals
        """

        # Unpack encodings:
        target_agent_encoding = encodings["target_agent_encoding"]
        node_encodings = encodings["context_encoding"]["combined"]
        node_masks = encodings["context_encoding"]["combined_masks"]

        # Predict goal log-probabilities
        goal_log_probs = self.compute_goal_probs(
            target_agent_encoding, node_encodings, node_masks
        )

        # If pretraining model, use ground truth goals
        if self.pre_train and self.training:
            max_nodes = node_masks.shape[1]
            goals = (
                encodings["node_seq_gt"][:, -1]
                .unsqueeze(1)
                .repeat(1, self.num_samples)
                .long()
                - max_nodes
            )
        else:
            # If fine-tuning or validating, sample goals
            goals = Categorical(
                torch.exp(goal_log_probs).unsqueeze(1).repeat(1, self.num_samples, 1)
            ).sample()

        # Aggregate context
        agg_enc = super(GoalConditioned, self).forward(encodings)

        # Repeat context vector for number of samples and append goal encodings
        agg_enc = agg_enc.unsqueeze(1).repeat(1, self.num_samples, 1)
        batch_indices = (
            torch.arange(agg_enc.shape[0]).unsqueeze(1).repeat(1, self.num_samples)
        )
        goal_encodings = node_encodings[batch_indices, goals]
        agg_enc = torch.cat((agg_enc, goal_encodings), dim=2)

        # Return outputs
        outputs = {"agg_encoding": agg_enc, "goal_log_probs": goal_log_probs}

        return outputs

    def compute_goal_probs(self, target_agent_encoding, node_encodings, node_masks):
        """
        Forward pass for goal prediction header
        :param target_agent_encoding: tensor encoding the target agent's past motion
        :param node_encodings: tensor of node encodings provided by the encoder
        :param node_masks: masks indicating whether a node exists for a given index in the tensor
        :return:
        """
        # Useful variables
        max_nodes = node_encodings.shape[1]
        target_agent_enc_size = target_agent_encoding.shape[-1]
        node_enc_size = node_encodings.shape[-1]

        # Concatenate node encodings with target agent encoding
        target_agent_encoding = target_agent_encoding.unsqueeze(1).repeat(
            1, max_nodes, 1
        )
        enc = torch.cat((target_agent_encoding, node_encodings), dim=2)

        # Form a single batch of encodings
        masks_goal = ~node_masks.unsqueeze(-1).bool()
        enc_batched = torch.masked_select(enc, masks_goal).reshape(
            -1, target_agent_enc_size + node_enc_size
        )

        # Compute goal log probabilities
        goal_ops_ = self.goal_op(
            self.leaky_relu(self.goal_h2(self.leaky_relu(self.goal_h1(enc_batched))))
        )
        goal_ops = torch.zeros_like(masks_goal).float()
        goal_ops = goal_ops.masked_scatter_(masks_goal, goal_ops_).squeeze(-1)
        goal_log_probs = self.log_softmax(goal_ops + torch.log(1 - node_masks))

        return goal_log_probs
