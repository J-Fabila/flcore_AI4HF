import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from typing import Dict, Tuple

def make_net(input_size: int, hidden_size: int, num_layers: int, output_size: int, 
             dropout: float = 0, batch_norm: bool = False, act: str = "elu", softplus: bool = True) -> nn.Sequential:
    """
    Create a multi-layer perceptron network.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Size of hidden layers.
        num_layers (int): Number of hidden layers.
        output_size (int): Size of the output.
        dropout (float): Dropout probability.
        batch_norm (bool): Whether to use batch normalization.
        act (str): Activation function ('elu' or 'relu').
        softplus (bool): Whether to add a Softplus activation at the end.

    Returns:
        nn.Sequential: The constructed network.
    """
    act_fn = nn.ELU if act == "elu" else nn.ReLU

    def layer_block(in_size: int, out_size: int) -> list:
        """Create a block of layers for the network."""
        layers = [nn.Linear(in_size, out_size), act_fn()]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_size))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        return layers

    layers = layer_block(input_size, hidden_size)
    for _ in range(num_layers - 1):
        layers.extend(layer_block(hidden_size, hidden_size))
    layers.append(nn.Linear(hidden_size, output_size))
    if softplus:
        layers.append(nn.Softplus())

    return nn.Sequential(*layers)

class BaseSurvODEFunc(nn.Module):
    """Base class for survival ODE functions."""
    def __init__(self, config: Dict):
        """
        Initialize the BaseSurvODEFunc.

        Args:
            config (Dict): Configuration dictionary.
        """
        super(BaseSurvODEFunc, self).__init__()
        self.nfe = 0
        self.batch_time_mode = False
        self.config = config

    def set_batch_time_mode(self, mode: bool = True) -> None:
        """
        Set batch time mode.

        Args:
            mode (bool): Whether to enable batch time mode.
        """
        self.batch_time_mode = mode

    def reset_nfe(self) -> None:
        """Reset the number of function evaluations."""
        self.nfe = 0

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODE function.

        Args:
            t (torch.Tensor): Time tensor.
            y (torch.Tensor): State tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("Not implemented.")

class ContextRecMLPODEFunc(BaseSurvODEFunc):
    """Context-aware Recurrent MLP ODE Function."""
    def __init__(self, config: Dict):
        """
        Initialize the ContextRecMLPODEFunc.

        Args:
            config (Dict): Configuration dictionary.
        """
        super(ContextRecMLPODEFunc, self).__init__(config)
        self.feature_size = config['mlp_output_size']
        self.hidden_size = config['ode_hidden_size']
        self.num_layers = config['ode_num_layers']
        self.batch_norm = config['ode_batch_norm']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = make_net(input_size=self.feature_size + 2, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, output_size=1,
                            batch_norm=self.batch_norm).to(self.device)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ODE function.

        Args:
            t (torch.Tensor): Time tensor.
            y (torch.Tensor): State tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        self.nfe += 1
        Lambda_t = y.index_select(-1, torch.tensor([0]).to(self.device)).view(-1, 1).to(self.device)
        T = y.index_select(-1, torch.tensor([1]).to(self.device)).view(-1, 1).to(self.device)
        x = y.index_select(-1, torch.tensor(range(2, y.size(-1))).to(self.device)).to(self.device)

        inp = torch.cat([Lambda_t, t.repeat(T.size()) * T, x.view(-1, self.feature_size)], dim=1).to(self.device)
        output = self.net(inp) * T
        zeros = torch.zeros_like(y.index_select(-1, torch.tensor(range(1, y.size(-1))).to(self.device))).to(self.device)
        output = torch.cat([output, zeros], dim=1).to(self.device)
        
        return output if self.batch_time_mode else output.squeeze(0)

class NonCoxFuncModel(nn.Module):
    """Non-Cox Function Model for survival analysis."""
    def __init__(self, config: Dict):
        """
        Initialize the NonCoxFuncModel.

        Args:
            config (Dict): Configuration dictionary.
        """
        super(NonCoxFuncModel, self).__init__()
        self.config = config
        self.feature_size = config['mlp_output_size']
        self.odefunc = ContextRecMLPODEFunc(config)
        self.set_last_eval(False)

    def set_last_eval(self, last_eval: bool = True) -> None:
        """
        Set last evaluation mode.

        Args:
            last_eval (bool): Whether to set last evaluation mode.
        """
        self.last_eval = last_eval

    def forward(self, inputs: Dict[str, torch.Tensor], mlp_outputs: torch.Tensor, full_eval: bool) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the Non-Cox Function Model.

        Args:
            inputs (Dict[str, torch.Tensor]): Input dictionary.
            mlp_outputs (torch.Tensor): MLP output tensor.
            full_eval (bool): Whether to perform full evaluation.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary.
        """
        device = next(self.parameters()).device
        orig_t = inputs['t'].squeeze(-1).to(device)
        orig_init_cond = inputs['init_cond'].to(device)
        features = mlp_outputs.to(device)
        init_cond = torch.cat([orig_init_cond.view(-1, 1), orig_t.view(-1, 1), features], dim=1).to(device)
        t = torch.tensor([0., 1.], device=device)

        outputs = {}
        self.odefunc.set_batch_time_mode(False)
        outputs["Lambda"] = odeint(self.odefunc, init_cond, t, rtol=1e-4, atol=1e-8)[1:].squeeze()
        self.odefunc.set_batch_time_mode(True)
        outputs["lambda"] = self.odefunc(t[1:], outputs["Lambda"]).squeeze()

        outputs["Lambda"] = outputs["Lambda"][:, 0]
        outputs["lambda"] = outputs["lambda"][:, 0] / orig_t

        if not self.training:
            self.odefunc.set_batch_time_mode(False)
            ones = torch.ones_like(orig_t)
            t = self.config['time_nums'] * ones
            init_cond = torch.cat([orig_init_cond.view(-1, 1), t.view(-1, 1), features], dim=1)
            t = torch.linspace(0, self.config['time_nums'], self.config['time_nums']+1, device=device)
            t = t / self.config['time_nums']
            outputs["hazard_seq"] = (1 - torch.exp(-odeint(self.odefunc, init_cond, t, rtol=1e-4, atol=1e-8))[1:, :, 0]).transpose(1, 0)
        
        return outputs

class SurvODELoss(nn.Module):
    """Survival ODE Loss function."""
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the SurvODELoss.

        Args:
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super(SurvODELoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reduction = reduction
    
    def forward(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the survival ODE loss.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Calculated loss.
        """
        return self.surv_ode_loss(outputs, labels)

    def surv_ode_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the survival ODE loss.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs.
            labels (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Calculated loss.
        """
        def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
            if reduction == 'none':
                return loss
            elif reduction == 'mean':
                return loss.mean()
            elif reduction == 'sum':
                return loss.sum()
            else:
                raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

        batch_loss = -labels.to(self.device) * torch.log(outputs["lambda"].clamp(min=1e-8)) + outputs["Lambda"]
        return _reduction(batch_loss, self.reduction)

class MLP_Model(nn.Module):
    """Multi-Layer Perceptron Model."""
    def __init__(self, config: Dict):
        """
        Initialize the MLP_Model.

        Args:
            config (Dict): Configuration dictionary.
        """
        super(MLP_Model, self).__init__()
        layers = []
        input_size = config["feature_size"]
        for hidden_size in config["mlp_hidden_sizes"]:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(p=0.2))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, config["mlp_output_size"]))
        self.mlp = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            inputs (Dict[str, torch.Tensor]): Input dictionary.

        Returns:
            torch.Tensor: MLP output.
        """
        features = inputs['features'].to(self.device)
        return self.mlp(features)

class MLP_SODEN(nn.Module):
    """MLP-based Survival ODE Network."""
    def __init__(self, config: Dict, num_labels: int = 1):
        """
        Initialize the MLP_SODEN.

        Args:
            config (Dict): Configuration dictionary.
            num_labels (int): Number of labels.
        """
        super(MLP_SODEN, self).__init__()
        self.num_labels = num_labels
        self.mlp = MLP_Model(config)
        self.classifier = NonCoxFuncModel(config)

    def forward(self, inputs: Dict[str, torch.Tensor], label: torch.Tensor, full_eval: bool = False) -> Tuple[list, torch.Tensor]:
        """
        Forward pass of the MLP-SODEN.

        Args:
            inputs (Dict[str, torch.Tensor]): Input dictionary.
            label (torch.Tensor): True labels.
            full_eval (bool): Whether to perform full evaluation.

        Returns:
            Tuple[list, torch.Tensor]: Output log and loss.
        """
        mlp_output = self.mlp(inputs)
        logits = self.classifier(inputs, mlp_output, full_eval)

        loss_fct = SurvODELoss(reduction='mean')
        outlog = [logits['lambda'], logits['lambda']]
        if not self.training:
            outlog = [logits['lambda'], logits['hazard_seq']]
        
        return outlog, loss_fct(logits, label.view(-1))