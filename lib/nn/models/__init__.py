from .convolution_disjoint import (
    MPTCN,
    MPTCN_GCNModel,
    MPTCN_GraphConvModel,
    MPTCN_DiffConvModel,
    MPTCN_ChebConvModel,
    MPTCN_GATConvModel,
)
from .mp_gru import MPGRU_DiffConvModel


def get_model_class(model_name: str):
    # Convolutional models  ###################################################
    if model_name == "mptcn_gcn":
        model = MPTCN_GCNModel
    elif model_name == "mptcn_gconv":
        model = MPTCN_GraphConvModel
    elif model_name == "mptcn_diff":
        model = MPTCN_DiffConvModel
    elif model_name == "mptcn_cheb":
        model = MPTCN_ChebConvModel
    elif model_name == "mptcn_gat":
        model = MPTCN_GATConvModel
    elif model_name == "mp_gru":
        model = MPGRU_DiffConvModel
    # Existing models  ########################################################
    elif model_name == 'gwnet':
        from tsl.nn.models import GraphWaveNetModel
        model = GraphWaveNetModel
    elif model_name == 'gwnet_tts':
        from .gwnet_tts import GraphWaveNetTTSModel
        model = GraphWaveNetTTSModel
    # Additional STGNN backbones  ##############################################
    elif model_name == 'dcrnn':
        from tsl.nn.models import DCRNNModel
        model = DCRNNModel
    elif model_name == 'stgcn':
        from tsl.nn.models import STCNModel
        model = STCNModel
    elif model_name == 'agcrn':
        from tsl.nn.models import AGCRNModel
        model = AGCRNModel
    elif model_name == 'agcrn_bilevel':
        from .agcrn_bilevel import AGCRNBilevelModel
        model = AGCRNBilevelModel
    else:
        raise NotImplementedError(f'Model "{model_name}" not available.')
    return model
