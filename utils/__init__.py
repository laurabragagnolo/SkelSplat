from .loss_utils import l1_loss, l2_loss, l2_loss_sqrt, huber_loss, l1_l2_loss
from .loss_utils import l1_huber_loss, l1_loss_masked
from .loss_utils import l1_masked_l2_loss, l1_masked_huber_loss
from .loss_utils import limb_3d_consistency_loss, no_consistency, cauchy_loss, l2_loss_gaussian, l1_loss_gaussian
from .loss_utils import l2_loss_gaussian_l1_loss_gaussian

from .general_utils import OptEarlyStopping, NotStopping


losses = {
    "l1": l1_loss,
    "l2": l2_loss,
    "l2_sqrt": l2_loss_sqrt,
    "huber": huber_loss,
    "l1_l2": l1_l2_loss,
    "l1_huber": l1_huber_loss,
    "l1_masked": l1_loss_masked,
    "l1_masked_l2": l1_masked_l2_loss,
    "l1_masked_huber": l1_masked_huber_loss,
    "cauchy": cauchy_loss,
    "l2_gaussian": l2_loss_gaussian,
    "l2_gaussian_l1_gaussian": l2_loss_gaussian_l1_loss_gaussian,
    "l1_gaussian": l1_loss_gaussian,
}

consistency_losses = {
    "3D_length_consistency": limb_3d_consistency_loss,
    "none": no_consistency
}

early_stopping_strategy = {
    "opt_early_stopping": OptEarlyStopping,
    "no_stopping": NotStopping
}