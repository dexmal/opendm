from transformers import PretrainedConfig, PreTrainedModel


class DMBaseConfig(PretrainedConfig):
    model_type = "opendm"


class DMPreTrainedModel(PreTrainedModel):
    config: DMBaseConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
