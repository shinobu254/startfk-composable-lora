lora_Linear_forward = None
lora_Linear_load_state_dict = None
lora_Conv2d_forward = None
lora_Conv2d_load_state_dict = None
lora_MultiheadAttention_forward = None
lora_MultiheadAttention_load_state_dict = None
is_sd_1_5 = False
def get_loaded_lora():
    global is_sd_1_5
    if lora_Linear_forward is None:
        load_lora_ext()
    import lora
    try:
        import networks
        is_sd_1_5 = True
    except ImportError:
        pass
    if is_sd_1_5:
        return networks.loaded_networks
    return lora.loaded_loras

def load_lora_ext():
    global is_sd_1_5
    global lora_Linear_forward
    global lora_Linear_load_state_dict
    global lora_Conv2d_forward
    global lora_Conv2d_load_state_dict
    global lora_MultiheadAttention_forward
    global lora_MultiheadAttention_load_state_dict
    if lora_Linear_forward is not None:
        return
    import lora
    is_sd_1_5 = False
    try:
        import networks
        is_sd_1_5 = True
    except ImportError:
        pass
    if is_sd_1_5:
        if hasattr(networks, "network_Linear_forward"):
            lora_Linear_forward = networks.network_Linear_forward
        if hasattr(networks, "network_Linear_load_state_dict"):
            lora_Linear_load_state_dict = networks.network_Linear_load_state_dict
        if hasattr(networks, "network_Conv2d_forward"):
            lora_Conv2d_forward = networks.network_Conv2d_forward
        if hasattr(networks, "network_Conv2d_load_state_dict"):
            lora_Conv2d_load_state_dict = networks.network_Conv2d_load_state_dict
        if hasattr(networks, "network_MultiheadAttention_forward"):
            lora_MultiheadAttention_forward = networks.network_MultiheadAttention_forward
        if hasattr(networks, "network_MultiheadAttention_load_state_dict"):
            lora_MultiheadAttention_load_state_dict = networks.network_MultiheadAttention_load_state_dict
    else:
        if hasattr(lora, "network_Linear_forward"):
            lora_Linear_forward = lora.lora_Linear_forward
        if hasattr(lora, "network_Linear_load_state_dict"):
            lora_Linear_load_state_dict = lora.lora_Linear_load_state_dict
        if hasattr(lora, "network_Conv2d_forward"):
            lora_Conv2d_forward = lora.lora_Conv2d_forward
        if hasattr(lora, "network_Conv2d_load_state_dict"):
            lora_Conv2d_load_state_dict = lora.lora_Conv2d_load_state_dict
        if hasattr(lora, "network_MultiheadAttention_forward"):
            lora_MultiheadAttention_forward = lora.lora_MultiheadAttention_forward
        if hasattr(lora, "network_MultiheadAttention_load_state_dict"):
            lora_MultiheadAttention_load_state_dict = lora.lora_MultiheadAttention_load_state_dict
