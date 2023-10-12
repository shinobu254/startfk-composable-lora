#
# Composable-Diffusion with Lora
#
import torch
import gradio as gr

import composable_lora
import composable_lora_function_handler
import lora_ext
import modules.scripts as scripts
from modules import script_callbacks
from modules.processing import StableDiffusionProcessing

def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lora

if not hasattr(composable_lora, 'Linear_forward_before_clora'):
    if hasattr(torch.nn, 'Linear_forward_before_lyco'):
        composable_lora.Linear_forward_before_clora = torch.nn.Linear_forward_before_lyco
    else:
        composable_lora.Linear_forward_before_clora = torch.nn.Linear.forward

if not hasattr(composable_lora, 'Conv2d_forward_before_clora'):
    if hasattr(torch.nn, 'Conv2d_forward_before_lyco'):
        composable_lora.Conv2d_forward_before_clora = torch.nn.Conv2d_forward_before_lyco
    else:
        composable_lora.Conv2d_forward_before_clora = torch.nn.Conv2d.forward

if not hasattr(composable_lora, 'MultiheadAttention_forward_before_clora'):
    if hasattr(torch.nn, 'MultiheadAttention_forward_before_lyco'):
        composable_lora.MultiheadAttention_forward_before_clora = torch.nn.MultiheadAttention_forward_before_lyco
    else:
        composable_lora.MultiheadAttention_forward_before_clora = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'Linear_forward_before_lora'):
    if hasattr(torch.nn, 'Linear_forward_before_lyco'):
        torch.nn.Linear_forward_before_lora = torch.nn.Linear_forward_before_lyco
    else:
        torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
    if hasattr(torch.nn, 'Conv2d_forward_before_lyco'):
        torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d_forward_before_lyco
    else:
        torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
    if hasattr(torch.nn, 'MultiheadAttention_forward_before_lyco'):
        torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention_forward_before_lyco
    else:
        torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward

if hasattr(torch.nn, 'Linear_forward_before_lyco'):
    composable_lora.lyco_notfound = False
else:
    composable_lora.lyco_notfound = True

#torch.nn.Linear.forward = composable_lora.lora_Linear_forward
#torch.nn.Conv2d.forward = composable_lora.lora_Conv2d_forward

def check_install_state():
    if not hasattr(composable_lora, "noop"):
        import warnings
        warnings.warn( #NOTICE: You Must Restart the Panel after Install composable_lora!
            "module 'composable_lora' not found! Please reinstall composable_lora and restart the Panel.")

script_callbacks.on_script_unloaded(unload)
if hasattr(script_callbacks, "on_before_reload"):
    script_callbacks.on_before_reload(check_install_state)
script_callbacks.on_before_ui(check_install_state)

class ComposableLoraScript(scripts.Script):
    def title(self):
        return "Composable Lora"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Composable Lora", open=False):
                if not hasattr(composable_lora, "noop"):
                    gr.Markdown('<span style="color:red">Error! Composable Lora install failed! Please reinstall composable_lora and restart the Panel.</span>')
                enabled = gr.Checkbox(value=False, label="Enabled")
                opt_composable_with_step = gr.Checkbox(value=False, label="Composable LoRA with step")
                opt_uc_text_model_encoder = gr.Checkbox(value=False, label="Use Lora in uc text model encoder")
                opt_uc_diffusion_model = gr.Checkbox(value=False, label="Use Lora in uc diffusion model")
                opt_plot_lora_weight = gr.Checkbox(value=False, label="Plot the LoRA weight in all steps")
                opt_single_no_uc = gr.Checkbox(value=False, label="Don't use LoRA in uc if there're no subprompts")
                opt_hires_step_as_global = gr.Checkbox(value=False, label="Treat hires step as global step")
        return [enabled, opt_composable_with_step, opt_uc_text_model_encoder, opt_uc_diffusion_model, opt_plot_lora_weight, opt_single_no_uc, opt_hires_step_as_global]

    def process(self, p: StableDiffusionProcessing, 
            enabled: bool, 
            opt_composable_with_step: bool, 
            opt_uc_text_model_encoder: bool, opt_uc_diffusion_model: 
            bool, opt_plot_lora_weight: bool, opt_single_no_uc: 
            bool, opt_hires_step_as_global: bool):
        lora_ext.load_lora_ext()
        if lora_ext.is_sd_1_5:
            import composable_lycoris
            if composable_lycoris.has_startfk_lycoris:
                print("Error! in sd panel 1.5, composable-lora not support with startfk-lycoris extension.")
        composable_lora.enabled = enabled
        composable_lora.opt_uc_text_model_encoder = opt_uc_text_model_encoder
        composable_lora.opt_uc_diffusion_model = opt_uc_diffusion_model
        composable_lora.opt_composable_with_step = opt_composable_with_step
        composable_lora.opt_plot_lora_weight = opt_plot_lora_weight
        composable_lora.opt_single_no_uc = opt_single_no_uc
        composable_lora.opt_hires_step_as_global = opt_hires_step_as_global

        composable_lora.num_batches = p.batch_size
        if hasattr(p, "hr_second_pass_steps"):
            hr_second_pass_steps = p.hr_second_pass_steps
        else:
            hr_second_pass_steps = 0
        if opt_hires_step_as_global:
            composable_lora.num_steps = p.steps + hr_second_pass_steps
        else:
            composable_lora.num_steps = p.steps
        composable_lora.num_hires_steps = hr_second_pass_steps

        if not hasattr(composable_lora, "noop"):
            raise ModuleNotFoundError( #NOTICE: You Must Restart the Panel after Install composable_lora!
                "No module named 'composable_lora'! Please reinstall composable_lora and restart the Panel.")
        composable_lora_function_handler.on_enable()
        composable_lora.reset_step_counters()

        prompt = p.all_prompts[0]
        composable_lora.negative_prompt = p.all_negative_prompts[0]
        composable_lora.load_prompt_loras(prompt)
        composable_lora.sd_processing = p

    def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
        composable_lora.sd_processing = p
        composable_lora.reset_counters()

    def postprocess(self, p, processed, *args):
        if not hasattr(composable_lora, "noop"):
            raise ModuleNotFoundError( #NOTICE: You Must Restart the Panel after Install composable_lora!
                "No module named 'composable_lora'! Please reinstall composable_lora and restart the Panel.")
        composable_lora_function_handler.on_disable()
        if composable_lora.enabled:
            if composable_lora.opt_plot_lora_weight:
                processed.images.extend([composable_lora.plot_lora()])
