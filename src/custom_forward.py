import torch
import torch.nn as nn
from typing import Any, Optional, Union


def my_lora_fwd(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                _tmp = lora_A(dropout(x))
                result = result + lora_B(_tmp) * scaling
            else:
                x = dropout(x)
                result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

        result = result.to(torch_result_dtype)

    return result

def merge_peft_lora_layers(model, adapter_name="default"):
    for name, module in model.named_modules():
        if hasattr(module, "base_layer"):

            lora_A = module.lora_A[adapter_name]
            lora_B = module.lora_B[adapter_name]
            base = module.base_layer

            if isinstance(base, nn.Linear):
                delta = lora_B.weight @ lora_A.weight
            elif isinstance(base, nn.Conv2d):
                A = lora_A.weight
                B = lora_B.weight
                delta = torch.matmul(B.flatten(start_dim=1), A.flatten(start_dim=1))
                delta = delta.view_as(base.weight)
            else:
                continue  # skip unsupported layer types

            # Apply scaling
            scaling = module.scaling[adapter_name]
            base.weight.data += delta * scaling

            # Clean up
            del module.lora_A[adapter_name]
            del module.lora_B[adapter_name]
            if hasattr(module, "lora_dropout") and adapter_name in module.lora_dropout:
                del module.lora_dropout[adapter_name]
            if hasattr(module, "use_dora") and adapter_name in module.use_dora:
                del module.use_dora[adapter_name]
            if hasattr(module, "lora_embedding_A") and adapter_name in module.lora_embedding_A:
                del module.lora_embedding_A[adapter_name]
            if hasattr(module, "lora_embedding_B") and adapter_name in module.lora_embedding_B:
                del module.lora_embedding_B[adapter_name]
            if hasattr(module, "lora_magnitude_vector") and adapter_name in module.lora_magnitude_vector:
                del module.lora_magnitude_vector[adapter_name]

def clean_lora_wrappers(model):
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if hasattr(child, "base_layer") and isinstance(child.base_layer, nn.Module):
                # Replace the wrapped module with its base_layer
                setattr(module, child_name, child.base_layer)


def MyUNet2DConditionModel_SD_forward_dual(self, x, mode):
    if mode == "teacher":
        features = [x]
        x = self.teacher_conv_in(x)
        skip = [x]
        for block in self.down_blocks:
            x = block(x, skip)
            features.append(x)
        x = self.mid_block(x)
        features.append(x)
        for block in self.up_blocks:
            x = block(x, skip)
            features.append(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.teacher_conv_out(x)
        features.append(x)
        return features
    else:
        features = [x]
        x = self.student_conv_in(x)
        skip = [x]
        for block in self.down_blocks:
            x = block(x, skip)
            features.append(x)
        x = self.mid_block(x)
        features.append(x)
        for block in self.up_blocks:
            x = block(x, skip)
            features.append(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.student_conv_out(x)
        features.append(x)
        return x, features
    
def MyUNet2DConditionModel_SD_forward(self, x):
    x = self.conv_in(x)
    skip = [x]
    for block in self.down_blocks:
        x = block(x, skip)
    x = self.mid_block(x)
    for block in self.up_blocks:
        x = block(x, skip)
    x = self.conv_norm_out(x)
    x = self.conv_act(x)
    x = self.conv_out(x)
    return x



def MyVAEDecoder_SD_forward(self, x):
    x = self.conv_in(x)
    x = self.mid_block(x)
    for up_block in self.up_blocks:
        x = up_block(x)
    x = self.conv_norm_out(x)
    x = self.conv_act(x)
    x = self.conv_out(x)
    return x

def MyCrossAttnDownBlock2D_SD_forward(self, x, skip):
    for i in range(2):
        x = self.resnets[i](x)
        x = self.attentions[i](x)
        skip.append(x)
    if self.downsamplers is not None:
        x = self.downsamplers[0](x)
        skip.append(x)
    return x

def MyCrossAttnUpBlock2D_SD_forward(self, x, skip):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, skip.pop()], dim=1))
        x = self.attentions[i](x)
    if self.upsamplers is not None:
        x = self.upsamplers[0](x)
    return x

def MyDownBlock2D_SD_forward(self, x, skip):
    for i in range(2):
        x = self.resnets[i](x)
        skip.append(x)
    if self.downsamplers is not None:
        x = self.downsamplers[0](x)
        skip.append(x)
    return x

def MyUNetMidBlock2DCrossAttn_SD_forward(self, x):
    x = self.resnets[0](x)
    x = self.attentions[0](x)
    x = self.resnets[1](x)
    return x

def MyUpBlock2D_SD_forward(self, x, skip):
    for i in range(3):
        x = self.resnets[i](torch.cat([x, skip.pop()], dim=1))
    if self.upsamplers is not None:
        x = self.upsamplers[0](x)
    return x

def MyResnetBlock2D_SD_forward(self, x_in):
    x = self.norm1(x_in)
    x = self.nonlinearity(x)
    x = self.conv1(x)
    x = self.norm2(x)
    x = self.nonlinearity(x)
    x = self.conv2(x)
    if self.in_channels == self.out_channels:
        return x + x_in
    return x + self.conv_shortcut(x_in)
            

def MyTransformer2DModel_SD_forward(self, x_in):
    b, c, h, w = x_in.shape
    x = self.norm(x_in)
    x = x.permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
    x = self.proj_in(x)
    for block in self.transformer_blocks:
        x = x + block.attn1(block.norm1(x))
        x = x + block.ff(block.norm3(x))
    x = self.proj_out(x)
    x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return x + x_in
