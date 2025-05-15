import timm
import torch
from torch import nn as nn
from collections import OrderedDict

class ExplainRetFound(nn.Module):
    def __init__(self, model, num_classes=5, input_size=224):
        super(ExplainRetFound, self).__init__()  
        self.size = input_size//16, input_size//16

        layers_before, layers_after = self.get_layer_before_after(model)
        main_block_layer = self.from_linear_to_conv_layers(model)
        
        in_ftrs = model.head.in_features
        cls_head = nn.Conv2d(in_channels=in_ftrs, out_channels=num_classes, kernel_size=1)

        self.layers_before = nn.Sequential(OrderedDict(layers_before))
        self.main_block_layer = nn.Sequential(OrderedDict(main_block_layer)) 
        self.layers_after = nn.Sequential(OrderedDict(layers_after))
        self.head = cls_head
        self.avgpool = nn.AvgPool2d(kernel_size=self.size, stride=(1,1), padding=0) 

    def get_layer_before_after(self, model):
        layers = {}   # all layers
        for name, layer in model.named_children():
            layers[name] = layer
            
        layers_before = {}
        layer_name_before = ['patch_embed', 'pos_drop', 'patch_drop', 'norm_pre']
        for name in layer_name_before:
            layers_before[name] = layers[name]

        layers_after = {}
        layer_name_after= ['norm', 'fc_norm', 'head_drop']
        for name in layer_name_after:
            layers_after[name] = layers[name]

        return layers_before, layers_after

    def from_linear_to_conv_layers(self, model):
        blocks = model.blocks
        #print(blocks)

        for idx, (name, layer) in enumerate(blocks.named_children()):
            #print(f'idx \t {idx} \t {name} \n {layer}')
            for name_, layer_ in layer.named_children():
                if (name_ == 'attn') or (name_ == 'mlp'):
                    for name__, layer__ in layer_.named_children():
                        if isinstance(layer__, nn.Linear):
                            in_ftrs = layer__.in_features
                            out_ftrs = layer__.out_features
                            
                            # Create a 1x1 convolutional layer
                            conv_layer = nn.Conv2d(in_channels=in_ftrs, out_channels=out_ftrs, kernel_size=1)
                            weight_bias = layer__.state_dict()
                            conv_layer.load_state_dict({"weight": weight_bias["weight"].view(out_ftrs, in_ftrs, 1, 1), 
                                                        "bias": weight_bias["bias"]})

                            # dynamic assignment
                            setattr(getattr(blocks[idx], name_), name__, conv_layer)

        layers_block = {}
        for name, layer in blocks.named_children():
            layers_block[name] = layer
            
        return layers_block

    def forward(self, x):
        h, w = self.size
    
        # Initial processing
        x = self.layers_before(x)
        bs, hw, c = x.shape  # Assuming input shape is (batch_size, height * width, channels)
        
        # Loop through all 24 blocks
        for idx, block in enumerate(self.main_block_layer):
            x = block.norm1(x)  # Normalization

            # Reshape before block processing
            x = x.view(bs, h, w, c).permute(0, 3, 1, 2)  # (bs, c, h, w) 
            
            # Attention mechanism
            x = block.attn.qkv(x) 
            #x = block.attn.q_norm(x)          #  ~ Identity()
            #x = block.attn.k_norm(x)          #  ~ Identity()
            #x = block.attn.attn_drop(x)       #  ~  Dropout(p=0.0, inplace=False)

            x = x.view(bs, 1024, 3, h, w).sum(dim=2)  # Sum over QKV
            x = block.attn.proj(x)
            #x = block.attn.proj_drop(x)       # Dropout(p=0.0, inplace=False)
    
            # Skip connection & dropout
            #x = block.ls1(x)                  #  ~ Identity()
            #x = block.drop_path1(x)           #  ~ Identity()
    
            # Feedforward MLP block
            x = x.view(bs, c, -1).permute(0, 2, 1)
            x = block.norm2(x)

            x = x.view(bs, h, w, c).permute(0, 3, 1, 2)
            x = block.mlp.fc1(x)
            x = block.mlp.act(x)
            #x = block.mlp.drop1(x)            # Dropout(p=0.0, inplace=False)
            #x = block.mlp.norm(x)             #  ~ Identity() 
            x = block.mlp.fc2(x)
            #x = block.mlp.drop2(x)            # Dropout(p=0.0, inplace=False)
    
            # Skip connection & dropout
            #x = block.ls2(x)                  # ~ Identity()
            #x = block.drop_path2(x)            # ~ Identity()
            x = x.view(bs, c, -1).permute(0, 2, 1)

        x = self.layers_after(x)
        x = x.view(bs, h, w, c).permute(0, 3, 1, 2)
        activation = self.head(x)
        out = self.avgpool(activation) 
        out = out.view(out.shape[0], -1)    # (bs, n_class)
        
        return out, activation