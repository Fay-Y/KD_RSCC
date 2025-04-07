
from transformers import AutoConfig
from .transformerK.models.bert.modeling_bert import BertEncoder
import torch
from abc import abstractmethod

import math
import torch as th
import torch.nn as nn

from .img_resnet import ResNet_Encoder
def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module


def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param

def filter_params(function, module):
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param
            
def apply_wd(module):
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x
    
class AttentionLayer(nn.Module):
    def __init__(self,text_dim = 768,image_dim=1024, hidden_size=2048,feedforward_size=128, dropout=0.1,n_heads=8):
        super(AttentionLayer, self).__init__()
        #self.attention = MHA()
        self.hidden_size = hidden_size
        self.input_dim = text_dim
        self.output_dim = text_dim
        self.wq = nn.Linear(self.input_dim, hidden_size)#word transform:query
        self.wk = nn.Linear(image_dim, hidden_size)#image transform:key
        self.wv = nn.Linear(image_dim, hidden_size) #image transform:value
        self.n_heads=n_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size // n_heads]))

        self.feedforward = nn.Sequential(
            nn.Linear(self.input_dim, feedforward_size),
            nn.ReLU(),
            nn.Linear(feedforward_size, self.input_dim)
        )
        self.norm2 = nn.LayerNorm(self.input_dim)
        self.reversion = nn.Linear(hidden_size, self.input_dim)
        
    def forward(self, text, image):
        query = self.wq(text)
        key = self.wk(image) 
        value = self.wv(image)
        bs = text.shape[0]
        Q = query.view(bs, -1, self.n_heads, self.hidden_size//
                   self.n_heads).permute(0, 2, 1, 3)
        K= key.view(bs, -1, self.n_heads, self.hidden_size //
                   self.n_heads).permute(0, 2, 1, 3)
        V = value.view(bs, -1, self.n_heads, self.hidden_size //
                   self.n_heads).permute(0, 2, 1, 3)

        self.scale = self.scale.to(Q.device)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2))/self.scale
        attention = self.dropout(torch.softmax(attention, dim=-1))

        x = torch.matmul(attention, V)
        
        x = x.permute(0, 2, 1, 3).contiguous()

        output = x.view(bs, -1, self.n_heads * (self.hidden_size// self.n_heads))
        output = self.reversion(output)

        feedforward_output = self.feedforward(output)
        output = output + self.dropout(feedforward_output)
        output_emb = self.norm2(output)
        
        return output_emb
    
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
    
class TransformerNetModel2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        dropout=0,
        num_heads=1,
        num_heads_upsample=-1,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if config is None:
            # print(config_name)
            config = AutoConfig.from_pretrained("/root/Diffusion-LM/bert")
            config.hidden_dropout_prob = dropout
            # config.hidden_size = 512

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        self.dropout = dropout
        self.num_heads = num_heads

        self.img_embedding = ResNet_Encoder()



        self.word_embedding = nn.Embedding(vocab_size, self.in_channels)

        self.lm_head = nn.Linear(self.in_channels, vocab_size)
        with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        self.input_up_proj = nn.Sequential(nn.Linear(in_channels, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        if init_pretrained:
            from transformers.models.bert.modeling_bert import BertModel
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddingss
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
            print('initializing from pretrained bert.')
        else:
            # print("config:",config)
            self.input_transformers = BertEncoder(config)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, out_channels))
        self.attention = AttentionLayer()
        self.time_emb = FourierFeatures(1,model_channels)
        self.time_in_proj = nn.Linear(model_channels, model_channels, bias=False)

    def get_embeds(self, input_ids):
        # return self.word_embedding(input_ids)['last_hidden_state']
        return self.word_embedding(input_ids)
        

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)


    def forward(self, x, sigma, mapping_cond = None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        c_noise = torch.log(sigma) / 4
        # emb_t = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))
        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]

        # cond1,cond2 = self.img_process(cond1,cond2)
        # print('time emb',time_emb.size())
        emb_inputs = self.position_embeddings(position_ids) + emb_x + time_emb.unsqueeze(1).expand(-1, seq_length, -1)
        cond = mapping_cond
        emb_inputs = self.attention(emb_inputs,cond)

        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        h = h.type(x.dtype)
        # print('model output',h.size())
        return h

    
    def img_process(self,x,y):
        x = self.img_embedding(x)
        y = self.img_embedding(y)
        return x,y
    
    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1 / 3):
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)
        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups