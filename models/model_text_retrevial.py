# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
from os.path import join as pjoin
from .model_utils import *
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
from .nystrom_attention import NystromAttention
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from CONCH.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


def initialize_weights(module):
    """
    This function initializes the network.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class TextEncoder(nn.Module):
    def __init__(self, conch_model):
        super().__init__()
        self.transformer = conch_model.text.transformer
        self.positional_embedding = conch_model.text.positional_embedding
        self.ln_final = conch_model.text.ln_final
        self.text_projection = conch_model.text.text_projection
        # Get dtype from one of the model's parameters
        self.dtype = next(conch_model.parameters()).dtype

    def forward(self, prompts):
        # Rest of the code remains the same
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, classnames, conch_model, provided_learnable=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = next(conch_model.parameters()).dtype
        ctx_dim = conch_model.text.ln_final.weight.shape[0]
        
        # Get the tokenizer
        self.tokenizer = get_tokenizer()

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            # Use the correct tokenize function with both arguments
            prompt = tokenize(self.tokenizer, [ctx_init])
            with torch.no_grad():
                embedding = conch_model.text.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if False:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            
        ### 此处做了修改
        if provided_learnable is not None:
            self.ctx = provided_learnable
        else:
            self.ctx = nn.Parameter(ctx_vectors)

        # Process class names
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [name for name in classnames]
        
        # Use the correct tokenize function with both arguments
        tokenized_prompts = tokenize(self.tokenizer, prompts)
        
        with torch.no_grad():
            embedding = conch_model.text.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        # Use the tokenizer's encode method for getting lengths
        self.name_lens = [len(self.tokenizer.encode(name, 
                                                   max_length=127,
                                                   truncation=True)) 
                         for name in classnames]
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Self_Attention(nn.Module):
    def __init__(self,input_dim,dk,dv):
        #dk表示k的长度
        super().__init__()
        self.scale=dk**-0.5 #根号dk
        #从输入提取q、k、v
        self.query=nn.Linear(input_dim,dk)
        self.key=nn.Linear(input_dim,dk)
        self.value=nn.Linear(input_dim,dv)

    def forward(self,x):
        #提取出q、k、v
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        #比如 q.shape=(1,4,2)(batch,4个token，token长为2)
        attention_weights=torch.matmul(q,k.transpose(-2,-1))*self.scale#求注意力分数
        #or  attention_weights=torch.matmul(q,k.transpose(1,2))*self.scale#根据q的维度，本质是一样的
        attention_weights=nn.functional.softmax(attention_weights,dim=-1)#对注意力分数归一化，最后一个维度做
        attended_values=torch.matmul(attention_weights,v)#将归一化后的权重与value相乘
        return attended_values


class TransLayer(nn.Module):
    
    def __init__(
        self, 
        norm_layer=nn.LayerNorm, 
        dim: int = 512, 
        attn: str = 'SA',
    ) -> None:
        """
        A Transformer Layer which uses NystromAttention to approcimate the attn.
        Args:
            norm_layer (nn.Module): the type of norm layer.
            dim (int): the dimension of input feature.
            attn (str): the attn type. You can choose 'MSA' or 'Nys'.
        """
        super().__init__()
        self.norm = norm_layer(dim)
        if attn == 'Nys':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = dim//8,
                heads = 8,
                num_landmarks = dim//2,    # number of landmarks
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=0.1
            )
            
        elif attn == 'SA':
            self.attn = Self_Attention(input_dim=dim, dk=256, dv=dim)
        

    def forward(
        self, 
        x: Tensor, 
    ) -> Tensor:
        """
        Args: 
            x (Tensor): the input feature, maybe in latent space. The shape of it should
                be [B, num_instance, latent_dim]
        Returns:
            (Tensor): the output transformated by Transformer Layer. The shape of it should 
            be [num_instance, latent_dim]
        """
        x = x.unsqueeze(0)
        x = x + self.attn(self.norm(x))
        x = x.squeeze(0)

        return x


def get_prototype_text_from_chatgpt():
    """get the description of prototypes from chatgpt.

    Returns:
        List: [num_prototype]
    """
    
    knowledge_from_chatGPT = {
        "Squamous epithelium": "Flat, plate-like cells with a centrally located nucleus.",
        "Columnar epithelium": "Elongated cells with a basally located, oval-shaped nucleus.",
        "Glandular epithelium": "Cells organized in gland-like structures, secreting various substances.",
        "Adipose tissue": "Large, round cells with a thin rim of cytoplasm and a peripheral nucleus, filled with a lipid droplet.",
        "Fibrous connective tissue": "Dense arrangement of collagen fibers and fibroblast cells with elongated nuclei.",
        "Cartilage": "Chondrocytes embedded in a matrix with a basophilic appearance, arranged in lacunae.",
        "Bone tissue": "Calcified matrix with embedded osteocytes in lacunae, connected by canaliculi.",
        "Skeletal muscle": "Long, cylindrical, multinucleated cells with visible striations.",
        "Smooth muscle": "Spindle-shaped cells with a single, centrally located nucleus and no visible striations.",
        "Cardiac muscle": "Branching, striated cells with a single, centrally located nucleus and intercalated discs between cells.",
        "Neurons": "Large, star-shaped cells with a prominent, round nucleus and several processes extending from the cell body.",
        "Glial cells": "Smaller, supportive cells with a less-defined shape and a small, dark nucleus.",
        "Lymphocytes": "Small, round cells with a large, dark nucleus and a thin rim of cytoplasm.",
        "Germinal centers": "Areas of active lymphocyte proliferation and differentiation, appearing as lighter-stained regions in lymphoid tissue.",
        "Erythrocytes": "Anucleate, biconcave, disc-shaped cells.",
        "Leukocytes": "Nucleated white blood cells with various morphologies, including neutrophils, lymphocytes, and monocytes.",
        "Hepatocytes": "Large, polygonal cells with a round, centrally located nucleus and abundant cytoplasm.",
        "Sinusoids": "Vascular channels between hepatocytes, lined by endothelial cells and Kupffer cells in liver tissue.",
        "Glomeruli": "Compact, round structures composed of capillaries and specialized cells with a visible Bowman's space in kidney tissue.",
        "Tubules": "Epithelial-lined structures with various cell types, including proximal and distal tubule cells in kidney tissue.",

        "Carcinoma": "Disorganized tissue architecture, cellular atypia, and possible invasion into surrounding tissues in epithelial-derived tissues.",
        "Sarcoma": "Pleomorphic cells, high cellularity, and possible invasion into surrounding tissues in mesenchymal-derived tissues.",
        "Lymphoma": "Atypical lymphocytes, disrupted lymphoid architecture, and possible effacement of normal lymphoid structures.",
        "Leukemia": "Increased number of abnormal white blood cells in blood smears or bone marrow aspirates, with variable size and nuclear morphology.",
        "Glioma": "Atypical glial cells, increased cellularity, possible necrosis, and disruption of normal central nervous system tissue architecture.",
        "Melanoma": "Atypical melanocytes with variable size, shape, and pigmentation, cellular atypia, and invasion of surrounding tissues."
        }
    
    # the template of text
    common_templates_t = 'a photo of the {}.'
        
    text_prototype = [common_templates_t.format(tissue_type).replace(".", ", which is {}".format(tissue_description)) for tissue_type, tissue_description in knowledge_from_chatGPT.items()]

    return text_prototype


def get_description_about_class(dataset):
    if dataset == 'task_camelyon_all_binary': 
        text_path_0 = '/data2/dongjiajun/code/xjh_few_shot_WSI_final/text_prompt/camelyon_all/generated_new_0.csv'
        text_path_1 = '/data2/dongjiajun/code/xjh_few_shot_WSI_final/text_prompt/camelyon_all/generated_new_1.csv'
    elif dataset == 'task_tcga_lung_subtyping':
        text_path_0 = '/data2/dongjiajun/code/xjh_few_shot_WSI_final/text_prompt/tcga_nsclc/generated_new_0.csv'
        text_path_1 =  '/data2/dongjiajun/code/xjh_few_shot_WSI_final/text_prompt/tcga_nsclc/generated_new_1.csv'
    elif dataset == 'task_tcga_brca_subtyping':
        text_path_0 = '/data2/dongjiajun/code/xjh_few_shot_WSI_final/text_prompt/tcga_brca/generated_new_0.csv'
        text_path_1 = '/data2/dongjiajun/code/xjh_few_shot_WSI_final/text_prompt/tcga_brca/generated_new_1.csv'
        
        
        
    text_prompt_0 = list(np.array(pd.read_csv(text_path_0, header=None, index_col=False).iloc[1:, 1:]).squeeze())
    text_prompt_1 = list(np.array(pd.read_csv(text_path_1, header=None, index_col=False).iloc[1:, 1:]).squeeze())
    
    return text_prompt_0, text_prompt_1

class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(512, 512)

    def forward(self, x):
        return self.k_proj(x)


class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear =nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output):
        logits = self.linear(mh_output)  # [B, num_token, dim] -> [B, num_token, num_experts]
        # 获取前K大的值和索引，沿列。
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        # 创建一个形状和logits相同全'-inf'矩阵，即(2,4,4)
        zeros = torch.full_like(logits, float('-inf'))
        # 按照索引和值填充上述zeros矩阵
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # 对其进行softmax，未被填充的位置会为0
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class NoisyTopkRouter(nn.Module):
    def __init__(self, num_experts, top_k, n_embed=512):
        super(NoisyTopkRouter, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # add noise
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise
        
        
        ###########################################
        #prob_ = F.softmax(noisy_logits, dim=-1)  # [1, num_tokens, num_experts]
        #mean_prob_ = prob_.squeeze(0).mean(dim=0)  # [num_experts]
        ##########################################
        

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        
        
        ###########################################
        #indices_ = indices.flatten()
        # one_hot_indices = F.one_hot(indices_, num_classes=self.num_experts)  # [top_k * num_tokens, num_experts]
        # mean_indices_ = one_hot_indices.float().mean(dim=0)  # [num_experts]
        # aux_loss = self.num_experts * (mean_prob_ @ mean_indices_)
        #print(mean_prob_, mean_indices_)
        ##########################################
        
        
        
        
        return router_output, indices#, aux_loss
    
    
class SparseMoE(nn.Module):
    def __init__(self, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(num_experts, top_k)
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.top_k = top_k
        self.aux_loss = None

    def forward(self, x, is_shared=None):
        if is_shared:
            return self.experts[0](x)
        
        # 1. 输入进入router得到两个输出
        gating_output, indices = self.router(x)
        #self.aux_loss = aux_loss
        #print(self.aux_loss)
        # 2.初始化全零矩阵，后续叠加为最终结果
        final_output = torch.zeros_like(x)

        # 3.展平，即把每个batch拼接到一起，这里对输入x和router后的结果都进行了展平
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        
        expert_lst = []
        score_lst = []
        mask_lst = []

        # 以每个专家为单位进行操作，即把当前专家处理的所有token都进行加权
        for i, expert in enumerate(self.experts):
            # 4. 对当前的专家(例如专家0)来说，查看其对所有tokens中哪些在前top2
            expert_mask = (indices == i).any(dim=-1)
            # 5. 展平操作
            flat_mask = expert_mask.view(-1)
            # 如果当前专家是任意一个token的前top2
            if flat_mask.any():
                # 6. 得到该专家对哪几个token起作用后，选取token的维度表示
                expert_input = flat_x[flat_mask]
                # 7. 将token输入expert得到输出
                expert_output = expert(expert_input)
                #print(expert_output.shape)

                # 8. 计算当前专家对于有作用的token的权重分数
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                
                ##########################
                expert_lst.append(expert_output)
                score_lst.append(gating_scores)
                mask_lst.append(expert_mask)
                ##########################
                
                
                
                # 9. 将expert输出乘上权重分数
                #weighted_output = expert_output * gating_scores
                

                # 10. 循环进行做种的结果叠加
                #final_output[expert_mask] += weighted_output.squeeze(1)

        return expert_lst, score_lst, mask_lst




class Text_Agumented_Model(nn.Module):
    def __init__(
        self, 
        dataset, 
        input_dim=1024,
        act='relu', 
        num_classes=2,
        bias=False, 
        dropout=True,
        k=1, 
        region_num=8, 
        hier_region=True,
        num_experts=8, 
        num_selected=2, 
        topk_ratio=0.2, 
    ):
        super(Text_Agumented_Model, self).__init__()
        
        # initialization
        self.loss_ce = nn.CrossEntropyLoss()
        self.is_training = True
        self.k = k
        
        self.dataset = dataset

        self.text_prototype = get_prototype_text_from_chatgpt()
        self.text_prompt_0, self.text_prompt_1 = get_description_about_class(dataset)
        self.text_prompt_total = self.text_prompt_0 + self.text_prompt_1
        
        if dataset == 'task_camelyon_all_binary':
        # self.class_prompt = ['Normal lymph node, low resolution: Oval structure with smooth border. Preserved architecture shows lighter cortex peripherally, darker medulla centrally. Evenly distributed follicles in cortex. No irregular masses or distortions visible.Normal lymph node, high resolution: Uniform small lymphocytes densely packed. Well-formed follicles in cortex, lymphocyte cords in medulla. Thin-walled blood vessels throughout. No atypical cells or architectural distortions.', 
        #                      'Metastatic lymph node, low resolution: Enlarged, irregularly shaped with distorted architecture. Effaced cortex-medulla distinction. Irregular metastatic deposits visible. Thickened/breached capsule. Abnormal blood vessel distribution.Metastatic lymph node, high resolution: Large, pleomorphic cancer cells interspersed in lymphoid tissue. Cells show irregular nuclei, prominent nucleoli, abundant cytoplasm. Atypical cell arrangements. Desmoplasia, increased mitoses, potential necrosis. Abnormal blood vessels present.']
            # self.class_prompt = ['Normal lymph node, high resolution: Uniform small lymphocytes densely packed. Well-formed follicles in cortex, lymphocyte cords in medulla. Thin-walled blood vessels throughout. No atypical cells or architectural distortions.', 
            #                 'Metastatic lymph node, high resolution: Large, pleomorphic cancer cells interspersed in lymphoid tissue. Cells show irregular nuclei, prominent nucleoli, abundant cytoplasm. Atypical cell arrangements. Desmoplasia, increased mitoses, potential necrosis. Abnormal blood vessels present.']
            self.class_prompt = ['Normal lymph node', 'Metastatic lymph node']
        #self.class_prompt = ['a WSI of normal lymph node', 'a WSI of metastatic lymph node']
        
        elif dataset == 'task_tcga_lung_subtyping':
            # self.class_prompt = ['A whole slide image of lung adenocarcinoma at high resolution with visually descriptive characteristics of clear cytoplasm, round or oval nuclei, prominent nucleoli, rich vascularity, irregular blood vessels, intratumoral septa, and heterogeneity.', 
            #                      'A whole slide image of lung squamous cell carcinoma at high resolution with visually descriptive characteristics of squamous cell differentiation, round structures with eosinophilic cytoplasm, distinct cell borders and abundant cytoplasm, enlarged nuclei, irregular nuclear shape, increased chromatin density.']
            self.class_prompt = ['Lung adenocarcinoma', 'Lung squamous cell carcinoma']
            
        
        elif dataset == 'task_tcga_brca_subtyping':
            # self.class_prompt = ['Invasive ductal carcinoma in a high-resolution whole slide image typically presents as irregular, solid nests or glands of pleomorphic tumor cells with hyperchromatic, atypical nuclei and frequent mitoses, infiltrating desmoplastic stroma, often arranged in a disorganized architecture with areas of necrosis and associated peritumoral lymphocytic infiltration.', 
            #                      'Invasive lobular carcinoma in a high-resolution whole slide image is characterized by small, monomorphic tumor cells arranged in single-file strands or diffuse infiltrative patterns within the stroma, often with discohesive growth and minimal nuclear atypia.']
            self.class_prompt = ['Invasive ductal carcinoma', 'Invasive lobular carcinoma']

            
        # text modality
        conch_model_cfg = 'conch_ViT-B-16'
        conch_checkpoint_path = '/data2/dongjiajun/code/xjh_few_shot_WSI_final/CONCH/checkpoints/conch/pytorch_model.bin'
        conch_model, preprocess = create_model_from_pretrained(conch_model_cfg, conch_checkpoint_path)
        _ = conch_model.eval()
        dtype = next(conch_model.parameters()).dtype
        
        self.tokenizer = get_tokenizer()
        
        
        tokenized_prompts_0 = tokenize(self.tokenizer, self.text_prompt_0)
        tokenized_prompts_1 = tokenize(self.tokenizer, self.text_prompt_1)
        tokenized_prompt = tokenize(self.tokenizer, self.text_prompt_total)
        
        class_without_learnable = tokenize(self.tokenizer, self.class_prompt)
        
        
        
        self.text_encoder = TextEncoder(conch_model.float())
        
        self.prompt_learner = PromptLearner(self.class_prompt, conch_model.float())
        
        
        
        
        
        #with torch.no_grad():
        embedding_0 = conch_model.text.token_embedding(tokenized_prompts_0).type(dtype)
        embedding_1 = conch_model.text.token_embedding(tokenized_prompts_1).type(dtype)
        embedding_total = conch_model.text.token_embedding(tokenized_prompt).type(dtype)
        class_without_learnable = conch_model.text.token_embedding(class_without_learnable).type(dtype)
            
            
        self.text_base_0 = self.text_encoder(embedding_0).to('cuda').detach()
        self.text_base_1 = self.text_encoder(embedding_1).to('cuda').detach()
        self.text_base = self.text_encoder(embedding_total).to('cuda').detach()
        self.class_without_learnable = self.text_encoder(class_without_learnable).to('cuda').detach()
        
        
        
        # 向retrevial的文本加入可学习的vectors
        # self.embedding_0 = embedding_0.to('cuda').detach()
        # self.embedding_1 = embedding_1.to('cuda').detach()
        # self.ctx_dim = conch_model.text.ln_final.weight.shape[0]
        # self.n_ctx = 16
        # ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=dtype)
        # nn.init.normal_(ctx_vectors, std=0.02)
        # self.ctx = nn.Parameter(ctx_vectors)
        # self.n_cls = 2
        # #self.ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        # self.retrevial_learnable = False
        # if self.retrevial_learnable:
        #     self.prompt_learner = PromptLearner(self.class_prompt, conch_model.float(), provided_learnable=self.ctx)
        #####################################
        
        
        
        
        # abmil 
        self.L = 512
        self.D = 128 
        self.K = 1
        
        
        # Visual adapter setting
        self.adapter = [nn.Linear(input_dim, 512)]
        self.adapter += [nn.LayerNorm(512)]
        if act == 'relu':
            self.adapter += [nn.ReLU()]
        elif act == 'gelu':
            self.adapter += [nn.GELU()]
        elif act == 'tanh':
            self.adapter += [nn.Tanh()]
        if dropout:
            self.adapter += [nn.Dropout(0.25)]
        self.adapter = nn.Sequential(*self.adapter)
        
        
        

        # The classifier of the feature aggregated.
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
        )
        
        
        
        

       
        # attention mat
        self.feature_dim = 512
        num_heads = 8
        self.head_dim = self.feature_dim // num_heads
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.o_proj = nn.Linear(self.feature_dim, self.feature_dim)
        # self.q_proj = nn.Linear(self.feature_dim, 256)
        # self.k_proj = nn.Linear(self.feature_dim, 256)
        # self.v_proj = nn.Linear(self.feature_dim, 256)
        # self.o_proj = nn.Linear(256, 256)
        self.num_heads = num_heads
        
        
        
        # the setting of training
        self.attn_sparse = False
        self.retrieval_dist = 'cos' ###################
        self.is_cross_modality = False
        self.is_translayer = False
        self.is_learnable = False
        
        self.k_vectors = None
        self.anti_k_vectors = None
        
        
        
        ############### MoE Setting ################
        self.moe = SparseMoE(num_experts=num_experts, top_k=num_selected)
        self.is_aux_loss = False
        self.is_moe = True
        self.learnable_text = None
        #self.weight_alpha = nn.Parameter(torch.tensor(1.0))
        self.topk_ratio = topk_ratio
        self.top_k_indices = None
        ############################################
        
        
        ######### Text Adapter #########
        
        # self.text_adapter = nn.Sequential(
        #      nn.Linear(512, 512),
        #      nn.ReLU(),
        # )
        ###############################
        
        
        self.agg = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        
        self.apply(initialize_weights)

    # def prompt_learning_retrevial_text(
    #     self, 
    #     embedding_base, 
    #     indices, 
    # ):
    #     ctx = self.ctx
    #     if ctx.dim() == 2:
    #         ctx = ctx.unsqueeze(0).expand(self.k, -1, -1)
    #     embedding = embedding_base[indices]
    #     prefix = embedding[:, :1, :]
    #     suffix = embedding[:, 1 + self.n_ctx :, :]
    #     #print(prefix.shape, self.ctx.shape)
    #     prompts = torch.cat(
    #             [
    #                 prefix,
    #                 ctx,
    #                 suffix,
    #             ],
    #             dim=1,
    #     )
    #     results = self.text_encoder(prompts)
    #     return results
        
    
    def filter_patch(
        self, 
        patch_feat, 
        text_embedding, 
        alpha=1, 
    ):
        similarity = torch.mm(patch_feat, text_embedding.t())
        
        threshold = similarity.mean() + alpha * similarity.std(unbiased=False)
        
        redundant = similarity > threshold
        
        keep_patch_indices = torch.where(~redundant)[0]
        
        return patch_feat[keep_patch_indices]
    
    def aggregator_by_text(
        self,
        patch_feat, 
        text_embedding, 
        agg_type='text', 
        if_topk=False, 
        no_aggregator=False, 
        is_moe=False, 
        is_shared=False, 
    ):
        # similarity = torch.mm(patch_feat, text_embedding.t())
        # text_similarity = similarity.mean(0)
        
        # score = F.softmax(text_similarity).unsqueeze(0)
        
        # aggre_text_embedding = score @ text_embedding
        if self.is_cross_modality:
            aggre_text_embedding = self.cross_modality(patch_feat, text_embedding, information_type=agg_type)
            #print(aggre_text_embedding.shape)
            return aggre_text_embedding
        
        
        num_text = text_embedding.shape[0]
        patch_feat = patch_feat.unsqueeze(0)
        text_embedding = text_embedding.unsqueeze(0)
        
        
        
        
        
        if self.attn_sparse:
            if agg_type == 'text': 
                aggre_text_embedding, attn_weights = self.cross_attn(patch_feat, text_embedding, text_embedding, information_type='text')
            elif agg_type == 'img':
                aggre_text_embedding, attn_weights = self.cross_attn(text_embedding, patch_feat, patch_feat, information_type='img')
                
            if self.is_learnable:
                aggre_text_embedding = aggre_text_embedding[0, :]
            # else:
            #     aggre_text_embedding = aggre_text_embedding.mean(0).unsqueeze(0)
                
            return aggre_text_embedding, attn_weights
        
        
        
        
        if agg_type == 'text': 
            if self.is_learnable:
                patch_feat = torch.cat([self.learnable_token, patch_feat], dim=1)
            aggre_text_embedding = self.cross_attn(patch_feat, text_embedding, text_embedding, information_type='text', if_topk=if_topk, is_moe=is_moe, is_shared=is_shared)
        elif agg_type == 'img':
            if self.is_learnable:
                text_embedding = torch.cat([self.learnable_token, text_embedding], dim=1)
            aggre_text_embedding = self.cross_attn(text_embedding, patch_feat, patch_feat, information_type='img', if_topk=if_topk, is_moe=is_moe, is_shared=is_shared)
            
        # if self.is_learnable:
        #     aggre_text_embedding = aggre_text_embedding[0].unsqueeze(0)
        #     #print(aggre_text_embedding.shape)
        # else:
        
        if num_text > 1:      
            if no_aggregator:
                a = F.softmax(self.agg(aggre_text_embedding), dim=0)
                aggre_text_embedding = torch.sum(aggre_text_embedding * a, dim=0) 
                return aggre_text_embedding
            else:
                aggre_text_embedding = aggre_text_embedding.mean(0).unsqueeze(0)
        else:
            aggre_text_embedding = aggre_text_embedding.unsqueeze(0)
        
        
        return aggre_text_embedding
    
    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.25,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight
    
    def cross_attn_without_pooling(
        self, 
        img_feats, 
        text_feats, 
        kv_type='img',
    ):
        img_feats = img_feats.unsqueeze(0)
        text_feats = text_feats.unsqueeze(0)
        
        if kv_type == 'img':
            queries = text_feats
            keys = img_feats
            values = img_feats
        elif kv_type == 'text':
            queries = img_feats
            keys = text_feats
            values = text_feats
        
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()
        
        # Linear projections
        # [batch_size, len, dim]
        query_states = self.q_proj(queries)
        key_states = self.k_proj(keys)
        value_states = self.v_proj(values)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output, attn_weights = self.scaled_dot_product_attention(
                query_states, key_states, value_states
        )
            
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
        
        # shape: [num of patch or prototype, 512]
        attn_output = self.o_proj(attn_output).squeeze()
        
        return attn_output, attn_weights
    
        
    def cross_attn(
        self, 
        queries, 
        keys, 
        values,  
        information_type='img', 
        attn_mask=None,
        if_topk=False,
        topk=50, 
        is_random=False,  
        is_moe=False, 
        is_shared=False, 
    ):        
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()
        
        # Linear projections
        # [batch_size, len, dim]
        if is_moe:
            #query_states = self.moe(queries, is_shared=is_shared)
            final_output = torch.zeros_like(queries)
            expert_lst, score_lst, mask_lst = self.moe(queries, is_shared=is_shared)
        else:
            query_states = self.q_proj(queries)
        # if is_moe:
        #     key_states = self.moe(keys)
        # else:
        key_states = self.k_proj(keys)
        value_states = self.v_proj(values)
        
        
            
        
        # Reshape for multi-head attention
        # [batch_size, num_heads, len, head_dim]
        
        #query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        
        for query_states, score, mask in zip(expert_lst, score_lst, mask_lst):
            q_len = query_states.shape[0]
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            if if_topk:
                scale = query_states.size(-1) ** -0.5
                qk_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
                #print(qk_scores.shape)
                
                topk = int(self.topk_ratio * kv_len)
                
                
                _, topk_indices = torch.topk(qk_scores, k=topk, dim=-1)
                topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
                
                selected_k = torch.gather(
                    key_states.unsqueeze(2).expand(-1, -1, q_len, -1, -1),          # [B, H, 1, Kv, D]
                    dim=3,                            # 在 Kv 维度 gather
                    index=topk_indices_exp            # [B, H, Q, k, D]
                )
                selected_v = torch.gather(
                    value_states.unsqueeze(2).expand(-1, -1, q_len, -1, -1),        # [B, H, 1, Kv, D]
                    dim=3,
                    index=topk_indices_exp
                )
                attn_scores = torch.matmul(
                    query_states.unsqueeze(-2),       # [B, H, Q, 1, D]
                    selected_k.transpose(-2, -1)      # [B, H, Q, D, k]
                ).squeeze(-2)  # [B, H, Q, k]
            
                attn_weights = F.softmax(attn_scores, dim=-1)
            
                # 加权求和: [B, H, Q, 1, k] @ [B, H, Q, k, D] → [B, H, Q, D]
                attn_output = torch.matmul(
                    attn_weights.unsqueeze(-2),       # [B, H, Q, 1, k]
                    selected_v                        # [B, H, Q, k, D]
                ).squeeze(-2)
                
                attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
                attn_output = self.o_proj(attn_output).squeeze()
            else:  
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=attn_mask
                )
                # attn_output = self.scaled_dot_product_attention(
                #     query_states, key_states, value_states
                # )
                
                # Reshape and project output
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
                
                # shape: [num of patch or prototype, 512]
                attn_output = self.o_proj(attn_output).squeeze()
            ###############要score###############    
            weighted_output = attn_output * score
            # print(weighted_output.shape)
            ####################################
            
            ###############不要score了###############
            # weighted_output = attn_output #* score #不要score了，看看如何
            # if weighted_output.dim() == 1:
            #     weighted_output = weighted_output.unsqueeze(0)
            #########################################
            #print(weighted_output.shape)
            final_output[mask] += weighted_output.squeeze(1)
            
        #print(final_output.shape)

        
        return final_output.squeeze()
    
    
    
    def region_selection(
      self,
      img_feats,   
      select_type='mean',
    ):
        img_feats = img_feats.unsqueeze(0)    
        B, L, C = img_feats.shape
        
        
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % self.region_num
        H, W = H+_n, W+_n
        region_size = int(H // self.region_num)
        region_num = self.region_num
        
        
        
        add_length = H * W - L
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num):
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            img_feats = torch.cat([img_feats, torch.zeros((B, add_length, C), device=img_feats.device)], dim = 1)
        
        img_feats = img_feats.view(B, H, W, C)
        img_feats = img_feats.view(B, H // region_size, region_size, W // region_size, region_size, C)
        
        # region: [num_regions*B, region_size, region_size, C]
        regions = img_feats.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
        
        # [num_regions*B, region_size*region_size, C]
        regions = regions.view(-1, region_size * region_size, C)
        
        if select_type == 'max':
            # [num_regions * B, C]
            x_regions, _ = torch.max(regions, dim=1)
        elif select_type == 'mean':
            x_regions = torch.mean(regions, dim=1)
        
        return x_regions
        
    def img_adapter(
        self, 
        img_feat, 
        attn='sa', 
    ):
        img_feat = img_feat.unsqueeze(0)
        if attn == 'sa':
            img_feat = self.cross_attn(img_feat, img_feat, img_feat)
        elif attn == 'nystorm':
            img_feat = self.nystrom_attn(img_feat).squeeze(0)
        return img_feat


    def cross_modality(
        self, 
        img_feat, 
        text_feat,
        information_type='img' 
    ):
        new_img_feat = self.img_trans(img_feat)
        new_text_feat = self.text_trans(text_feat)
        
        # img to text
        scale = img_feat.size(-1) ** -0.5
            # [num_img, num_text] 
        attn_mat = torch.matmul(
            self.img_proj(img_feat), 
            self.text_proj(text_feat).transpose(0, 1)
        ) / scale
        
        if information_type == 'text':
            # [num_img, num_text]
            attn_weight = torch.softmax(attn_mat, dim=-1)
            final_feat = torch.matmul(attn_weight, new_text_feat)
        else:
            # [num_text, num_img]
            attn_weight = torch.softmax(attn_mat.transpose(0, 1), dim=-1)
            final_feat = torch.matmul(attn_weight, new_img_feat)
        
        final_feat = final_feat.mean(0).unsqueeze(0)
        return final_feat
        
    def compute_diversity_loss(
        self, 
        vectors, 
    ):
        M = vectors.shape[0]
        
        norms = F.normalize(vectors, p=2, dim=1)
        
        cosine_similarity = torch.mm(norms, norms.transpose(0, 1))
        
        total_sum = 0
        
        for i in range(M):
            for j in range(i + 1, M):
                total_sum += cosine_similarity[i, j]
        
        diversity_loss = (2 / (M * (M - 1))) * total_sum
        
        return diversity_loss
        
    
    def retrieval_k(
        self, 
        x, 
        label,  
        retrevial_num=None, 
        text_embedding=None, 
    ):
        x = x.float()
        x = self.adapter(x)
        
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts)
        
        
        
        
        
        
        #x_img = self.aggregator_by_text(x, text_features, agg_type='img')
        #text_x = self.aggregator_by_text(x, text_features, agg_type='text')
        # 此处有修改
        
        if retrevial_num is None:
            retrevial_num = self.k
        
        if text_embedding is not None:
            text_features = text_embedding
        
        text_x = self.aggregator_by_text(x, text_features, agg_type='img', if_topk=False, no_aggregator=False, is_moe=self.is_moe, is_shared=False)
        query = text_x
        
        
        # 添加adapter
        ######################
        #query = self.text_adapter(query)
        ######################
        
        
        
        if label == 0:
            vectors = self.text_base_0
            anti_vectors = self.text_base_1
            #self.learnable_text = text_features[0].unsqueeze(0)
            #embedding_base = self.embedding_0
            #anti_embedding_base = self.embedding_1
        else:
            vectors = self.text_base_1
            anti_vectors = self.text_base_0
            #self.learnable_text = text_features[1].unsqueeze(0)
            #embedding_base = self.embedding_1
            #anti_embedding_base = self.embedding_0
            
            
        if self.retrieval_dist == 'random':
            indices = torch.randperm(vectors.shape[0])[:retrevial_num]
            anti_indices = torch.randperm(anti_vectors.shape[0])[:retrevial_num]
            self.k_vectors = vectors[indices]
            self.anti_k_vectors = anti_vectors[anti_indices]
            return
            
        query = F.normalize(query, p=2, dim=1)
        norm_vectors = F.normalize(vectors, p=2, dim=1)
        norm_anti_vectors = F.normalize(anti_vectors, p=2, dim=1)
            
        if self.retrieval_dist == 'cos':
            distances = torch.mm(query, norm_vectors.transpose(0, 1))
            anti_distances = torch.mm(query, norm_anti_vectors.transpose(0, 1))
            
        elif self.retrieval_dist == 'l2':
            distances = torch.cdist(query, norm_vectors, p=2)
            anti_distances = torch.cdist(query, norm_anti_vectors, p=2)

        top_k_indices = torch.topk(distances, k=retrevial_num, dim=1, largest=False).indices
        # if self.retrevial_learnable:
        #     top_k_vectors = self.prompt_learning_retrevial_text(embedding_base, top_k_indices.squeeze(0))
        # else:
        top_k_vectors = vectors[top_k_indices.squeeze(0)]
        
        
        self.top_k_indices = top_k_indices.squeeze(0)
        
        # Random shuffle
        #################################################
        # shuffled_idx = torch.randperm(top_k_vectors.shape[0])
        # top_k_vectors = top_k_vectors[shuffled_idx]
        ################################################
        
        
        top_k_anti_indices = torch.topk(anti_distances, k=retrevial_num, dim=1, largest=True).indices
        # if self.retrevial_learnable:
        #     top_k_anti_vectors = self.prompt_learning_retrevial_text(anti_embedding_base, top_k_anti_indices.squeeze(0))
        # else:
        top_k_anti_vectors = anti_vectors[top_k_anti_indices.squeeze(0)]
        
        self.k_vectors = top_k_vectors
        self.anti_k_vectors = top_k_anti_vectors
        
    def moe_visualization(
      self,  
      cur,  
    ):
        # one IDC sample
        
        idc_sample = torch.load('/data/shihuazhan/conch_features/brca/pt_files/TCGA-A2-A259-01Z-00-DX1.7289CD72-CB74-41D4-B4AC-4EA5FDFEC666.pt').float().to('cuda')
        # one ILC sample
        ilc_sample = torch.load('/data/shihuazhan/conch_features/brca/pt_files/TCGA-D8-A27T-01Z-00-DX1.1E3A4D57-9CF2-4EBF-B74D-ADD7BD8CBFA5.pt').float().to('cuda')
        
        # one normal sample
        normal_sample = torch.load('/data2/dongjiajun/code/xjh_few_shot_WSI_final/visualization/selected_sample/camelyon_normal_01.pt').float().to('cuda')
        
        # one tumor sample
        tumor_sample = torch.load('/data2/dongjiajun/code/xjh_few_shot_WSI_final/visualization/selected_sample/camelyon_tumor_01.pt').float().to('cuda')
        
        
        # luad sample
        import random
        random.seed(2)
        indice_random = random.sample(range(0, 1000), 50)
        luad_path = '/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files'
        luad_filename = os.listdir(luad_path)
        luad_sample_filename = [os.path.join(luad_path, file) for file in luad_filename]
        selected_sample_filename = [luad_sample_filename[i] for i in indice_random]
        luad_sample = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-62-A46R-01Z-00-DX1.AD823FBA-A63F-4D36-8B84-2C995DE5FC47.pt').float().to('cuda')
        luad_sample1 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-62-A46R-01Z-00-DX1.AD823FBA-A63F-4D36-8B84-2C995DE5FC47.pt').float().to('cuda')
        luad_sample2 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-55-8096-01Z-00-DX1.c833417d-10c1-4430-a241-d6f5496e1cd9.pt').float().to('cuda')
        luad_sample3 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-49-4514-01Z-00-DX3.3b247c08-3069-4100-9df4-734895adb954.pt').float().to('cuda')
        luad_sample4 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-05-4425-01Z-00-DX1.82B093EE-49BC-4FD9-91AC-4CC89944309D.pt').float().to('cuda')
        luad_sample5 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-J2-A4AE-01Z-00-DX1.42C5DE4A-7787-4E59-8969-D12503262C96.pt').float().to('cuda')
        luad_sample6 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-55-8097-01Z-00-DX1.2f847b65-a5dc-41be-9dd0-a1e11df3cd10.pt').float().to('cuda')
        luad_sample7 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-78-7535-01Z-00-DX1.c4ca06f3-22d1-4e39-85c8-98d1fa2b0e60.pt').float().to('cuda')
        luad_sample8 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-55-A48X-01Z-00-DX1.A46C6373-8458-4D55-88C3-4C70A05F9F47.pt').float().to('cuda')
        luad_sample9 = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-64-1676-01Z-00-DX1.3B8F1FAC-0FB2-45DE-AFA1-6ED451A4A61B.pt').float().to('cuda')
        
        # lusc sample
        lusc_sample = torch.load('/data2/shihuazhan/dataset/tcga_nsclc/conch/pt_files/TCGA-37-4133-01Z-00-DX1.3dfcb0df-6f4b-4df7-9d04-085b36e03024.pt').float().to('cuda')
        
        
        text_features = self.text_encoder(self.prompt_learner())
        expert_lst = self.moe.experts
        router = self.moe.router
        output_0 = []
        output_1 = []
        text_feature_expert_output_0 = []
        text_feature_expert_output_1 = []
        for idx, expert in enumerate(expert_lst):
            output_0.append(expert(self.text_base_0))
            output_1.append(expert(self.text_base_1))
            text_feature_expert_output_0.append(expert(text_features[0]))
            text_feature_expert_output_1.append(expert(text_features[1]))
            
        
        
        
        gating_output, indices = router(text_features)
        
        
        selected_0 = []
        selected_1 = []
        num_knowledge = self.text_base_0.shape[0]
        for idx in range(num_knowledge):
            selected_0.append(router(self.text_base_0[idx]))
            selected_1.append(router(self.text_base_1[idx]))
            

        example_ls = [torch.load(i).float().to('cuda') for i in selected_sample_filename]
        final_feature_ls = []
        for i in example_ls:
            x = self.aggregator_by_text(i, text_features, agg_type='img', if_topk=False, is_moe=self.is_moe, no_aggregator=False)
            final_feature_ls.append(x)
        
        
        
        data = {
            'score_list': gating_output,
            'indices': indices, 
            'text_base_0': self.text_base_0, 
            'text_base_1': self.text_base_1,
            'text_feature': text_features, 
            'expert_output_list_0': output_0, 
            'expert_output_list_1': output_1,
            'top_k_indices': self.top_k_indices, 
            'selected_0': selected_0, 
            'selected_1': selected_1, 
            'text_0': self.text_prompt_0, 
            'text_1': self.text_prompt_1, 
            'normal_sample': normal_sample, 
            'tumor_sample': tumor_sample, 
            'trans_normal_sample': self.v_proj(self.adapter(normal_sample)),
            'trans_tumor_sample': self.v_proj(self.adapter(tumor_sample)),
            'key_normal_sample': self.k_proj(self.adapter(normal_sample)),
            'key_tumor_sample': self.k_proj(self.adapter(tumor_sample)),
            'idc_sample': idc_sample, 
            'ilc_sample': ilc_sample, 
            'trans_idc_sample': self.v_proj(self.adapter(idc_sample)),
            'trans_ilc_sample': self.v_proj(self.adapter(ilc_sample)),
            'key_idc_sample': self.k_proj(self.adapter(idc_sample)),
            'key_ilc_sample': self.k_proj(self.adapter(ilc_sample)),
            'luad_sample': luad_sample, 
            'lusc_sample': lusc_sample, 
            'trans_luad_sample': self.v_proj(self.adapter(luad_sample)),
            'trans_lusc_sample': self.v_proj(self.adapter(lusc_sample)),
            'key_luad_sample': self.k_proj(self.adapter(luad_sample)),
            'key_lusc_sample': self.k_proj(self.adapter(lusc_sample)),
            'text_feature_expert_output_0': text_feature_expert_output_0,
            'text_feature_expert_output_1': text_feature_expert_output_1,
            'final_feature_ls': final_feature_ls,
            'example_ls': example_ls,
        }
        
        torch.save(data, pjoin('/data2/dongjiajun/code/xjh_few_shot_WSI_final/visualization/new_camelyon/'+self.dataset, 'text_feature_{}.pt'.format(cur)))
        
        return 
    
    def forward(
        self, 
        x, 
        label, 
        text_embedding=None, 
    ):
        x = x.float()
        x = self.adapter(x)
        #print(x.shape)
        
        #x = self.moe(x)
        
        
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts) #self.class_without_learnable #self.text_encoder(prompts)
        
        
        ##############################
        # img_x = self.aggregator_by_text(x, text_features, agg_type='img', if_topk=False, is_moe=True, no_aggregator=False)
        # logits = self.classifier(img_x)
        # loss = self.loss_ce(logits, label)
        # Y_prob = F.softmax(logits, dim = 1)
        # Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
        
        
        # return Y_prob, Y_hat, loss
        ##############################
        
        
        
        
        if self.is_training:
            T = 1
            
            # 此处有修改
            img_x = self.aggregator_by_text(x, text_features, agg_type='img', if_topk=False, is_moe=self.is_moe, no_aggregator=False)
            img_logits = self.classifier(img_x)
            
            # 此处有修改
            extra_x = self.aggregator_by_text(x, text_embedding, agg_type='img', if_topk=True, no_aggregator=False, is_moe=self.is_moe)
            extra_logits = self.classifier(extra_x)#.unsqueeze(0)
            logits = extra_logits + img_logits #+ img_logits #(logits + extra_logits) / T #logits + torch.sigmoid(self.alpha) * text_logits
            loss = self.loss_ce(logits, label) #+ 0.001 * self.moe.aux_loss
            
            #print(self.moe.aux_loss)
        else:
            #text_features = self.moe(text_features)
            
            # 此处有修改
            extra_x = self.aggregator_by_text(x, text_features, agg_type='img', if_topk=False, is_moe=self.is_moe, no_aggregator=False)
            extra_logits = self.classifier(extra_x)#.unsqueeze(0)
            logits = extra_logits #+ img_logits #logits + extra_logits
            loss = self.loss_ce(logits, label)
        
        # if self.is_training:
        #     logits = (logits + extra_logits) / 2
        
        
        #loss = self.loss_ce(logits, label)
            
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
        
        
        return Y_prob, Y_hat, loss 
        
            
           

        
        # prompts0 = self.prompt_learner0()
        # prompts1 = self.prompt_learner1()
        
        
        if self.is_training:
            
            
            
            text_x = self.aggregator_by_text(x, text_features, agg_type='text')
            query = text_x
            if label == 0:
                vectors = self.text_base_0
                anti_vectors = self.text_base_1
            else:
                vectors = self.text_base_1
                anti_vectors = self.text_base_0
                
            query = F.normalize(query, p=2, dim=1)
            norm_vectors = F.normalize(vectors, p=2, dim=1)
            norm_anti_vectors = F.normalize(anti_vectors, p=2, dim=1)
                
            if self.retrieval_dist == 'cos':
                distances = torch.mm(query, norm_vectors.transpose(0, 1))
                anti_distances = torch.mm(query, norm_anti_vectors.transpose(0, 1))
                
            elif self.retrieval_dist == 'l2':
                distances = torch.cdist(query, norm_vectors, p=2)
                anti_distances = torch.cdist(query, norm_anti_vectors, p=2)
            
            elif self.retrieval_dist == 'patch':
                text_vectors = F.normalize(vectors, p=2, dim=1)
                img_vectors = F.normalize(origin, p=2, dim=1)
                # [num_text, num_patch]
                sim_scores = torch.mm(text_vectors, img_vectors.transpose(0, 1))
                distances = sim_scores.sum(dim=1, keepdim=True).t()
                

            top_k_indices = torch.topk(distances, k=self.k, dim=1, largest=False).indices
            top_k_vectors = vectors[top_k_indices.squeeze(0)]
            
            
            # top_k_anti_indices = torch.topk(anti_distances, k=self.k, dim=1, largest=True).indices
            # top_k_anti_vectors = anti_vectors[top_k_anti_indices.squeeze(0)]
            
            #top_k_vectors = torch.cat([self.learnable_token, top_k_vectors], dim=0)
            
            retrevial_x_no_pooling, _ = self.cross_attn_without_pooling(x, top_k_vectors, kv_type='img')
            diversity_loss = self.compute_diversity_loss(retrevial_x_no_pooling)
            extra_x = retrevial_x_no_pooling.mean(0).unsqueeze(0)
                        
            extra_logits = self.classifier(extra_x)

        # if self.is_training:    
        img_logits = self.classifier(x_img)
        total_logits = img_logits 
        # else:
        #     x_img = self.aggregator_by_text(x, torch.cat([self.learnable_token, text_features], dim=0), agg_type='img')
        #     img_logits = self.classifier(x_img)
        #     total_logits = img_logits

        
        if self.is_training:
            logits = 0.5 * total_logits  + 0.5 * extra_logits
        else:
            logits = total_logits
        
        if self.is_training:
            loss = self.loss_ce(logits, label) #+ 0.1 * diversity_loss
        else:
            loss = self.loss_ce(logits, label)
        
        
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]
        
        
        return Y_prob, Y_hat, loss 
        
        
              
        

    
    
    