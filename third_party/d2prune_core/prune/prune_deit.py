

import math
import time
import gc
import torch
import torch.nn as nn
import transformers
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from utils import timeit
from tqdm import tqdm, trange
import numpy as np
import os


from .d2prune_utils import D2SparseGPT, D2Wanda, D2ADMM
from .pruner_zero import PrunerZero
from .sparsegpt import SparseGPT
from .wanda import Wanda
from .admm_grad import AdmmGrad



class D2Prune_DeiT:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.device = args.device  # 'cpu' or 'cuda:0
        self.sparsity_ratio = args.sparsity_ratio
        self.nsamples = args.nsamples
        self.target_layer_names = args.target_layer_names  # []
        self.d2_sparsegpt = args.d2_sparsegpt
        self.d2_wanda = args.d2_wanda
        self.d2_admm = args.d2_admm
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = self.args.logger

        self.layers_activations = []
        self.save_activations = False

        self.layers_attention_score = []
        self.save_attention_score = False

    def init_model(self): # share
        self.model.eval()
        # self.use_cache = self.model.config.use_cache
        # self.model.config.use_cache = False
        self.layers = self.model.vit.encoder.layer

    def capture_attention_output(self, layer, inps):
        """
        仅对第一个样本运行一次前向传播，开启 output_attentions=True
        """
        # 取第一个样本进行可视化 (Batch size=1)
        inp = inps[0].unsqueeze(0)

        with torch.no_grad():
            # LLaMA Layer forward 返回值通常是 (hidden_states, self_attn_weights, present_key_value)
            # 必须传入 output_attentions=True
            outputs = layer(
                inp,
                # attention_mask=attention_mask,
                # position_ids=position_ids,
                # position_embeddings=position_embeddings,
                output_attentions=True
            )

            # outputs[1] 是 attention weights [batch, num_heads, seq_len, seq_len]
            attn_weights = outputs[1] # [1,32, 4096, 4096]

            # 我们只需要 CPU 上的数据，且只需要 float32 节省空间
            return attn_weights.squeeze(0).detach().cpu().numpy() # [32,4096,4096]

    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res


    def check_sparsity(self, tolerance=1e-6):
        # self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                # count += (W==0).sum().item()
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                # sub_count += (W == 0).sum().item()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        # self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count)/total_params


    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        '''
        use gpu device == embed_tokens.weight.device, if cpu, turn to gpu
        '''
        # image input-->[batch, 3, 224, 224]
        inps = train_loader
        self.bs = inps.shape[0]
        device = self.model.vit.embeddings.patch_embeddings.projection.weight.device  #
        if device.type == 'cpu':
            device = self.device
            self.model.vit.embeddings.to(device)
        else:
            device = device.index
            self.model.vit.embeddings.to(device)
        self.logger.info(f"using gpu to calibrate-->device: {device}")

        # dtype = next(iter(self.model.parameters())).dtype  # torch.float32

        # inps = torch.zeros((bs, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
        #                    device=device)
        inps.requires_grad = False
        # cache = {'i': 0}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps = inp
                raise ValueError

        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        # try:
        #     self.model(inps.to(device))
        # except ValueError:
        #     pass
        inps = self.model.vit.embeddings(inps.to(device))

        self.layers[layer_ind] = self.layers[layer_ind].module
        self.model.vit.embeddings.to("cpu")
        torch.cuda.empty_cache()
        return inps

    def forward_layer_wrapper(self, layer, inps, GPT0, GPT1):
        subset = self.find_layers(layer)
        gpts = {}
        for name in subset:
            if name not in self.target_layer_names: # update wights
                gpts[name] = GPT0(self.args, subset[name])
            else:
                gpts[name] = GPT1(self.args, subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        if self.bs > self.args.batch_size: # 4096
            tmp_res = []
            for i1 in range(0, self.bs, self.args.batch_size):
                j1 = min(i1+self.args.batch_size, self.bs)
                tmp_res.append(layer(inps[i1:j1])[0]) # 0,256
            inps = torch.cat(tmp_res, dim=0)
        else:
            inps = layer(inps)[0]
        for h in handles:
            h.remove()
        return subset, gpts

    @timeit
    def prune_layer_weight(self, subset, gpts):
        for i, name in enumerate(subset):
            if name not in self.target_layer_names: # update wights
                if self.d2_sparsegpt:
                    self.logger.info(f"pruning {name} by D2-SparseGPT: r1={self.args.r1}, r2={self.args.r2}")
                elif self.d2_admm:
                    self.logger.info(f"pruning {name} by D2_Admm")
                else:
                    self.logger.info(f"pruning {name} by SparseGPT")
            else:
                if self.d2_wanda:
                    self.logger.info(f"pruning {name} by D2-Wanda: r1={self.args.r1}, r2={self.args.r2}")
                else:
                    self.logger.info(f"pruning {name} by Wanda")
            gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
            gpts[name].free()
            torch.cuda.empty_cache()

    @timeit
    def prune_vit(self, train_loader):
        self.init_model()
        inps = self.prepare_layer_calibration(train_loader)
        for i in trange(len(self.layers), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            if f"model.layers.{i}" in self.model.hf_device_map:
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                inps = inps.to(dev)
            elif layer.attention.attention.query.weight.device.type == 'cpu':
                dev = self.device
                layer.to(dev)
                inps = inps.to(dev)
            start = time.time()
            # 1. forward layer wrapper
            # update layers
            if self.args.d2_sparsegpt:
                GPT0 = D2SparseGPT
            elif self.args.d2_admm:
                GPT0 = D2ADMM
            else:
                GPT0 = SparseGPT
            # non-update layers
            if self.args.d2_wanda:
                GPT1 = D2Wanda
            else:
                GPT1= Wanda

            # 1. forward layer wrapper
            subset, gpts= self.forward_layer_wrapper(layer, inps, GPT0, GPT1)
            # 2. pruning layer weight
            self.prune_layer_weight(subset, gpts)
            if self.save_attention_score:
                sparse_attn = self.capture_attention_output(layer, inps) # [32,128,128] # numpy
                self.layers_attention_score.append(sparse_attn)
            # 3. forward layers
            with torch.no_grad():
                if self.bs > self.args.batch_size: # 4096
                    tmp_res = []
                    for i1 in range(0, self.bs, self.args.batch_size):
                        j1 = min(i1+self.args.batch_size, self.bs)
                        tmp_res.append(layer(inps[i1:j1])[0]) # 0,256
                    inps = torch.cat(tmp_res, dim=0)
                else:
                    inps = layer(inps)[0]
            self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
            if self.save_activations:
                self.layers_activations.append((torch.norm(inps, p=2, dim=(0,1)) ** 1 /inps.shape[0]).cpu().numpy().tolist()) # 2
            del layer, subset, gpts
            gc.collect()
            torch.cuda.empty_cache()
            if self.args.free:
                self.layers[i].to("cpu")
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")
        if self.save_activations:
            self.layers_activations = np.array(self.layers_activations)
            np.save(f"{os.path.join(self.args.output_dir, 'layers-output-activations.npy')}", self.layers_activations)

        if self.save_attention_score:
            self.layers_attention_score = np.array(self.layers_attention_score) # [layer_num, head_num, seqlen, seqlen]
            np.save(f"{os.path.join(self.args.output_dir, 'layers-attention-score.npy')}", self.layers_attention_score)


class Prune_Deit:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.nsamples = args.nsamples
        self.device = args.device

        self.sparsity_ratio = args.sparsity_ratio
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = args.logger
        self.layers_activations = []
        self.save_activations = False  # test
        self.layers_attention_score = []
        self.save_attention_score = False

    def init_model(self): # share
        self.model.eval()
        # self.use_cache = self.model.config.use_cache
        # self.model.config.use_cache = False
        self.layers = self.model.vit.encoder.layer

    def capture_attention_output(self, layer, inps):
        """
        仅对第一个样本运行一次前向传播，开启 output_attentions=True
        """
        # 取第一个样本进行可视化 (Batch size=1)
        inp = inps[0].unsqueeze(0)

        with torch.no_grad():
            # LLaMA Layer forward 返回值通常是 (hidden_states, self_attn_weights, present_key_value)
            # 必须传入 output_attentions=True
            outputs = layer(
                inp,
                # attention_mask=attention_mask,
                # position_ids=position_ids,
                # position_embeddings=position_embeddings,
                output_attentions=True
            )

            # outputs[1] 是 attention weights [batch, num_heads, seq_len, seq_len]
            attn_weights = outputs[1] # [1,32, 4096, 4096]

            # 我们只需要 CPU 上的数据，且只需要 float32 节省空间
            return attn_weights.squeeze(0).detach().cpu().numpy() # [32,4096,4096]
    @classmethod
    def find_layers(cls, module, layers=[nn.Linear], name=''):
        if type(module) in layers:
            return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(cls.find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
        return res

    def check_sparsity(self, tolerance=1e-6):
        # self.model.config.use_cache = False
        count = 0
        total_params = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            subset = self.find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                # count += (W==0).sum().item()
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
                # sub_count += (W == 0).sum().item()
                sub_count += (W == 0).sum().cpu().item()
                sub_params += W.numel()
            self.logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        # self.model.config.use_cache = self.use_cache
        error = abs(float(count) / total_params - self.sparsity_ratio)
        if error <= tolerance:
            self.logger.info("Pruning correctly executed")
        else:
            self.logger.info("Pruning not performed correctly")
        return float(count)/total_params



    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        '''
        use gpu device == embed_tokens.weight.device, if cpu, turn to gpu
        '''
        # image input-->[batch, 3, 224, 224]
        inps = train_loader
        self.bs = inps.shape[0]
        device = self.model.vit.embeddings.patch_embeddings.projection.weight.device  #
        if device.type == 'cpu':
            device = self.device
            self.model.vit.embeddings.to(device)
        else:
            device = device.index
            self.model.vit.embeddings.to(device)
        self.logger.info(f"using gpu to calibrate-->device: {device}")

        # dtype = next(iter(self.model.parameters())).dtype  # torch.float32

        # inps = torch.zeros((bs, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
        #                    device=device)
        inps.requires_grad = False
        # cache = {'i': 0}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps = inp
                raise ValueError

        self.layers[layer_ind] = Catcher(self.layers[layer_ind])
        # try:
        #     self.model(inps.to(device))
        # except ValueError:
        #     pass
        inps = self.model.vit.embeddings(inps.to(device))

        self.layers[layer_ind] = self.layers[layer_ind].module
        self.model.vit.embeddings.to("cpu")
        torch.cuda.empty_cache()
        return inps

    def forward_layer_wrapper(self, layer, inps, GPT):
        subset = self.find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = GPT(self.args, subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        if self.bs > self.args.batch_size: # 4096
            tmp_res = []
            for i1 in range(0, self.bs, self.args.batch_size):
                j1 = min(i1+self.args.batch_size, self.bs)
                tmp_res.append(layer(inps[i1:j1])[0]) # 0,256
            inps = torch.cat(tmp_res, dim=0)
        else:
            inps = layer(inps)[0]
        for h in handles:
            h.remove()
        return subset, gpts

    @timeit
    def prune_layer_weight(self, subset, gpts):
        for i, name in enumerate(subset):
            if self.args.prune_method == 'sparsegpt':
                self.logger.info(f"pruning {name} by SparseGPT")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m,
                                       blocksize=128, percdamp=.01)
                gpts[name].free()

            elif self.args.prune_method == 'wanda':
                self.logger.info(f"pruning {name} by Wanda")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m)
                gpts[name].free()

            elif self.args.prune_method == 'pruner-zero':
                self.logger.info(f"pruning {name} by Pruner-Zero")
                indexed_name = f'{name}_{self.index_layer}'
                gradients = self.gradients_l2[indexed_name]
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, gradients, engine=self.engine)
                gpts[name].free()

            elif self.args.prune_method == 'admm-grad':
                self.logger.info(f"pruning {name} by ADMM-Grad")
                gpts[name].fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, percdamp=.1, iterative_prune=15, iters=20, per_out=False)
                gpts[name].free()

            else:
                raise NotImplementedError
            torch.cuda.empty_cache()

    @timeit
    def prune_vit(self, train_loader):
        self.init_model()
        inps = self.prepare_layer_calibration(train_loader)
        if self.args.prune_method == 'pruner-zero':
            self.logger.info("you must loading model gradient for pruner-zero")
            self.gradients_l2 = self.args.gradients_l2
            self.engine = self.args.engine
        for i in trange(len(self.layers), desc='Pruning Processing'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'
            if f"model.layers.{i}" in self.model.hf_device_map:
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                inps = inps.to(dev)
            elif layer.attention.attention.query.weight.device.type == 'cpu':
                dev = self.device
                layer.to(dev)
                inps = inps.to(dev)
            start = time.time()
            # 1. forward layer wrapper
            if self.args.prune_method == 'sparsegpt':
                GPT = SparseGPT
            elif self.args.prune_method == 'wanda':
                GPT = Wanda
            elif self.args.prune_method == 'pruner-zero':
                GPT = PrunerZero
            elif self.args.prune_method == 'admm-grad':
                GPT = AdmmGrad
            else:
                raise NotImplementedError
            # 1. forward layer wrapper
            subset, gpts= self.forward_layer_wrapper(layer, inps, GPT)
            # 2. pruning layer weight
            if not self.args.is_dense:
                self.prune_layer_weight(subset, gpts)
            if self.save_attention_score:
                sparse_attn = self.capture_attention_output(layer, inps) # [32,128,128] # numpy
                self.layers_attention_score.append(sparse_attn)
            # 3. forward layers
            with torch.no_grad():
                if self.bs > self.args.batch_size: # 4096
                    tmp_res = []
                    for i1 in range(0, self.bs, self.args.batch_size):
                        j1 = min(i1+self.args.batch_size, self.bs)
                        tmp_res.append(layer(inps[i1:j1])[0]) # 0,256
                    inps = torch.cat(tmp_res, dim=0)
                else:
                    inps = layer(inps)[0]
            self.logger.info(f"layer {i} finished pruning, run time:{time.time() - start}")
            if self.save_activations:
                self.layers_activations.append((torch.norm(inps, p=2, dim=(0,1)) ** 1 /inps.shape[0]).cpu().numpy().tolist()) # 2
            del layer, subset, gpts
            gc.collect()
            torch.cuda.empty_cache()
            if self.args.free:
                self.layers[i].to("cpu")
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        prune_ratio = self.check_sparsity()
        self.logger.info(f"sparsity ratio check {prune_ratio:.4f}")
        if self.save_activations:
            self.layers_activations = np.array(self.layers_activations)
            np.save(f"{os.path.join(self.args.output_dir, 'layers-output-activations.npy')}", self.layers_activations)

        if self.save_attention_score:
            self.layers_attention_score = np.array(self.layers_attention_score) # [layer_num, head_num, seqlen, seqlen]
            np.save(f"{os.path.join(self.args.output_dir, 'layers-attention-score.npy')}", self.layers_attention_score)
