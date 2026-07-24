import torch.nn as nn
import torch
import torch.nn.functional as F
import gc
import time
from utils import timeit
from tqdm import tqdm, trange
import numpy as np
import math
import copy
import random
import os

from .d2prune_utils import D2SparseGPT, D2Wanda, D2ADMM
from .pruner_zero import PrunerZero
from .sparsegpt import SparseGPT
from .wanda import Wanda
from .admm_grad import AdmmGrad

from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class D2Prune_MIXTRAL:
    """
    D2Prune for Mixtral MoE models:
    1. Dynamic Sparsity Management (OWL) for per-block tuning
    2. MoE-aware ROSE pruning for expert weights
    3. Intelligent routing consistency detection and selective tuning
    """

    def __init__(self, args, model, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = args.device
        self.nsamples = args.nsamples

        self.sparsity_ratio = args.sparsity_ratio
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = args.logger

        # D2Prune specific settings
        self.d2_sparsegpt = args.d2_sparsegpt
        self.d2_wanda = args.d2_wanda
        self.d2_admm = args.d2_admm
        self.target_layer_names = getattr(args, 'target_layer_names', [])

        # Statistics
        self.layers_activations = []
        self.save_activations = False
        self.layers_attention_score = []
        self.save_attention_score = False

        # OWL-related
        self.all_layers_ratio = []
        self.all_layers_blocks_ratio = []

    def init_model(self):
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.layers = self.model.model.layers
        self.init_tensorboard()

    def init_tensorboard(self):
        self.tb_enabled = getattr(self.args, "tb_enabled", True)
        self.tb_logdir = getattr(self.args, "tb_logdir", f"{self.args.output_dir}/runs")
        self.tb_writer = getattr(self.args, "tb_writer", None)

        if self.tb_enabled and (self.tb_writer is None):
            try:
                from torch.utils.tensorboard import SummaryWriter
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                run_dir = os.path.join(self.tb_logdir, f"D2Prune_Mixtral-{ts}")
                self.tb_writer = SummaryWriter(log_dir=run_dir)
            except ImportError:
                self.tb_enabled = False

        def _tb_add(fn_name, *a, **kw):
            w = self.tb_writer
            if w is None: return
            getattr(w, fn_name)(*a, **kw)
        self._tb_add = _tb_add

    @staticmethod
    def check_sparsity(model):
        """计算MoE专家权重的稀疏度"""
        count = 0
        total_params = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "experts" in name:
                W = module.weight.data
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
        if total_params == 0:
            return 0.0
        return float(count) / total_params

    @staticmethod
    def check_outlier_mean(mask, threshold):
        """计算超过平均值threshold倍的离群值比例"""
        W = mask
        count = 0
        total_params = 0
        max_shred = torch.mean(W) * threshold
        count += (W > max_shred).sum().item()
        total_params += W.numel()
        outlier_ratio = float(count) / total_params * 100
        return outlier_ratio

    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        """获取第一层的输入和输出作为校准数据"""
        if hasattr(self.model.model, "embed_tokens"):
            device = self.model.model.embed_tokens.weight.device
        else:
            device = self.device

        if device.type == 'cpu':
            device = self.device
            if hasattr(self.model.model, "embed_tokens"):
                self.model.model.embed_tokens.to(device)

        self.logger.info(f"Calibration Capture Device: {device}")

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros(
            (self.nsamples, self.model.seq_len, self.model.config.hidden_size),
            dtype=dtype, device=device
        )
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs.get('attention_mask')
                cache['position_ids'] = kwargs.get('position_ids')
                cache["position_embeddings"] = kwargs.get('position_embeddings')
                raise ValueError

        self.layers[layer_ind] = Catcher(self.layers[layer_ind])

        for batch in train_loader:
            try:
                if isinstance(batch, (list, tuple)):
                    b = batch[0].to(device)
                else:
                    b = batch.to(device)
                self.model(b.reshape(-1, self.model.seq_len))
            except ValueError:
                pass

        self.layers[layer_ind] = self.layers[layer_ind].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        position_embeddings = cache["position_embeddings"]

        torch.cuda.empty_cache()
        return inps, outs, attention_mask, position_ids, position_embeddings

    def collect_rose_stats(self, layer, inps, attention_mask, position_ids, position_embeddings):
        """ROSE: 分块收集MoE输入统计信息"""
        self.logger.info("Collecting ROSE stats (Chunked execution)...")

        moe_input_cache = []
        def hook_fn(module, input, output):
            moe_input_cache.append(input[0].detach().cpu())

        handle = layer.block_sparse_moe.register_forward_hook(hook_fn)

        def get_batch_meta(tensor, start, end):
            if tensor is None: return None
            if tensor.shape[0] == inps.shape[0]:
                return tensor[start:end].to(self.device)
            elif tensor.shape[0] == 1:
                current_bs = end - start
                return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
            else:
                return tensor.to(self.device)

        forward_batch_size = 4
        with torch.no_grad():
            num_samples = inps.shape[0]
            for i in range(0, num_samples, forward_batch_size):
                end = min(i + forward_batch_size, num_samples)
                batch_inps = inps[i:end].to(self.device)
                batch_mask = get_batch_meta(attention_mask, i, end)
                batch_pos = get_batch_meta(position_ids, i, end)
                batch_emb = None
                if position_embeddings is not None:
                    if isinstance(position_embeddings, tuple):
                        batch_emb = tuple(x.to(self.device) for x in position_embeddings)
                    else:
                        batch_emb = position_embeddings.to(self.device)

                layer(batch_inps, attention_mask=batch_mask, position_ids=batch_pos, position_embeddings=batch_emb)
                del batch_inps, batch_mask, batch_pos, batch_emb

        handle.remove()
        torch.cuda.empty_cache()

        if len(moe_input_cache) == 0:
            return None

        full_moe_input = torch.cat(moe_input_cache, dim=0)
        del moe_input_cache

        moe_layer = layer.block_sparse_moe
        gate = moe_layer.gate
        num_experts = self.model.config.num_local_experts
        top_k = self.model.config.num_experts_per_tok

        expert_rose_data = {i: {'w1_w3_in': [], 'w2_in': []} for i in range(num_experts)}

        calc_batch_size = 16
        total_samples = full_moe_input.shape[0]

        for i in range(0, total_samples, calc_batch_size):
            end = min(i + calc_batch_size, total_samples)
            batch_x = full_moe_input[i:end].to(self.device)
            bs, seq_len, hidden_dim = batch_x.shape
            flat_inps = batch_x.view(-1, hidden_dim)

            with torch.no_grad():
                router_logits = gate(flat_inps)
                routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                routing_weights = routing_weights.to(flat_inps.dtype)

            for expert_idx in range(num_experts):
                mask = selected_experts.eq(expert_idx)
                token_rows, k_cols = torch.where(mask)

                if token_rows.numel() == 0:
                    continue

                g = routing_weights[token_rows, k_cols].unsqueeze(1)
                X_subset = flat_inps[token_rows]
                X_weighted = (X_subset * g).cpu()
                expert_rose_data[expert_idx]['w1_w3_in'].append(X_weighted)

                chunk_limit = 4096
                w2_in_chunks = []
                with torch.no_grad():
                    e = moe_layer.experts[expert_idx]
                    num_sub_tokens = X_subset.shape[0]
                    for k in range(0, num_sub_tokens, chunk_limit):
                        k_end = min(k + chunk_limit, num_sub_tokens)
                        sub_x = X_subset[k:k_end]
                        sub_g = g[k:k_end]
                        inter = torch.nn.functional.silu(e.w1(sub_x)) * e.w3(sub_x)
                        inter_weighted = inter * sub_g
                        w2_in_chunks.append(inter_weighted.cpu())

                expert_rose_data[expert_idx]['w2_in'].append(torch.cat(w2_in_chunks, dim=0))

            del batch_x, flat_inps, router_logits, routing_weights, selected_experts
            torch.cuda.empty_cache()

        self.logger.info("Merging ROSE stats...")
        for i in range(num_experts):
            if len(expert_rose_data[i]['w1_w3_in']) > 0:
                expert_rose_data[i]['w1_w3_in'] = torch.cat(expert_rose_data[i]['w1_w3_in'], dim=0)
                expert_rose_data[i]['w2_in'] = torch.cat(expert_rose_data[i]['w2_in'], dim=0)
            else:
                expert_rose_data[i]['w1_w3_in'] = None
                expert_rose_data[i]['w2_in'] = None

        return expert_rose_data

    def compute_baseline_routing(self, layer, inps, attention_mask, position_ids, position_embeddings):
        """计算基准路由：top-k专家选择"""
        self.logger.info("    Computing baseline routing (before pruning)...")

        moe_layer = layer.block_sparse_moe
        gate = moe_layer.gate
        top_k = self.model.config.num_experts_per_tok
        all_selected = []

        def get_batch_meta(tensor, start, end):
            if tensor is None: return None
            if tensor.shape[0] == inps.shape[0]:
                return tensor[start:end].to(self.device)
            elif tensor.shape[0] == 1:
                current_bs = end - start
                return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
            else:
                return tensor.to(self.device)

        forward_batch_size = 8
        with torch.no_grad():
            num_samples = inps.shape[0]
            for i in range(0, num_samples, forward_batch_size):
                end = min(i + forward_batch_size, num_samples)
                batch_inps = inps[i:end].to(self.device)
                batch_mask = get_batch_meta(attention_mask, i, end)
                batch_pos = get_batch_meta(position_ids, i, end)
                batch_emb = None
                if position_embeddings is not None:
                    if isinstance(position_embeddings, tuple):
                        batch_emb = tuple(x.to(self.device) for x in position_embeddings)
                    else:
                        batch_emb = position_embeddings.to(self.device)

                moe_inputs = []
                def hook_fn(module, input, output):
                    moe_inputs.append(input[0].detach())

                handle = layer.block_sparse_moe.register_forward_hook(hook_fn)
                _ = layer(batch_inps, attention_mask=batch_mask, position_ids=batch_pos, position_embeddings=batch_emb)[0]
                handle.remove()

                if len(moe_inputs) > 0:
                    moe_input = moe_inputs[0]
                    bs, seq_len, hidden_dim = moe_input.shape
                    flat_inp = moe_input.view(-1, hidden_dim)
                    gate_dtype = next(gate.parameters()).dtype
                    flat_inp = flat_inp.to(dtype=gate_dtype)
                    router_logits = gate(flat_inp)
                    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
                    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                    all_selected.append(selected_experts.cpu())

        if len(all_selected) > 0:
            baseline_routing = torch.cat(all_selected, dim=0)
            self.logger.info(f"    Baseline routing computed: {baseline_routing.shape[0]} tokens with top-{top_k} expert selection")
            return baseline_routing
        return None

    def compute_routing_mismatch(self, layer, inps, baseline_routing, attention_mask, position_ids, position_embeddings):
        """计算路由不匹配率（比较top-k专家集合）"""
        self.logger.info("    Computing routing mismatch (after pruning)...")

        if baseline_routing is None:
            self.logger.warning("    No baseline routing, cannot compute mismatch")
            return 1.0

        moe_layer = layer.block_sparse_moe
        gate = moe_layer.gate
        top_k = self.model.config.num_experts_per_tok
        current_selected = []

        def get_batch_meta(tensor, start, end):
            if tensor is None: return None
            if tensor.shape[0] == inps.shape[0]:
                return tensor[start:end].to(self.device)
            elif tensor.shape[0] == 1:
                current_bs = end - start
                return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
            else:
                return tensor.to(self.device)

        forward_batch_size = 8
        with torch.no_grad():
            num_samples = inps.shape[0]
            for i in range(0, num_samples, forward_batch_size):
                end = min(i + forward_batch_size, num_samples)
                batch_inps = inps[i:end].to(self.device)
                batch_mask = get_batch_meta(attention_mask, i, end)
                batch_pos = get_batch_meta(position_ids, i, end)
                batch_emb = None
                if position_embeddings is not None:
                    if isinstance(position_embeddings, tuple):
                        batch_emb = tuple(x.to(self.device) for x in position_embeddings)
                    else:
                        batch_emb = position_embeddings.to(self.device)

                moe_inputs = []
                def hook_fn(module, input, output):
                    moe_inputs.append(input[0].detach())

                handle = layer.block_sparse_moe.register_forward_hook(hook_fn)
                _ = layer(batch_inps, attention_mask=batch_mask, position_ids=batch_pos, position_embeddings=batch_emb)[0]
                handle.remove()

                if len(moe_inputs) > 0:
                    moe_input = moe_inputs[0]
                    bs, seq_len, hidden_dim = moe_input.shape
                    flat_inp = moe_input.view(-1, hidden_dim)
                    gate_dtype = next(gate.parameters()).dtype
                    flat_inp = flat_inp.to(dtype=gate_dtype)
                    router_logits = gate(flat_inp)
                    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
                    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                    current_selected.append(selected_experts.cpu())

        if len(current_selected) > 0:
            current_routing = torch.cat(current_selected, dim=0)

            # 调试：检查baseline和current的原始值
            self.logger.info(f"    [DEBUG] baseline_routing shape: {baseline_routing.shape}, sample: {baseline_routing[:5]}")
            self.logger.info(f"    [DEBUG] current_routing shape: {current_routing.shape}, sample: {current_routing[:5]}")

            # 方法1：直接比较（不排序），看是否完全相同
            direct_matches = (baseline_routing == current_routing).all(dim=1).float().sum().item()
            direct_mismatch = 1.0 - (direct_matches / baseline_routing.shape[0])
            self.logger.info(f"    [DEBUG] Direct comparison (no sort): mismatch={direct_mismatch:.4f}")

            # 方法2：排序后比较
            baseline_sorted = torch.sort(baseline_routing, dim=1)[0]
            current_sorted = torch.sort(current_routing, dim=1)[0]

            self.logger.info(f"    [DEBUG] baseline_sorted sample: {baseline_sorted[:5]}")
            self.logger.info(f"    [DEBUG] current_sorted sample: {current_sorted[:5]}")

            sorted_matches = (baseline_sorted == current_sorted).all(dim=1).float().sum().item()
            mismatch_ratio = 1.0 - (sorted_matches / baseline_routing.shape[0])

            self.logger.info(f"    Routing Mismatch: {mismatch_ratio:.4f} ({int(baseline_routing.shape[0]-sorted_matches)}/{baseline_routing.shape[0]} tokens have different top-{top_k} expert sets)")

            # 如果都是0，尝试输出更多诊断信息
            if mismatch_ratio == 0.0:
                self.logger.warning(f"    [WARN] Mismatch is 0.0! This may indicate:")
                self.logger.warning(f"           1. Expert weights changed too little to affect routing")
                self.logger.warning(f"           2. Router logits: min={router_logits.min():.4f}, max={router_logits.max():.4f}, std={router_logits.std():.4f}")
                # 检查权重是否真的变化了
                for e_idx, expert in enumerate(moe_layer.experts):
                    sparsity = (expert.w1.weight.data == 0).float().mean().item()
                    self.logger.warning(f"           Expert {e_idx} sparsity: {sparsity:.4f}")

            return mismatch_ratio
        return 1.0

    @torch.no_grad()
    def get_layer_dynamic_sparsity(self, layer, expert_rose_data, granularity='per-block'):
        """OWL: 计算per-block动态稀疏度"""
        if granularity == 'per-block':
            self.layer_outlier_ratios = []
            self.block_sizes = []
            moe_layer = layer.block_sparse_moe

            for expert_idx, expert in enumerate(moe_layer.experts):
                w1_w3_data = expert_rose_data[expert_idx]['w1_w3_in']
                if w1_w3_data is None or w1_w3_data.shape[0] == 0:
                    block_outlier_ratio = 0.0
                else:
                    # 计算权重的度量：|W|^2 * input_scale
                    W_w1 = expert.w1.weight.data
                    W_w3 = expert.w3.weight.data
                    W_metric = torch.abs(W_w1).mean() + torch.abs(W_w3).mean()
                    block_outlier_ratio = self.check_outlier_mean(W_metric.cpu().unsqueeze(0), self.args.Hyper_m)

                self.layer_outlier_ratios.append(block_outlier_ratio)
                self.block_sizes.append(expert.w1.weight.numel() + expert.w3.weight.numel())

            total_params = sum(self.block_sizes)
            block_weights = np.array(self.block_sizes) / total_params
            self.all_blocks_ratio = np.array(self.layer_outlier_ratios)

            if self.all_blocks_ratio.max() > self.all_blocks_ratio.min():
                self.all_blocks_ratio = (self.all_blocks_ratio - self.all_blocks_ratio.min()) / \
                                      (self.all_blocks_ratio.max() - self.all_blocks_ratio.min())

            target_sparsity = self.args.sparsity_ratio
            delta = (self.all_blocks_ratio - np.mean(self.all_blocks_ratio)) * self.args.Lambda * 2
            self.all_blocks_ratio = np.clip(target_sparsity + delta, 0.1, 0.95)

            current_weighted_sparsity = np.sum(self.all_blocks_ratio * block_weights)
            scale = target_sparsity / current_weighted_sparsity
            self.all_blocks_ratio = 1 - np.clip(self.all_blocks_ratio * scale, 0.1, 0.95)

            self.logger.info(f"Block sparsity: {1-self.all_blocks_ratio}, Target: {target_sparsity:.4f}, Weighted: {np.sum((1-self.all_blocks_ratio) * block_weights):.4f}")
            return self.all_blocks_ratio

        return None

    @timeit
    def prune_llm(self, train_loader):
        """主剪枝流程"""
        self.init_model()
        inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)

        def get_batch_meta(tensor, start, end):
            if tensor is None: return None
            if tensor.shape[0] == inps.shape[0]:
                return tensor[start:end].to(self.device)
            elif tensor.shape[0] == 1:
                current_bs = end - start
                return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
            else:
                return tensor.to(self.device)

        def feed_pruner_chunks(pruner, data_cpu, chunk_size=2048):
            num_tokens = data_cpu.shape[0]
            for i in range(0, num_tokens, chunk_size):
                end = min(i + chunk_size, num_tokens)
                batch_gpu = data_cpu[i:end].to(self.device)
                pruner.add_batch(batch_gpu, None)
                del batch_gpu
            torch.cuda.empty_cache()

        for i in trange(len(self.layers), desc='Pruning Mixtral Layers'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'

            if f"model.layers.{i}" in self.model.hf_device_map:
                dev = self.model.hf_device_map[f"model.layers.{i}"]
            elif hasattr(layer, "block_sparse_moe") and layer.block_sparse_moe.gate.weight.device.type == 'cpu':
                dev = self.device
                layer.to(dev)
            else:
                dev = self.device
                layer.to(dev)

            if hasattr(layer, 'block_sparse_moe'):
                self.logger.info(f"Layer {i}: Collecting ROSE Statistics...")

                rose_data = self.collect_rose_stats(layer, inps, attention_mask, position_ids, position_embeddings)

                if rose_data is not None:
                    # 计算动态稀疏度
                    if self.args.dsm == 'owl':
                        dynamic_sparsity = self.get_layer_dynamic_sparsity(layer, rose_data, 'per-block')
                    else:
                        dynamic_sparsity = [self.args.sparsity_ratio] * len(layer.block_sparse_moe.experts)

                    experts = layer.block_sparse_moe.experts
                    for e_idx in range(len(experts)):
                        expert = experts[e_idx]
                        e_data = rose_data[e_idx]

                        if e_data['w1_w3_in'] is None or e_data['w1_w3_in'].shape[0] == 0:
                            self.logger.info(f"    Expert {e_idx}: [Cold] Tokens=0 | Method: Magnitude Pruning")
                            subset = {'w1': expert.w1, 'w3': expert.w3, 'w2': expert.w2}
                            for name, module in subset.items():
                                W = module.weight.data
                                sparsity = dynamic_sparsity[e_idx] if hasattr(self, 'args') and self.args.dsm == 'owl' else self.args.sparsity_ratio
                                thresh = torch.topk(W.abs().view(-1), int(W.numel() * (1 - sparsity))).values.min()
                                module.weight.data *= W.abs().gt(thresh).float()
                            continue

                        token_count = e_data['w1_w3_in'].shape[0]
                        method_name = self.args.prune_method if hasattr(self.args, 'prune_method') else 'sparsegpt'
                        self.logger.info(f"    Expert {e_idx}: [Active] Tokens={token_count} | Method: ROSE-{method_name.upper()}")

                        subset = {'w1': expert.w1, 'w3': expert.w3, 'w2': expert.w2}

                        # Select pruner method based on config
                        gpts = {}
                        for name, module in subset.items():
                            if self.d2_wanda:
                                # D2Wanda: 混合注意力感知的剪枝（对MoE可选）
                                gpts[name] = D2Wanda(self.args, module)
                                self.logger.info(f"      {name} using D2-Wanda")
                            elif self.d2_sparsegpt:
                                gpts[name] = D2SparseGPT(self.args, module)
                                self.logger.info(f"      {name} using D2-SparseGPT")
                            elif self.d2_admm:
                                gpts[name] = D2ADMM(self.args, module)
                                self.logger.info(f"      {name} using D2-ADMM")
                            else:
                                # 标准方法选择
                                if method_name == 'sparsegpt':
                                    gpts[name] = SparseGPT(self.args, module)
                                elif method_name == 'wanda':
                                    gpts[name] = Wanda(self.args, module)
                                elif method_name == 'pruner-zero':
                                    gpts[name] = PrunerZero(self.args, module)
                                elif method_name == 'admm-grad':
                                    gpts[name] = AdmmGrad(self.args, module)
                                else:
                                    gpts[name] = SparseGPT(self.args, module)
                                self.logger.info(f"      {name} using {method_name.upper()}")

                        # Feed data to pruners
                        w13_cpu = e_data['w1_w3_in']
                        chunk_size = 2048
                        num_tokens = w13_cpu.shape[0]
                        for start in range(0, num_tokens, chunk_size):
                            end = min(start + chunk_size, num_tokens)
                            chunk_gpu = w13_cpu[start:end].to(dev)
                            gpts['w1'].add_batch(chunk_gpu, None)
                            gpts['w3'].add_batch(chunk_gpu, None)
                            del chunk_gpu

                        feed_pruner_chunks(gpts['w2'], e_data['w2_in'], chunk_size=1024)

                        # Prune
                        sparsity = dynamic_sparsity[e_idx] if self.args.dsm == 'owl' else self.args.sparsity_ratio
                        for name, pruner in gpts.items():
                            pruner.fasterprune(sparsity, self.prune_n, self.prune_m)
                            pruner.free()

                        del gpts, subset

                    del rose_data
                    torch.cuda.empty_cache()

            # Forward pass
            if inps.device.type != 'cpu':
                inps = inps.cpu()

            outs_cpu = torch.zeros_like(inps, device='cpu')

            for j in range(self.nsamples):
                with torch.no_grad():
                    inp_gpu = inps[j].unsqueeze(0).to(dev)
                    m_mask = get_batch_meta(attention_mask, j, j+1)
                    m_pos = get_batch_meta(position_ids, j, j+1)
                    m_emb = None
                    if position_embeddings is not None:
                        if isinstance(position_embeddings, tuple):
                            m_emb = tuple(x.to(dev) for x in position_embeddings)
                        else:
                            m_emb = position_embeddings.to(dev)

                    out_gpu = layer(inp_gpu, attention_mask=m_mask, position_ids=m_pos, position_embeddings=m_emb)[0]
                    outs_cpu[j] = out_gpu.cpu()
                    del inp_gpu, out_gpu

            inps, outs = outs_cpu, inps

            if self.args.free:
                layer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        self.model.config.use_cache = self.use_cache
        final_sparsity = self.check_sparsity(self.model)
        self.logger.info(f"Final MoE Expert Sparsity: {final_sparsity:.4f}")

        if self.tb_writer:
            self.tb_writer.flush()






class Prune_MIXTRAL:
    def __init__(self, args, model, tokenizer=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.nsamples = args.nsamples
        self.device = args.device

        self.sparsity_ratio = args.sparsity_ratio
        self.prune_n = args.prune_n
        self.prune_m = args.prune_m
        self.logger = args.logger

        # 统计保存
        self.layers_activations = []
        self.save_activations = False
        self.layers_attention_score = []
        self.save_attention_score = False

        # 梯度/MeZO相关
        self.gradients_l2 = getattr(args, 'gradients_l2', None)
        self.engine = getattr(args, 'engine', None)

    def init_model(self):
        self.model.eval()
        self.use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        self.layers = self.model.model.layers
        self.init_tensorboard()

    def init_tensorboard(self):
        self.tb_enabled = getattr(self.args, "tb_enabled", True)
        self.tb_logdir  = getattr(self.args, "tb_logdir", f"{self.args.output_dir}/runs")
        self.tb_writer  = getattr(self.args, "tb_writer", None)

        if self.tb_enabled and (self.tb_writer is None):
            try:
                from torch.utils.tensorboard import SummaryWriter
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                run_dir = os.path.join(self.tb_logdir, f"ROSE_Mixtral-{ts}")
                self.tb_writer = SummaryWriter(log_dir=run_dir)
            except ImportError:
                self.tb_enabled = False

        def _tb_add(fn_name, *a, **kw):
            w = self.tb_writer
            if w is None: return
            getattr(w, fn_name)(*a, **kw)
        self._tb_add = _tb_add

    @staticmethod
    def check_sparsity(model):
        count = 0
        total_params = 0
        for name, module in model.named_modules():
            # 只统计 MoE 相关的 Linear，忽略 Attention
            if isinstance(module, nn.Linear) and "experts" in name:
                W = module.weight.data
                count += (W == 0).sum().cpu().item()
                total_params += W.numel()
        if total_params == 0: return 0.0
        return float(count) / total_params

    @torch.no_grad()
    def prepare_layer_calibration(self, train_loader, layer_ind=0):
        """
        获取第一层的输入，包含 position_embeddings
        """
        # 获取设备
        if hasattr(self.model.model, "embed_tokens"):
            device = self.model.model.embed_tokens.weight.device
        else:
            device = self.device

        if device.type == 'cpu':
            device = self.device
            if hasattr(self.model.model, "embed_tokens"):
                self.model.model.embed_tokens.to(device)

        self.logger.info(f"Calibration Capture Device: {device}")

        dtype = next(iter(self.model.parameters())).dtype
        inps = torch.zeros((self.nsamples, self.model.seq_len, self.model.config.hidden_size), dtype=dtype,
                           device=device)
        inps.requires_grad = False
        cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_mask'] = kwargs.get('attention_mask')
                cache['position_ids'] = kwargs.get('position_ids')
                # 必须捕获 position_embeddings
                cache["position_embeddings"] = kwargs.get('position_embeddings')
                raise ValueError

        # 替换第一层进行捕获
        self.layers[layer_ind] = Catcher(self.layers[layer_ind])

        for batch in train_loader:
            try:
                if isinstance(batch, (list, tuple)):
                    b = batch[0].to(device)
                else:
                    b = batch.to(device)
                # 运行模型直到第一层抛出异常
                self.model(b.reshape(-1, self.model.seq_len))
            except ValueError:
                pass

        # 还原第一层
        self.layers[layer_ind] = self.layers[layer_ind].module

        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        position_embeddings = cache["position_embeddings"]

        torch.cuda.empty_cache()
        return inps, outs, attention_mask, position_ids, position_embeddings

    def collect_rose_stats(self, layer, inps, attention_mask, position_ids, position_embeddings):
            """
            ROSE 核心逻辑 (修复切片 bug 版)
            """
            self.logger.info("Collecting ROSE stats (Chunked execution)...")

            # 1. Hook 捕获输入 (CPU offload)
            moe_input_cache = []
            def hook_fn(module, input, output):
                moe_input_cache.append(input[0].detach().cpu())

            handle = layer.block_sparse_moe.register_forward_hook(hook_fn)

            # 辅助函数：处理单样本 metadata 的切片/广播
            def get_batch_meta(tensor, start, end):
                if tensor is None: return None
                # 如果 tensor 包含了所有样本 (batch=nsamples)，则切片
                if tensor.shape[0] == inps.shape[0]:
                    return tensor[start:end].to(self.device)
                # 如果 tensor 是单样本 (batch=1)，则重复以匹配当前 batch
                elif tensor.shape[0] == 1:
                    current_bs = end - start
                    # 大多数 HF 模型支持 [1, Seq] 的广播，但为了安全我们 expand
                    # 注意：对于 position_embeddings (tuple)，在后面单独处理
                    return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
                else:
                    # 其他情况直接搬运，让模型自己处理广播
                    return tensor.to(self.device)

            # 运行 Layer Forward
            forward_batch_size = 4
            with torch.no_grad():
                num_samples = inps.shape[0]
                for i in range(0, num_samples, forward_batch_size):
                    end = min(i + forward_batch_size, num_samples)

                    # Input 肯定是全量的，需要切片
                    batch_inps = inps[i:end].to(self.device)

                    # Metadata 智能切片
                    batch_mask = get_batch_meta(attention_mask, i, end)
                    batch_pos = get_batch_meta(position_ids, i, end)

                    # position_embeddings 特殊处理 (它是 tuple)
                    batch_emb = None
                    if position_embeddings is not None:
                        # position_embeddings 通常是 (cos, sin)，且来源于 Catcher，通常是 [1, Seq, Dim]
                        # 我们不需要切片，也不需要 repeat (apply_rotary_pos_emb 支持广播)
                        # 直接移到 device 即可
                        if isinstance(position_embeddings, tuple):
                            batch_emb = tuple(x.to(self.device) for x in position_embeddings)
                        else:
                            batch_emb = position_embeddings.to(self.device)

                    layer(batch_inps,
                        attention_mask=batch_mask,
                        position_ids=batch_pos,
                        position_embeddings=batch_emb)

                    del batch_inps, batch_mask, batch_pos, batch_emb

            handle.remove()
            torch.cuda.empty_cache()

            if len(moe_input_cache) == 0:
                return None

            full_moe_input = torch.cat(moe_input_cache, dim=0)
            del moe_input_cache

            # --- 分块计算 ROSE ---
            moe_layer = layer.block_sparse_moe
            gate = moe_layer.gate
            num_experts = self.model.config.num_local_experts
            top_k = self.model.config.num_experts_per_tok

            expert_rose_data = {i: {'w1_w3_in': [], 'w2_in': []} for i in range(num_experts)}

            calc_batch_size = 16
            total_samples = full_moe_input.shape[0]

            for i in range(0, total_samples, calc_batch_size):
                end = min(i + calc_batch_size, total_samples)

                # 1. 搬运一小块数据到 GPU
                batch_x = full_moe_input[i:end].to(self.device)
                bs, seq_len, hidden_dim = batch_x.shape
                flat_inps = batch_x.view(-1, hidden_dim)

                # 2. Router Forward
                with torch.no_grad():
                    router_logits = gate(flat_inps)
                    routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
                    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                    routing_weights = routing_weights.to(flat_inps.dtype)

                # 3. 分发并计算
                for expert_idx in range(num_experts):
                    mask = selected_experts.eq(expert_idx)
                    token_rows, k_cols = torch.where(mask)

                    if token_rows.numel() == 0:
                        continue

                    g = routing_weights[token_rows, k_cols].unsqueeze(1)

                    # g = g ** (0)
                    X_subset = flat_inps[token_rows]

                    # a) w1/w3 输入
                    # X_weighted = (X_subset * g).cpu()
                    X_weighted = (X_subset).cpu()
                    expert_rose_data[expert_idx]['w1_w3_in'].append(X_weighted)

                    # b) w2 输入 (Chunked)
                    chunk_limit = 4096
                    w2_in_chunks = []

                    with torch.no_grad():
                        e = moe_layer.experts[expert_idx]
                        num_sub_tokens = X_subset.shape[0]

                        for k in range(0, num_sub_tokens, chunk_limit):
                            k_end = min(k + chunk_limit, num_sub_tokens)
                            sub_x = X_subset[k:k_end]
                            sub_g = g[k:k_end]
                            sub_g = sub_g * 0.5

                            inter = torch.nn.functional.silu(e.w1(sub_x)) * e.w3(sub_x)
                            inter_weighted = inter * sub_g
                            w2_in_chunks.append(inter_weighted.cpu())

                    expert_rose_data[expert_idx]['w2_in'].append(torch.cat(w2_in_chunks, dim=0))

                del batch_x, flat_inps, router_logits, routing_weights, selected_experts
                torch.cuda.empty_cache()

            # Merge
            self.logger.info("Merging ROSE stats...")
            for i in range(num_experts):
                if len(expert_rose_data[i]['w1_w3_in']) > 0:
                    expert_rose_data[i]['w1_w3_in'] = torch.cat(expert_rose_data[i]['w1_w3_in'], dim=0)
                    expert_rose_data[i]['w2_in'] = torch.cat(expert_rose_data[i]['w2_in'], dim=0)
                else:
                    expert_rose_data[i]['w1_w3_in'] = None
                    expert_rose_data[i]['w2_in'] = None

            return expert_rose_data

    def compute_baseline_routing(self, layer, inps, attention_mask, position_ids, position_embeddings):
            """
            计算基准路由：剪枝前 gate 对所有 token 的专家选择
            返回：baseline_routing [num_tokens, top_k], 记录每个 token 的 top-k 专家选择集合
            """
            self.logger.info("    Computing baseline routing (before pruning)...")

            moe_layer = layer.block_sparse_moe
            gate = moe_layer.gate
            top_k = self.model.config.num_experts_per_tok

            all_selected = []

            def get_batch_meta(tensor, start, end):
                if tensor is None: return None
                if tensor.shape[0] == inps.shape[0]:
                    return tensor[start:end].to(self.device)
                elif tensor.shape[0] == 1:
                    current_bs = end - start
                    return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
                else:
                    return tensor.to(self.device)

            forward_batch_size = 8
            with torch.no_grad():
                num_samples = inps.shape[0]
                for i in range(0, num_samples, forward_batch_size):
                    end = min(i + forward_batch_size, num_samples)
                    batch_inps = inps[i:end].to(self.device)
                    batch_mask = get_batch_meta(attention_mask, i, end)
                    batch_pos = get_batch_meta(position_ids, i, end)
                    batch_emb = None
                    if position_embeddings is not None:
                        if isinstance(position_embeddings, tuple):
                            batch_emb = tuple(x.to(self.device) for x in position_embeddings)
                        else:
                            batch_emb = position_embeddings.to(self.device)

                    moe_inputs = []
                    def hook_fn(module, input, output):
                        moe_inputs.append(input[0].detach())

                    handle = layer.block_sparse_moe.register_forward_hook(hook_fn)
                    _ = layer(batch_inps, attention_mask=batch_mask, position_ids=batch_pos, position_embeddings=batch_emb)[0]
                    handle.remove()

                    if len(moe_inputs) > 0:
                        moe_input = moe_inputs[0]
                        bs, seq_len, hidden_dim = moe_input.shape
                        flat_inp = moe_input.view(-1, hidden_dim)
                        # Ensure dtype matches gate's weight dtype
                        gate_dtype = next(gate.parameters()).dtype
                        flat_inp = flat_inp.to(dtype=gate_dtype)
                        router_logits = gate(flat_inp)
                        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
                        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                        # selected_experts shape: [num_tokens, top_k]
                        all_selected.append(selected_experts.cpu())

            if len(all_selected) > 0:
                baseline_routing = torch.cat(all_selected, dim=0)
                self.logger.info(f"    Baseline routing computed: {baseline_routing.shape[0]} tokens with top-{top_k} expert selection")
                return baseline_routing
            return None

    def compute_routing_mismatch(self, layer, inps, baseline_routing, attention_mask, position_ids, position_embeddings):
            """
            计算剪枝后路由与基准路由的不匹配率
            比较的是 top-k 专家选择集合是否相同（集合的所有k个专家选择都要匹配）
            返回值：mismatch_ratio (0-1), 越大说明路由改变越大，越需要调整
            """
            self.logger.info("    Computing routing mismatch (after pruning)...")

            if baseline_routing is None:
                self.logger.warning("    No baseline routing, cannot compute mismatch")
                return 1.0  # 默认认为需要调整

            moe_layer = layer.block_sparse_moe
            gate = moe_layer.gate
            top_k = self.model.config.num_experts_per_tok

            current_selected = []

            def get_batch_meta(tensor, start, end):
                if tensor is None: return None
                if tensor.shape[0] == inps.shape[0]:
                    return tensor[start:end].to(self.device)
                elif tensor.shape[0] == 1:
                    current_bs = end - start
                    return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
                else:
                    return tensor.to(self.device)

            forward_batch_size = 8
            with torch.no_grad():
                num_samples = inps.shape[0]
                for i in range(0, num_samples, forward_batch_size):
                    end = min(i + forward_batch_size, num_samples)
                    batch_inps = inps[i:end].to(self.device)
                    batch_mask = get_batch_meta(attention_mask, i, end)
                    batch_pos = get_batch_meta(position_ids, i, end)
                    batch_emb = None
                    if position_embeddings is not None:
                        if isinstance(position_embeddings, tuple):
                            batch_emb = tuple(x.to(self.device) for x in position_embeddings)
                        else:
                            batch_emb = position_embeddings.to(self.device)

                    moe_inputs = []
                    def hook_fn(module, input, output):
                        moe_inputs.append(input[0].detach())

                    handle = layer.block_sparse_moe.register_forward_hook(hook_fn)
                    _ = layer(batch_inps, attention_mask=batch_mask, position_ids=batch_pos, position_embeddings=batch_emb)[0]
                    handle.remove()

                    if len(moe_inputs) > 0:
                        moe_input = moe_inputs[0]
                        bs, seq_len, hidden_dim = moe_input.shape
                        flat_inp = moe_input.view(-1, hidden_dim)
                        # Ensure dtype matches gate's weight dtype
                        gate_dtype = next(gate.parameters()).dtype
                        flat_inp = flat_inp.to(dtype=gate_dtype)
                        router_logits = gate(flat_inp)
                        routing_weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float)
                        routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                        # selected_experts shape: [num_tokens, top_k]
                        current_selected.append(selected_experts.cpu())

            if len(current_selected) > 0:
                current_routing = torch.cat(current_selected, dim=0)

                # 调试：检查baseline和current的原始值
                self.logger.info(f"    [DEBUG] baseline_routing shape: {baseline_routing.shape}, sample: {baseline_routing[:5]}")
                self.logger.info(f"    [DEBUG] current_routing shape: {current_routing.shape}, sample: {current_routing[:5]}")

                # 方法1：直接比较（不排序），看是否完全相同
                direct_matches = (baseline_routing == current_routing).all(dim=1).float().sum().item()
                direct_mismatch = 1.0 - (direct_matches / baseline_routing.shape[0])
                self.logger.info(f"    [DEBUG] Direct comparison (no sort): mismatch={direct_mismatch:.4f}")

                # 方法2：排序后比较
                baseline_sorted = torch.sort(baseline_routing, dim=1)[0]
                current_sorted = torch.sort(current_routing, dim=1)[0]

                self.logger.info(f"    [DEBUG] baseline_sorted sample: {baseline_sorted[:5]}")
                self.logger.info(f"    [DEBUG] current_sorted sample: {current_sorted[:5]}")

                sorted_matches = (baseline_sorted == current_sorted).all(dim=1).float().sum().item()
                mismatch_ratio = 1.0 - (sorted_matches / baseline_routing.shape[0])

                self.logger.info(f"    Routing Mismatch: {mismatch_ratio:.4f} ({int(baseline_routing.shape[0]-sorted_matches)}/{baseline_routing.shape[0]} tokens have different top-{top_k} expert sets)")

                # 如果都是0，尝试输出更多诊断信息
                if mismatch_ratio == 0.0:
                    self.logger.warning(f"    [WARN] Mismatch is 0.0! This may indicate:")
                    self.logger.warning(f"           1. Expert weights changed too little to affect routing")
                    self.logger.warning(f"           2. Router logits: min={router_logits.min():.4f}, max={router_logits.max():.4f}, std={router_logits.std():.4f}")
                    # 检查权重是否真的变化了
                    for e_idx, expert in enumerate(moe_layer.experts):
                        sparsity = (expert.w1.weight.data == 0).float().mean().item()
                        self.logger.warning(f"           Expert {e_idx} sparsity: {sparsity:.4f}")

                return mismatch_ratio
            return 1.0

    def refine_router_analytical(self, layer_idx, layer, inps, teacher_router_logits, attention_mask, position_ids, position_embeddings):
            """
            ROSE-Align：闭式解法拟合 Dense Teacher 路由 Logits

            核心思想：
            - teacher_router_logits 来自 dense (未剪枝) 模型的gate输出
            - student (已剪枝) 的gate权重通过ridge regression拟合teacher logits
            - 这样可以恢复routing的一致性，减少MoE专家不匹配问题

            数学模型：
            min_W ||X W^T - Z_teacher||_F^2 + λ ||W||_F^2
            其中 X: [num_tokens, hidden_dim]，Z_teacher: [num_tokens, num_experts]
            解：W^T = (X^T X + λI)^{-1} X^T Z_teacher
            """
            self.logger.info(f"  >> [Router-Align] Layer {layer_idx} Start (Fitting Dense Teacher Logits)...")

            if teacher_router_logits is None or teacher_router_logits.shape[0] == 0:
                self.logger.warning(f"  >> Layer {layer_idx}: No teacher logits, skipping router tuning")
                return

            moe_layer = layer.block_sparse_moe
            gate = moe_layer.gate
            num_experts = self.model.config.num_local_experts
            hidden_dim = self.model.config.hidden_size

            self.logger.info(f"    Teacher logits shape: {teacher_router_logits.shape}")

            # === Step 1: 收集所有token的MoE输入 ===
            moe_inputs_list = []

            def get_batch_meta(tensor, start, end):
                if tensor is None: return None
                if tensor.shape[0] == inps.shape[0]:
                    return tensor[start:end].to(self.device)
                elif tensor.shape[0] == 1:
                    current_bs = end - start
                    return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
                else:
                    return tensor.to(self.device)

            forward_batch_size = 8
            with torch.no_grad():
                num_samples = inps.shape[0]
                for i in range(0, num_samples, forward_batch_size):
                    end = min(i + forward_batch_size, num_samples)
                    batch_inps = inps[i:end].to(self.device)
                    batch_mask = get_batch_meta(attention_mask, i, end)
                    batch_pos = get_batch_meta(position_ids, i, end)
                    batch_emb = None
                    if position_embeddings is not None:
                        if isinstance(position_embeddings, tuple):
                            batch_emb = tuple(x.to(self.device) for x in position_embeddings)
                        else:
                            batch_emb = position_embeddings.to(self.device)

                    moe_inputs = []
                    def hook_fn(module, input, output):
                        moe_inputs.append(input[0].detach().cpu())

                    handle = layer.block_sparse_moe.register_forward_hook(hook_fn)
                    _ = layer(batch_inps, attention_mask=batch_mask, position_ids=batch_pos, position_embeddings=batch_emb)[0]
                    handle.remove()

                    if len(moe_inputs) > 0:
                        moe_inputs_list.append(moe_inputs[0])

            if len(moe_inputs_list) == 0:
                self.logger.warning(f"  >> Layer {layer_idx}: No MoE inputs collected, skipping")
                return

            # 合并所有MoE输入
            full_moe_input = torch.cat(moe_inputs_list, dim=0)  # [总batch数, seq_len, hidden_dim]
            # 展平为2D: [总token数, hidden_dim]
            full_moe_input = full_moe_input.reshape(-1, full_moe_input.shape[-1])
            num_total_tokens = full_moe_input.shape[0]

            self.logger.info(f"    Collected MoE inputs: {full_moe_input.shape}")

            # === Step 2: 对齐X和Z_teacher的token数量 ===
            # X: [num_collected_tokens, hidden_dim]
            # Z_teacher: [num_baseline_tokens, num_experts]
            num_teacher_tokens = teacher_router_logits.shape[0]

            if num_total_tokens != num_teacher_tokens:
                self.logger.warning(f"    Token mismatch: collected={num_total_tokens}, teacher={num_teacher_tokens}, aligning...")

                if num_total_tokens > num_teacher_tokens:
                    # 截断X
                    full_moe_input = full_moe_input[:num_teacher_tokens]
                    num_total_tokens = num_teacher_tokens
                else:
                    # 重复Z_teacher
                    repeat_times = (num_total_tokens + num_teacher_tokens - 1) // num_teacher_tokens
                    teacher_router_logits = teacher_router_logits.repeat(repeat_times, 1)[:num_total_tokens]

            self.logger.info(f"    Aligned shapes: X={full_moe_input.shape}, Z={teacher_router_logits.shape}")

            # === Step 3: 线性最小二乘求解（带NaN检查和数值稳定化） ===
            # 添加bias项：X_aug = [X, 1]
            ones = torch.ones((num_total_tokens, 1), dtype=full_moe_input.dtype)
            X_augmented = torch.cat([full_moe_input, ones], dim=1)  # [num_tokens, hidden_dim+1]

            with torch.no_grad():
                X_gpu = X_augmented.to(self.device).float()
                Z_gpu = teacher_router_logits.to(self.device).float()

                # === NaN检查 ===
                if torch.isnan(X_gpu).any() or torch.isinf(X_gpu).any():
                    self.logger.warning(f"    X contains NaN/Inf, clipping to valid range...")
                    X_gpu = torch.clamp(X_gpu, min=-1e6, max=1e6)
                    X_gpu = torch.where(torch.isnan(X_gpu) | torch.isinf(X_gpu),
                                       torch.zeros_like(X_gpu), X_gpu)

                if torch.isnan(Z_gpu).any() or torch.isinf(Z_gpu).any():
                    self.logger.warning(f"    Z contains NaN/Inf, skipping router tuning for Layer {layer_idx}")
                    return

                # === Ridge Regression: 求解 (X^T X + λI) W = X^T Z ===
                # 这比lstsq更数值稳定
                try:
                    lambda_reg = 1e-4  # 正则化系数

                    # 计算 X^T X 和 X^T Z
                    XtX = X_gpu.t() @ X_gpu
                    XtZ = X_gpu.t() @ Z_gpu

                    # 添加正则化项
                    XtX_reg = XtX + lambda_reg * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)

                    # 求解线性系统
                    solution = torch.linalg.solve(XtX_reg, XtZ)
                    self.logger.info(f"    Linear system solved via ridge regression: solution shape={solution.shape}")

                except Exception as e:
                    self.logger.error(f"    Ridge regression failed: {str(e)[:80]}, skipping router tuning")
                    return

                # 提取权重和偏差
                # solution: [hidden_dim+1, num_experts]
                W_new = solution[:-1, :].t()  # [num_experts, hidden_dim]
                b_new = solution[-1, :]        # [num_experts]

            # === Step 4: 更新gate权重 ===
            gate_dtype = gate.weight.data.dtype
            gate.weight.data = W_new.to(dtype=gate_dtype)
            if gate.bias is not None:
                gate.bias.data = b_new.to(dtype=gate_dtype)

            self.logger.info(f"    Gate weights updated via closed-form solution")
            self.logger.info(f"      weight: {gate.weight.shape}, bias: {gate.bias.shape if gate.bias is not None else 'None'}")

            # === Step 5: 验证效果 ===
            with torch.no_grad():
                test_input = full_moe_input[:min(100, num_total_tokens)].to(self.device)
                gate_dtype = next(gate.parameters()).dtype
                test_input = test_input.to(dtype=gate_dtype)

                updated_logits = gate(test_input)  # [100, num_experts]
                teacher_test = teacher_router_logits[:min(100, num_total_tokens)].to(self.device)

                # 计算MSE来衡量拟合质量
                mse = ((updated_logits - teacher_test.float()) ** 2).mean().item()

                # 验证topk是否恢复
                top_k = self.model.config.num_experts_per_tok
                updated_topk = torch.topk(updated_logits, top_k, dim=1)[1]
                teacher_topk = torch.topk(teacher_test.float(), top_k, dim=1)[1]

                match_rate = (torch.sort(updated_topk, dim=1)[0] == torch.sort(teacher_topk, dim=1)[0]).all(dim=1).float().mean().item()

                self.logger.info(f"    Verification:")
                self.logger.info(f"      Logit MSE: {mse:.6f}")
                self.logger.info(f"      Top-{top_k} Recovery Rate: {match_rate*100:.1f}%")

            self.logger.info(f"  >> Router tuning complete (Layer {layer_idx})")

    def refine_router(self, layer_idx, layer, inps, teacher_outs, attention_mask, position_ids, position_embeddings, baseline_routing=None):
            """
            Router Refinement (智能选择性调整版)
            1. 比较剪枝前后路由的不匹配情况
            2. 如果不匹配率低，说明路由保持得很好，跳过调整
            3. 如果不匹配率高，说明路由改变了，需要通过训练恢复
            """
            self.logger.info(f"  >> [Router-Tuning] Layer {layer_idx} Start...")

            # === Step 0: 检查路由是否需要调整 ===
            mismatch_ratio = self.compute_routing_mismatch(layer, inps, baseline_routing, attention_mask, position_ids, position_embeddings)
            mismatch_threshold = 0.3  # 阈值：低于此值说明路由改变不大，可以跳过调整

            if mismatch_ratio < mismatch_threshold:
                self.logger.info(f"  >> Layer {layer_idx}: Routing mismatch {mismatch_ratio:.4f} < {mismatch_threshold} | SKIPPING router tuning (路由保持得很好)")
                # return  # 跳过调整
            else:
                self.logger.info(f"  >> Layer {layer_idx}: Routing mismatch {mismatch_ratio:.4f} >= {mismatch_threshold} | PERFORMING router tuning (路由改变明显，需要调整)")

            moe_layer = layer.block_sparse_moe
            gate = moe_layer.gate

            # Freeze experts, Unfreeze gate
            gate.requires_grad_(True)
            for p in moe_layer.experts.parameters():
                p.requires_grad_(False)

            # === Step 1: 计算 teacher_outs 的统计信息用于后续归一化 ===
            teacher_outs_fp32 = teacher_outs.float()
            teacher_mean = teacher_outs_fp32.mean()
            teacher_std = teacher_outs_fp32.std()
            if teacher_std < 1e-8:
                teacher_std = 1.0
            self.logger.info(f"    Teacher outputs stats - Mean: {teacher_mean:.6f}, Std: {teacher_std:.6f}")

            # === Step 2: 选择更稳健的损失函数和优化器 ===
            # Huber Loss 对异常值更鲁棒，SmoothL1Loss 的门槛设置较合理
            loss_fn = torch.nn.SmoothL1Loss(beta=0.1, reduction='mean')

            # 使用 AdamW 但学习率更合理 (针对剪枝权重)
            optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-4, weight_decay=0.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

            batch_size = 4
            num_batches = math.ceil(inps.shape[0] / batch_size)

            def get_batch_meta(tensor, start, end):
                if tensor is None: return None
                if tensor.shape[0] == inps.shape[0]:
                    return tensor[start:end].to(self.device)
                elif tensor.shape[0] == 1:
                    current_bs = end - start
                    return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
                else:
                    return tensor.to(self.device)

            # === Step 3: 检查模型精度，若为 fp16 则后续需要转换 ===
            layer_dtype = next(layer.parameters()).dtype
            convert_to_fp32 = (layer_dtype == torch.float16)
            if convert_to_fp32:
                self.logger.info(f"    Layer dtype is {layer_dtype}, will convert to fp32 for training")
                # 临时转换模块到 fp32
                layer_original_dtype = next(layer.parameters()).dtype
                layer = layer.float()

            total_epochs = 3
            total_batches = 0
            skipped_batches = 0

            for epoch in range(total_epochs):
                epoch_loss = 0.0
                valid_batches = 0

                for b in range(num_batches):
                    start = b * batch_size
                    end = min((b+1)*batch_size, inps.shape[0])

                    bx = inps[start:end].to(self.device)
                    by = teacher_outs[start:end].to(self.device)

                    b_mask = get_batch_meta(attention_mask, start, end)
                    b_pos = get_batch_meta(position_ids, start, end)
                    b_emb = None
                    if position_embeddings is not None:
                        if isinstance(position_embeddings, tuple):
                            b_emb = tuple(x.to(self.device) for x in position_embeddings)
                        else:
                            b_emb = position_embeddings.to(self.device)

                    optimizer.zero_grad()

                    with torch.no_grad():
                        # 检查输入是否包含异常值
                        bx_check = bx.float()
                        if torch.isnan(bx_check).any() or torch.isinf(bx_check).any():
                            self.logger.warning(f"    [Warning] Batch {b} input contains NaN/Inf, skipping...")
                            skipped_batches += 1
                            total_batches += 1
                            continue

                    # === 关键修复：强制 fp32 forward pass ===
                    if convert_to_fp32:
                        bx_compute = bx.float()
                        if b_mask is not None and b_mask.dtype == torch.float16:
                            b_mask = b_mask.float()
                        if b_emb is not None and isinstance(b_emb, tuple):
                            b_emb = tuple(x.float() if x.dtype == torch.float16 else x for x in b_emb)
                        elif b_emb is not None and hasattr(b_emb, 'dtype') and b_emb.dtype == torch.float16:
                            b_emb = b_emb.float()
                    else:
                        bx_compute = bx

                    # Student Forward
                    try:
                        student_out = layer(bx_compute,
                                            attention_mask=b_mask,
                                            position_ids=b_pos,
                                            position_embeddings=b_emb)[0]
                    except Exception as e:
                        self.logger.warning(f"    [Warning] Batch {b} Forward failed: {str(e)[:80]}, skipping...")
                        skipped_batches += 1
                        total_batches += 1
                        continue

                    # === 转换为 float32 并归一化 ===
                    student_out_fp32 = student_out.float()
                    by_fp32 = by.float()

                    # 检查输出是否包含异常值
                    if torch.isnan(student_out_fp32).any() or torch.isinf(student_out_fp32).any():
                        self.logger.warning(f"    [Warning] Batch {b} Student output contains NaN/Inf, skipping...")
                        skipped_batches += 1
                        total_batches += 1
                        del student_out, student_out_fp32, by_fp32
                        continue

                    if torch.isnan(by_fp32).any() or torch.isinf(by_fp32).any():
                        self.logger.warning(f"    [Warning] Batch {b} Teacher output contains NaN/Inf, skipping...")
                        skipped_batches += 1
                        total_batches += 1
                        del student_out, student_out_fp32, by_fp32
                        continue

                    # 归一化输出以稳定损失计算
                    student_out_normalized = (student_out_fp32 - teacher_mean) / (teacher_std + 1e-8)
                    by_normalized = (by_fp32 - teacher_mean) / (teacher_std + 1e-8)

                    # 使用 SmoothL1Loss 代替 MSE，更鲁棒
                    loss = loss_fn(student_out_normalized, by_normalized)

                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"    [Warning] Batch {b} Loss is NaN/Inf ({loss:.6f}), skipping...")
                        skipped_batches += 1
                        total_batches += 1
                        del student_out, student_out_fp32, by_fp32, student_out_normalized, by_normalized
                        continue

                    loss.backward()

                    # === 梯度检查和裁剪 ===
                    grad_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)
                    if grad_norm > 10.0:
                        self.logger.warning(f"    [Warning] Batch {b} Gradient norm: {grad_norm:.4f}")

                    optimizer.step()

                    epoch_loss += loss.item()
                    valid_batches += 1
                    total_batches += 1

                    del student_out, student_out_fp32, by_fp32, student_out_normalized, by_normalized

                # 更新学习率
                scheduler.step()

                if valid_batches > 0:
                    avg_loss = epoch_loss / valid_batches
                    self.logger.info(f"     Epoch {epoch+1}/{total_epochs} | Avg Loss: {avg_loss:.6f} | Valid: {valid_batches}/{num_batches}")
                else:
                    self.logger.error(f"     Epoch {epoch+1}/{total_epochs} | All batches skipped! total={total_batches}")

            self.logger.info(f"  >> Router-Tuning Summary: Total {total_batches} batches, {total_batches - skipped_batches} used, {skipped_batches} skipped ({100*skipped_batches/max(total_batches,1):.1f}%)")

            # === 恢复原始精度 ===
            if convert_to_fp32:
                layer = layer.to(layer_original_dtype)

            gate.requires_grad_(False)
            for p in moe_layer.experts.parameters():
                p.requires_grad_(True)

    def cache_dense_teacher_router_logits(self, inps, attention_mask, position_ids, position_embeddings):
        """
        缓存 dense (未剪枝) 模型的每层 router raw logits
        用于后续 ROSE-Align 中让 pruned student 拟合 teacher logits
        """
        self.logger.info("=" * 80)
        self.logger.info("STEP 0: Caching Dense Teacher Router Logits (Pre-Pruning)")
        self.logger.info("=" * 80)

        dense_teacher_logits = {}  # {layer_idx: [num_tokens, num_experts]}

        def get_batch_meta(tensor, start, end):
            if tensor is None: return None
            if tensor.shape[0] == inps.shape[0]:
                return tensor[start:end].to(self.device)
            elif tensor.shape[0] == 1:
                current_bs = end - start
                return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
            else:
                return tensor.to(self.device)

        # 逐层缓存 dense teacher 的 router logits
        for i in trange(len(self.layers), desc='Caching Dense Teacher Logits'):
            layer = self.layers[i]

            # Device handling
            if f"model.layers.{i}" in self.model.hf_device_map:
                dev = self.model.hf_device_map[f"model.layers.{i}"]
            elif hasattr(layer, "block_sparse_moe") and layer.block_sparse_moe.gate.weight.device.type == 'cpu':
                dev = self.device
                layer.to(dev)
            else:
                dev = self.device
                layer.to(dev)

            # 只缓存 MoE 层的 gate logits
            if hasattr(layer, 'block_sparse_moe'):
                moe_layer = layer.block_sparse_moe
                gate = moe_layer.gate

                logits_list = []
                forward_batch_size = 8

                with torch.no_grad():
                    num_samples = inps.shape[0]
                    for j in range(0, num_samples, forward_batch_size):
                        end_j = min(j + forward_batch_size, num_samples)
                        batch_inps = inps[j:end_j].to(dev)
                        batch_mask = get_batch_meta(attention_mask, j, end_j)
                        batch_pos = get_batch_meta(position_ids, j, end_j)
                        batch_emb = None
                        if position_embeddings is not None:
                            if isinstance(position_embeddings, tuple):
                                batch_emb = tuple(x.to(dev) for x in position_embeddings)
                            else:
                                batch_emb = position_embeddings.to(dev)

                        # 捕获 MoE 输入
                        moe_inputs = []
                        def hook_fn(module, input, output):
                            moe_inputs.append(input[0].detach())

                        handle = layer.block_sparse_moe.register_forward_hook(hook_fn)
                        _ = layer(batch_inps, attention_mask=batch_mask, position_ids=batch_pos, position_embeddings=batch_emb)[0]
                        handle.remove()

                        # 通过 gate 获取 raw logits
                        if len(moe_inputs) > 0:
                            moe_input = moe_inputs[0]
                            bs, seq_len, hidden_dim = moe_input.shape
                            flat_inp = moe_input.view(-1, hidden_dim)
                            gate_dtype = next(gate.parameters()).dtype
                            flat_inp = flat_inp.to(dtype=gate_dtype)
                            logits = gate(flat_inp)  # [num_tokens, num_experts]
                            logits_list.append(logits.cpu())

                        del batch_inps, batch_mask, batch_pos, batch_emb

                    torch.cuda.empty_cache()

                if len(logits_list) > 0:
                    dense_teacher_logits[i] = torch.cat(logits_list, dim=0)
                    self.logger.info(f"  Layer {i}: Cached {dense_teacher_logits[i].shape[0]} tokens")

            if self.args.free:
                layer.to("cpu")
            gc.collect()

        self.logger.info(f"Dense teacher logits cached for {len(dense_teacher_logits)} layers")
        return dense_teacher_logits

    @timeit
    def prune_llm(self, train_loader):
        self.init_model()
        # 1. 准备校准数据
        inps, outs, attention_mask, position_ids, position_embeddings = self.prepare_layer_calibration(train_loader)

        # 2. 缓存 dense teacher 的 router logits (用于 ROSE-Align)
        if self.args.tune_router and getattr(self.args, 'router_method', 'analytical') == 'analytical':
            self.dense_teacher_logits = self.cache_dense_teacher_router_logits(inps, attention_mask, position_ids, position_embeddings)

        self.all_layers_ratio = []

        def get_batch_meta(tensor, start, end):
            if tensor is None: return None
            if tensor.shape[0] == inps.shape[0]:
                return tensor[start:end].to(self.device)
            elif tensor.shape[0] == 1:
                current_bs = end - start
                return tensor.repeat(current_bs, *([1]*(tensor.ndim-1))).to(self.device)
            else:
                return tensor.to(self.device)

        # 辅助函数：分块喂给 Pruner
        def feed_pruner_chunks(pruner, data_cpu, chunk_size=2048):
            num_tokens = data_cpu.shape[0]
            for i in range(0, num_tokens, chunk_size):
                end = min(i + chunk_size, num_tokens)
                batch_gpu = data_cpu[i:end].to(self.device)
                pruner.add_batch(batch_gpu, None)
                del batch_gpu
            torch.cuda.empty_cache()

        for i in trange(len(self.layers), desc='Pruning Mixtral Layers'):
            layer = self.layers[i]
            self.index_layer = f'layer_{i}'

            # Device handling
            if f"model.layers.{i}" in self.model.hf_device_map:
                dev = self.model.hf_device_map[f"model.layers.{i}"]
            elif hasattr(layer, "block_sparse_moe") and layer.block_sparse_moe.gate.weight.device.type == 'cpu':
                dev = self.device
                layer.to(dev)
            else:
                dev = self.device
                layer.to(dev)

            # === Step A: Dense Teacher Router Logits (从缓存中读取) ===
            # ROSE-Align: student 拟合 dense teacher 的 logits，不是拟合自己的 logits
            teacher_router_logits = None
            teacher_outs = None

            if getattr(self.args, 'tune_router', False) and hasattr(layer, 'block_sparse_moe'):
                router_method = getattr(self.args, 'router_method', 'analytical')

                # 从缓存的 dense teacher logits 中读取当前层的数据
                if i in self.dense_teacher_logits:
                    teacher_router_logits = self.dense_teacher_logits[i]
                    self.logger.info(f"    Using cached dense teacher logits: shape={teacher_router_logits.shape}")
                else:
                    self.logger.warning(f"    Layer {i}: Dense teacher logits not cached, router tuning may not work properly")

                # SGD 方法需要 teacher_outs，这里不再支持（改为只支持 analytical 方法拟合 dense teacher logits）
                if router_method == 'sgd':
                    self.logger.warning(f"    Layer {i}: SGD method requires teacher_outs, switching to analytical mode for ROSE-Align")
                    router_method = 'analytical'



            # === Step B: Engine Setup ===
            method_name = self.args.prune_method # 获取当前方法名用于打印
            if method_name == 'sparsegpt':
                GPT = SparseGPT
            elif method_name == 'wanda':
                GPT = Wanda
            elif method_name == 'pruner-zero':
                GPT = PrunerZero
                if i == 0:
                    self.engine = self.args.engine
                    self.gradients_l2 = self.args.gradients_l2
            elif method_name == 'admm-grad':
                GPT = AdmmGrad
            else:
                raise NotImplementedError

            # === Step C: ROSE Pruning ===
            if hasattr(layer, 'block_sparse_moe'):
                self.logger.info(f"Layer {i}: Collecting ROSE Statistics...")

                # === 关键：在剪枝前保存基准路由 ===
                if getattr(self.args, 'tune_router', False) and hasattr(layer, 'block_sparse_moe'):
                    baseline_routing = self.compute_baseline_routing(layer, inps, attention_mask, position_ids, position_embeddings)

                # collect_rose_stats 内部处理 device
                rose_data = self.collect_rose_stats(layer, inps, attention_mask, position_ids, position_embeddings)

                if rose_data is not None:
                    experts = layer.block_sparse_moe.experts
                    for e_idx in range(len(experts)):
                        expert = experts[e_idx]
                        e_data = rose_data[e_idx]

                        # Fallback Logic
                        if e_data['w1_w3_in'] is None or e_data['w1_w3_in'].shape[0] == 0:
                            # === 打印日志: Fallback ===
                            self.logger.info(f"    Expert {e_idx}: [Cold] Tokens=0 | Method: Magnitude Pruning (Fallback)")

                            subset = {'w1': expert.w1, 'w3': expert.w3, 'w2': expert.w2}
                            for name, module in subset.items():
                                W = module.weight.data
                                thresh = torch.topk(W.abs().view(-1), int(W.numel() * (1 - self.sparsity_ratio))).values.min()
                                module.weight.data *= W.abs().gt(thresh).float()
                            continue

                        # === 打印日志: ROSE ===
                        token_count = e_data['w1_w3_in'].shape[0]
                        self.logger.info(f"    Expert {e_idx}: [Active] Tokens={token_count} | Method: ROSE-{method_name.upper()}")

                        subset = {'w1': expert.w1, 'w3': expert.w3, 'w2': expert.w2}
                        gpts = {}
                        for name, module in subset.items():
                            gpts[name] = GPT(self.args, module)

                        # 1. w1 & w3 (Chunked)
                        w13_cpu = e_data['w1_w3_in']
                        chunk_size = 2048
                        num_tokens = w13_cpu.shape[0]
                        for start in range(0, num_tokens, chunk_size):
                            end = min(start + chunk_size, num_tokens)
                            chunk_gpu = w13_cpu[start:end].to(dev)
                            gpts['w1'].add_batch(chunk_gpu, None)
                            gpts['w3'].add_batch(chunk_gpu, None)
                            del chunk_gpu

                        # 2. w2 (Chunked)
                        feed_pruner_chunks(gpts['w2'], e_data['w2_in'], chunk_size=1024)

                        # Prune
                        for name, pruner in gpts.items():
                            extra_kwargs = {}
                            if method_name == 'sparsegpt':
                                extra_kwargs = {'blocksize': 128, 'percdamp': 0.01}
                            elif method_name == 'pruner-zero':
                                full_name = f"model.layers.{i}.block_sparse_moe.experts.{e_idx}.{name}"
                                grads = self.gradients_l2.get(full_name, None) if self.gradients_l2 else None
                                extra_kwargs = {'gradients': grads, 'engine': self.engine}

                            pruner.fasterprune(self.sparsity_ratio, self.prune_n, self.prune_m, **extra_kwargs)
                            pruner.free()

                        del gpts, subset

                    del rose_data
                    torch.cuda.empty_cache()

            # === Step D: Router Refinement (ROSE-Align: Fit Dense Teacher Logits) ===
            if getattr(self.args, 'tune_router', False) and hasattr(layer, 'block_sparse_moe'):
                if teacher_router_logits is not None and teacher_router_logits.shape[0] > 0:
                    self.logger.info(f"      Layer {i}: ROSE-Align - Fitting dense teacher router logits")
                    self.refine_router_analytical(i, layer, inps, teacher_router_logits, attention_mask, position_ids, position_embeddings)
                else:
                    self.logger.warning(f"      Layer {i}: Dense teacher logits missing, skipping router alignment")

            # === Step E: Forward Pass ===
            # 将 inps 确保在 CPU 循环
            if inps.device.type != 'cpu':
                inps = inps.cpu()

            outs_cpu = torch.zeros_like(inps, device='cpu')

            for j in range(self.nsamples):
                with torch.no_grad():
                    inp_gpu = inps[j].unsqueeze(0).to(dev)

                    m_mask = get_batch_meta(attention_mask, j, j+1)
                    m_pos = get_batch_meta(position_ids, j, j+1)
                    m_emb = None
                    if position_embeddings is not None:
                        if isinstance(position_embeddings, tuple):
                             m_emb = tuple(x.to(dev) for x in position_embeddings)
                        else:
                             m_emb = position_embeddings.to(dev)

                    out_gpu = layer(inp_gpu,
                                    attention_mask=m_mask,
                                    position_ids=m_pos,
                                    position_embeddings=m_emb)[0]

                    outs_cpu[j] = out_gpu.cpu()
                    del inp_gpu, out_gpu

            if self.save_activations:
                tmp_outs = outs_cpu.float()
                act_norm = (torch.norm(tmp_outs, p=2, dim=(0,1)) / tmp_outs.shape[0]).numpy().tolist()
                self.layers_activations.append(act_norm)

            inps, outs = outs_cpu, inps

            if self.args.free:
                layer.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # Finalize
        self.model.config.use_cache = self.use_cache
        final_sparsity = self.check_sparsity(self.model)
        self.logger.info(f"Final MoE Expert Sparsity: {final_sparsity:.4f}")

        if self.tb_writer:
            self.tb_writer.flush()
        if self.save_activations:
            np.save(os.path.join(self.args.output_dir, 'layers-output-activations.npy'), np.array(self.layers_activations))