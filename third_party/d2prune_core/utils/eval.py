import time
from typing import List
import os
import lm_eval
import torch
import torch.nn as nn
from lm_eval.models.huggingface import HFLM
from loguru import logger
from tqdm import tqdm
from .memory import cleanup_memory, distribute_model
from .tools import timeit
from timm.utils import accuracy

# llava
from typing import List, Dict, Any, Tuple, Optional
from argparse import Namespace

from lmms_eval import evaluator
from lmms_eval.utils import handle_non_serializable
from lmms_eval.models.simple.llava_hf import LlavaHf



def eval_ppl(args, model, test_loader: torch.Tensor, is_split=False):
    with torch.no_grad():
        if '70b' in args.model.lower() or is_split:
            ppl = eval_ppl_split(args, model, test_loader)
        else:
            ppl = eval_ppl_no_split(args, model, test_loader)
    return ppl

@timeit
def eval_ppl_no_split(args, model, test_loader):
    '''
    suitable for models <70b, cpu/single gpu/
    '''
    nlls = []
    nsamples = len(test_loader)
    logger.info(f"eval data total samples:{nsamples}")
    if model.device == 'cpu':
        model = model.to(args.device)
    start = time.time()
    for i, inputs in tqdm(enumerate(test_loader)):
        if i % 50 == 0:
            args.logger.info(f"sample {i}")
        inputs = inputs.reshape(1, model.seq_len).to(args.device) # [1,4096]
        lm_logits = model(inputs).logits # 输出形状 [1, seq_len, vocab_size]

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous() # [1,4095,vocab_size]
        shift_labels = inputs[:, 1:] # window is 1?[
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seq_len # 总损失
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len)) # e(平均交叉熵损失)
    args.logger.info(f"total eval run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()

    return ppl.item()

@timeit
def eval_ppl_split(args, model, test_loader):
    if 'opt' in args.model.lower():
        ppl = opt_eval(args, model, test_loader)
        return ppl
    elif 'llama' in args.model.lower():
        import transformers
        if transformers.__version__ >= '4.51.0':
            ppl = llama_eval_v2(args, model, test_loader)
        else:
            ppl = llama_eval(args, model, test_loader)
        return ppl
    elif 'mistral' in args.model.lower():
        ppl = mistral_eval(args, model, test_loader)
        return ppl

    elif 'qwen' in args.model.lower():
        ppl = qwen_eval(args, model, test_loader)
        return ppl
    elif 'pangu' in args.model.lower():
        ppl = pangu_eval(args, model, test_loader)
        return ppl
    else:
        raise ValueError(f'Unknown model {args.model}')

@torch.no_grad()
def opt_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers
    device = model.model.decoder.embed_tokens.weight.device
    if device.type == 'cpu':
        model.model.decoder.embed_tokens.to(args.device)
        model.model.decoder.embed_positions.to(args.device)
        model.model.decoder.final_layer_norm.to(args.device)
        device = args.device
    else:
        device = device.index
    print(f"device:{device}")
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.decoder.embed_tokens.to("cpu")
    model.model.decoder.embed_positions.to("cpu")
    model.model.decoder.final_layer_norm.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(args.device)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def llama_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def llama_eval_v2(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache["position_embeddings"] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache["position_embeddings"]
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()


@torch.no_grad()
def mistral_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache["position_embeddings"] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache["position_embeddings"]
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def pangu_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache["position_embeddings"] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache["position_embeddings"]
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def qwen_eval(args, model, test_loader):
    start = time.time()
    nsamples = len(test_loader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    device = model.model.embed_tokens.weight.device
    if device.type == 'cpu':
        device = args.device
        model.model.embed_tokens.to(args.device)
    else:
        device = device.index
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seq_len, model.config.hidden_size), dtype=dtype, device=device)
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache["position_embeddings"] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = test_loader[i].reshape(-1, model.seq_len).to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    model.model.embed_tokens.to("cpu")
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache["position_embeddings"]
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(args.device)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(args.device)
    model.lm_head = model.lm_head.to(args.device)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0) # [1,1,2048]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = test_loader[i].reshape(1, model.seq_len)[:, 1:].to(args.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seq_len
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seq_len))
    args.logger.info(f"total eval ppl run time:{(time.time() - start):.2f} seconds")
    args.logger.info(f"test ppl:{ppl.item():.3f}")
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return ppl.item()

#---------------------------------llms for zero-shot acc evaluation (lm-eval)------------------------------------
@timeit
def eval_zero_shot(args, model, tokenizer, task_list: List[str] = None, batch_size = 8):
    cleanup_memory()

    if args.distribute: # 70b
        distribute_model(model)
    else:
        model.to(args.device)

    if '66b' in args.model:
        batch_size= batch_size // 2

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size) # dtype-->model.dtype
    task_manager = lm_eval.tasks.TaskManager()
    if not task_list:
        task_list = args.tasks
    tasks = task_manager.match_tasks(task_list)
    start = time.time()
    results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size=batch_size)
    args.logger.info(f"total eval zero shot run time:{(time.time() - start):.2f} seconds")
    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in
                   results['results'].items()}
    metric_vals1 = {
        task: round(max(result.get('acc,none', 0), result.get('acc_norm,none', 0)), 4)
        for task, result in results['results'].items()
    }
    mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    mean_acc_val1 = round(sum(metric_vals1.values()) / len(metric_vals1.values()), 4)
    std_vals = {task: round(result.get('acc_norm_stderr,none', result['acc_stderr,none']), 4) for task, result in
                results['results'].items()}
    mean_std_val = round(sum(std_vals.values()) / len(std_vals.values()), 4)
    metric_vals['acc_avg'] = mean_acc_val
    results['results']['AVERAGE'] = {
        "acc,none": mean_acc_val,
        "acc_stderr,none": mean_std_val
    }
    results['results']['AVERAGE1'] = {
        "acc,none": mean_acc_val1,
        "acc_stderr,none": mean_std_val
    }
    return results


#---------------------------------deit for imagenet acc evaluation------------------------------------
@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    acc1_list = []
    acc5_list = []
    for batch in data_loader:
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output.logits, target)
        else:
            output = model(images)
            loss = criterion(output.logits, target)

        acc1, acc5 = accuracy(output.logits, target, topk=(1, 5))
        acc1_list.append(acc1.item())
        acc5_list.append(acc5.item())
    return acc1_list, acc5_list


# ------------------------------------------------llava for zero-shot acc evaluation (llms-eval)---------------------------------
'''
https://github.com/EvolvingLMMs-Lab/lmms-eval
'''
def _select_primary_metric(task_name: str, metrics: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    针对 lmms-eval 的多模态任务，从 metrics 里选出“主指标”和它的 stderr 对应字段名。
    根据你贴出来的结果结构做了针对性适配：
      - scienceqa_img: exact_match,none / exact_match_stderr,none
      - mmbench_en_dev: gpt_eval_score,none / gpt_eval_score_stderr,none
      - mmbench_en_test: test 集通常无标签，这里可能为空
      - mmvet: gpt_eval_score,none / gpt_eval_score_stderr,none
    其它任务可以按需再扩。
    """
    if task_name == "scienceqa_img":
        metric_name = "exact_match,none"
        stderr_name = "exact_match_stderr,none"
        return metric_name, stderr_name

    if task_name in ["mmbench_en_dev", "mmbench_en_test"]:
        metric_name = "gpt_eval_score,none"
        stderr_name = "gpt_eval_score_stderr,none"
        return metric_name, stderr_name

    if task_name == "mmvet":
        metric_name = "gpt_eval_score,none"
        stderr_name = "gpt_eval_score_stderr,none"
        return metric_name, stderr_name

    # 2) 对其它任务做一个兜底逻辑：找第一个看起来像 acc / accuracy 的数值型指标
    numeric_keys = [
        k for k, v in metrics.items()
        if isinstance(v, (int, float))
        and not k.endswith("_stderr,none")
        and "alias" not in k
        and "submission" not in k
    ]
    if not numeric_keys:
        return None, None

    # 优先包含 acc / accuracy 字样
    for k in numeric_keys:
        if "acc" in k.lower() or "accuracy" in k.lower():
            # 对应的 stderr 名试着加一个后缀
            stderr_name = k.replace(",none", "_stderr,none")
            return k, stderr_name if stderr_name in metrics else None

    # 否则就用第一个
    k0 = numeric_keys[0]
    stderr_name = k0.replace(",none", "_stderr,none")
    return k0, stderr_name if stderr_name in metrics else None

# large multimodal models (lmms)
@timeit
def eval_lmm_zero_shot(
    args,
    model,
    task_list: List[str] = None,
    batch_size: int = 1,
    limit: Optional[int] = None,
) -> Dict[str, Any]:

    if not isinstance(model, str):
        if args.distribute: # 70b
            distribute_model(model)
        else:
            model.to(args.device)


    # External evaluators must read credentials from the process environment.
    hf = LlavaHf(pretrained=model, batch_size=batch_size) if not isinstance(model, str) else None

    # 1. 整理任务列表
    if task_list is None:
        task_list = getattr(args, "tasks", [])
    if not task_list:
        raise ValueError("No tasks specified. Please set args.tasks or pass task_list explicitly.")

    # 2. llava_hf 限制 batch_size_per_gpu == 1，这里保护一下
    if batch_size != 1:
        args.logger.warning(f"llava_hf not surport batch_size_per_gpu={batch_size}, change to 1 forcefully")

    batch_size = 1

    cli_args = Namespace(
        output_path=os.path.join(args.output_dir, 'lmms_eval_logs'),
        process_with_media=True,
    )

    # 4. 调用 lmms-eval 进行评测
    lmms_model_name = getattr(args, "lmms_model", "llava_hf")
    model_path = getattr(args, "model_path", args.model)
    # device_map = getattr(args, "device_map", "auto")

    if hasattr(args, "logger"):
        args.logger.info(
            f"[lmms-eval] start zero-shot eval, model={lmms_model_name}, "
            f"pretrained={model_path}, tasks={task_list}, limit={limit}"
        )

    start = time.time()
    results = evaluator.simple_evaluate(
        model=hf if hf is not None else "llava_hf",
        # model=lmms_model_name,
        # model_args=f"pretrained={model},device_map={args.device}",
        tasks=task_list,
        num_fewshot=0,
        batch_size=batch_size,
        limit=limit,
        bootstrap_iters=0,
        cli_args=cli_args,
    )
    run_time = time.time() - start

    if hasattr(args, "logger"):
        args.logger.info(f"[lmms-eval] total zero-shot run time: {run_time:.2f} seconds")

    # 5. 仿照你原来的风格，抽取每个任务的“主指标”和 stderr，计算平均
    metric_vals: Dict[str, float] = {}
    metric_vals1: Dict[str, float] = {}
    std_vals: Dict[str, float] = {}

    for task_name, metrics in results.get("results", {}).items():
        if not isinstance(metrics, dict):
            continue
        # 跳过纯 group 节点，例如 mmbench_en（只有 alias）
        numeric_present = any(
            isinstance(v, (int, float)) for k, v in metrics.items()
        )
        if not numeric_present:
            continue

        metric_key, stderr_key = _select_primary_metric(task_name, metrics)
        if metric_key is None:
            continue

        val = metrics.get(metric_key, None)
        if not isinstance(val, (int, float)):
            continue

        # 主分数
        acc_val = float(val)
        metric_vals[task_name] = round(acc_val, 4)
        metric_vals1[task_name] = round(acc_val, 4)  # 这里暂时同一数值，你也可以做别的策略

        # stderr（如果有）
        if stderr_key and isinstance(metrics.get(stderr_key, None), (int, float)):
            std_vals[task_name] = round(float(metrics[stderr_key]), 4)

    # 6. 汇总平均分 / 平均 stderr，写回 results['results']['AVERAGE']
    if metric_vals:
        mean_acc_val = round(sum(metric_vals.values()) / len(metric_vals), 4)
    else:
        mean_acc_val = None

    if metric_vals1:
        mean_acc_val1 = round(sum(metric_vals1.values()) / len(metric_vals1), 4)
    else:
        mean_acc_val1 = None

    if std_vals:
        mean_std_val = round(sum(std_vals.values()) / len(std_vals), 4)
    else:
        mean_std_val = None

    if mean_acc_val is not None:
        results.setdefault("results", {})
        results["results"]["AVERAGE"] = {
            "metric": "primary_metric",
            "value": mean_acc_val,
            "stderr": mean_std_val,
        }
        results["results"]["AVERAGE1"] = {
            "metric": "primary_metric",
            "value": mean_acc_val1,
            "stderr": mean_std_val,
        }


    return results
