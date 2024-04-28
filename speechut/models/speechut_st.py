# ----------------------------------------------------------------------------
# SpeechUT: Bridging Speech and Text with Hidden-Unit for Encoder-Decoder Based Speech-Text Pre-training (https://arxiv.org/abs/2210.03730)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechUT
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------
import logging
import contextlib
import torch

import torch.nn as nn
from argparse import Namespace
from fairseq import utils
from dataclasses import dataclass
from typing import Any
from fairseq import checkpoint_utils, tasks
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.tasks import FairseqTask
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)
from fairseq.models.hubert import HubertAsrConfig

def mix_input(audio, source, align_pad, align_lengths, phone_info):
    # token replace
    audio=audio.transpose(0,1)
    source=source.transpose(0,1)
    
    mixseq = []
    bsz, _, fdim = audio.shape
    for i in range(bsz):
        word_length = align_lengths[i].item()
        seq = torch.zeros(0, fdim).cuda().type_as(audio)
        phone_idx, _ = torch.sort(phone_info[i])
        for j in range(word_length):
            if j not in phone_idx:
                st, en = align_pad[i, j, 0:2]
                audio_seq = audio[i, st:en+1, :]
                seq = torch.cat((seq, audio_seq), dim=0)
            else:
                st, en = align_pad[i, j, 2:4]
                text_seq = source[i, st:en+1, :]
                seq = torch.cat((seq, text_seq), dim=0)
        mixseq.append(seq)
    mixseq_length = torch.LongTensor([seq.size(0) for seq in mixseq]).cuda()
    max_len = torch.max(mixseq_length).item()
    mixseq_pad = torch.zeros(bsz, max_len, fdim).cuda().type_as(audio)
    for i, seq in enumerate(mixseq):
        mixseq_pad[i, :seq.size(0)] = seq
    mixseq_encoder_padding_mask = lengths_to_padding_mask(mixseq_length)
    mixseq_pad = mixseq_pad.transpose(0, 1)
    return mixseq_pad, mixseq_encoder_padding_mask

logger = logging.getLogger(__name__)

    
def freeze_model(model, to_freeze_dict, keep_step=None):
    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            #print(name)
            pass
        else:
            param.requires_grad = False
    return model

@dataclass
class SpeechUTS2TConfig(HubertAsrConfig):
    ### the following config is only for the compatibility to fairseq speech_to_text task
    input_feat_per_channel: Any = None
    input_channels: Any = None
    speaker_to_id: Any = None
    textual_encoder_embed_dim: Any = None
    top_k: int = 0
    queue_k: int = 0
    freeze_pretrain: int = 0
    freeze_pretrain_mt: int = 9000
    pretrain: Any = None
    task_num: int = 2
    stage2: Any = None
    


@register_model("speechut_st_legacy", dataclass=SpeechUTS2TConfig)
class SpeechUTS2T(BaseFairseqModel):
    """An encoder-decoder model."""
    def __init__(self, cfg: SpeechUTS2TConfig, encoder: FairseqEncoder):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.emb=cfg.encoder_embed_dim
        self.is_text_input= False
        self.top_k=cfg.top_k
        self.queue_k=cfg.queue_k
        self.Start=False
        self.pretrain = cfg.pretrain


    def get_mixed_input(self, audio, source, align_pad, align_lengths, phone_info):
        mix_output = mix_input(audio, source, align_pad, align_lengths, phone_info)
        mixseq, mixseq_encoder_padding_mask = mix_output
        #positions = self.embed_source_positions(mixseq_encoder_padding_mask).transpose(0, 1)
        #mixseq += positions
        #if self.layernorm_embedding is not None:
        #    mixseq = self.layernorm_embedding(mixseq)
        #mixseq = self.dropout_module(mixseq)
        return mixseq, mixseq_encoder_padding_mask

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    def _build_queue(self):
        self.register_buffer("queue_audio", torch.randn( self.queue_k,self.emb,dtype=torch.float32))
        self.register_buffer("queue_text", torch.randn(self.queue_k,self.emb,dtype=torch.float32))
        self.queue_audio = nn.functional.normalize(self.queue_audio, dim=0)
        self.queue_text = nn.functional.normalize(self.queue_text, dim=0)

        self.register_buffer("queue_ptr_audio", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_text", torch.zeros(1, dtype=torch.long))
        if self.pretrain is not None:
            queue_pretrain = checkpoint_utils.load_checkpoint_to_cpu(self.pretrain)
            self.queue_audio=queue_pretrain["model"]["queue_audio"]
            self.queue_text.load_state_dict=queue_pretrain["model"]["queue_text"]
        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys[0].shape[0]
        #print(batch_size)
        ptr_audio = int(self.queue_ptr_audio)
        ptr_text  = int(self.queue_ptr_text)

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr_text+batch_size) >= self.queue_k:
            #print("error")
            self.Start=True
            var=ptr_text+batch_size-self.queue_k
            audio_clone=self.queue_audio[var:ptr_audio,:].clone()
            text_clone=self.queue_text[var:ptr_text,:].clone()
            #pop queue
            self.queue_audio[0:self.queue_k-batch_size,:] = audio_clone
            self.queue_text[0:self.queue_k-batch_size,:] = text_clone
            #add new
            self.queue_audio[self.queue_k-batch_size:,:] = keys[0]
            self.queue_text[ self.queue_k-batch_size:,:]=keys[1]
        else:
            
            self.queue_audio[ptr_audio:ptr_audio + batch_size,:] = keys[0]
            self.queue_text[ ptr_text:ptr_text+batch_size,:]=keys[1]
        ptr_audio = (ptr_audio + batch_size) % self.queue_k  # move pointer
        ptr_text = (ptr_text + batch_size) % self.queue_k  # move pointer

        self.queue_ptr_audio[0] = ptr_audio
        self.queue_ptr_text[0]=ptr_text

    
    @classmethod
    def build_model(cls, cfg: SpeechUTS2TConfig, task: FairseqTask):
        """Build a new model instance."""
        encoder = SpeechUTEncoder(cfg, task)
        return cls(cfg, encoder)
    
    
    def set_mt_only(self):
        self.is_text_input = True
        self.encoder.is_text_input = True
        
    def forward(self, src_tokens, src_lengths, prev_output_tokens,is_text_input=False, **kwargs):
        if self.is_text_input:
            is_text_input = True
        pre_state_dict=["w2v_model.unit_embed_tokens_new.weight",
                        "w2v_model.layer_norm_text.weight",
                        "w2v_model.layer_norm_text.bias",
                        "w2v_model.embed_positions._float_tensor"
                        ]
        if self.encoder.fp:
            self.encoder = freeze_model(model=self.encoder, to_freeze_dict=pre_state_dict)
        else:
            for (name, param) in self.encoder.named_parameters():
                param.requires_grad = True
        encoder_out = self.encoder(src_tokens, src_lengths,is_text_input=is_text_input, **kwargs)
        x = self.encoder.final_dropout(encoder_out['encoder_out'][0])  # (T, B, C)
        decoder_out = self.encoder.w2v_model.decoder(
                    prev_output_tokens, encoder_out=encoder_out, **kwargs
                )
        if self.training:
            return decoder_out, encoder_out
        return decoder_out
    
    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.encoder.w2v_model.decoder(prev_output_tokens, **kwargs)

    def get_ctc_outputs(self,encoder_out):
        padding_mask = encoder_out["encoder_padding_mask"][0]
        encoder_out = encoder_out['encoder_out'][0]
        logits = self.encoder.ctc_proj(encoder_out)  # T x B x C
        #logits = self.encoder.final_dropout(encoder_out['encoder_out'][0])  # (T, B, C)
        out = utils.log_softmax(logits.float(), dim=-1)
        lens = out.new_full((out.shape[1],), out.shape[0]).long()
        if len(padding_mask) > 0:
            lens -= padding_mask[0].sum(dim=-1)
        return out, lens


    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """For decoder decoding."""
        return self.encoder.w2v_model.decoder.get_normalized_probs(net_output, log_probs, sample)
        
    @property
    def decoder(self):
        return self.encoder.w2v_model.decoder


class SpeechUTEncoder(FairseqEncoder):
    """
    Modified from fairseq.models.hubert.hubert_asr.HubertEncoder
    1. make it compatible with fairseq speech_to_text task
    2. make it compatible with encoder-decoder model
    """
    def __init__(self, cfg: SpeechUTS2TConfig, task):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            print(cfg.w2v_path)
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)



        w2v_args.task.data = cfg.data
        pretrain_task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            pretrain_task.load_state_dict(state["task_state"])
        else:
            pretrain_task.load_state_dict(task.state_dict())

        model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
        if state is not None and not cfg.no_pretrained_weights:
            try:            
                model.load_state_dict(state["model"], strict=True)
            except Exception as e:
                logger.warn(e)
                model.load_state_dict(state["model"], strict=False)
        self.cfg=cfg
        model.remove_pretraining_modules()

        super().__init__(pretrain_task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim
        self.w2v_model = model
        self.is_text_input = False # default
        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.freeze_pretrain=cfg.freeze_pretrain
        self.freeze_pretrain_mt=cfg.freeze_pretrain_mt
        self.num_updates = 0
        self.fp = self.num_updates <= self.freeze_pretrain
        self.ctc_proj = nn.Linear(d, len(task.target_dictionary))
        
    

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
        self.fp = self.num_updates < self.freeze_pretrain

        #if self.fp:
        #    print('***************************'+"fix pretrain---step:  "+str(self.num_updates)+'*******************************')
    
    def forward(self, src_tokens=None, src_lengths=None,is_text_input=False, **kwargs):
        if self.is_text_input:
            is_text_input = True
        if is_text_input:
            w2v_args = {
                "src_tokens": src_tokens,
                "mask": self.apply_mask and self.training,
                "fp":  self.fp,
            }
        else:
            w2v_args = {
                "source": src_tokens,
                "padding_mask": lengths_to_padding_mask(src_lengths),
                "mask": self.apply_mask and self.training,
            }

        ft = self.freeze_finetune_updates <= self.num_updates


        with torch.no_grad() if not ft else contextlib.ExitStack():
            if is_text_input:
                result = self.w2v_model.forward_text(**w2v_args)
                x=result["encoder_out"]
                embedding=result["embedding"]
                padding_mask=result["padding_mask"]
                
                x = x[0]
                return {
            "encoder_embedding": embedding,
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "padding_mask": [padding_mask],
        }
            else:
                embedding, x, padding_mask = self.w2v_model.extract_features(**w2v_args)
                        # B x T x C -> T x B x C
                x = x.transpose(0, 1)
                return {
                "encoder_embedding": embedding,
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [padding_mask],  # B x T
                "padding_mask": [padding_mask],
            }
    
    def forward_torchscript(self, net_input):
        """A TorchScript-compatible version of forward.

        Forward the encoder out.
        """
        '''_net_input = {
            "source": net_input["audio"],
            "padding_mask": lengths_to_padding_mask(net_input["audio_lengths"]),
            "mask": False,
        }'''
                
        _net_input = {
            "source": net_input["src_tokens"],
            "padding_mask": lengths_to_padding_mask(net_input["src_lengths"]),
            "mask": False,
        }

        embedding, x, padding_mask = self.w2v_model.extract_features(**_net_input)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = {
            "encoder_out" : [x],
            "encoder_padding_mask" : [padding_mask],
        }
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = [
                x.index_select(1, new_order) for x in encoder_out["encoder_out"]
            ]
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [
                x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]
            ]
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return 


