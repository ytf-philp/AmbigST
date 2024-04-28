# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List
import torch
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from speechut.criterions.label_smoothed_cross_entropy_with_contrastive_loss import \
    LabelSmoothedCrossEntropyWithContrastiveCriterion
#from speechut.criterions.contrastive_loss import \
#    InnerCLR
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
LARGE_NUM = 1e9
from fairseq.data.data_utils import lengths_to_padding_mask

@register_criterion("cross_entropy_with_supervised_contrastive")
class CrossEntropyWithSupervised(LabelSmoothedCrossEntropyWithContrastiveCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            contrastive_weight=0.0,
            mt_weight=0.0,
            contrastive_temperature=1.0,
            use_dual_ctr=False,
            ctr_dropout_rate=0.0,
            queue_k=768,
            top_k=8,
    ):
        print("align策略修改：0420: without kl_loss")
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy,contrastive_weight,mt_weight,
                         contrastive_temperature, use_dual_ctr,ctr_dropout_rate,queue_k,top_k)
        self.negative_w = 0.5
        self.temperature = 1
        self.weight = 0.5 #contrastive audio weight
        self.tag = "embedding_contrastive"
        self.contrast_mode = 'one'
        self.padding_idx = task.target_dictionary.pad()
        self.alpha=1.0
        self.contrast_tmp=torch.tensor(0.0)

        
    def compute_loss_with_lprobs(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs, target
    
    
    def compute_kl_loss(self, model, net_output, pad_mask=None, reduce=True):
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)
        #7 ,48
        p,q = torch.split(net_prob, net_prob.size(0) // 2, dim=0)
        p_tec, q_tec = torch.split(net_prob_tec, net_prob_tec.size(0) // 2, dim=0)
        pad_mask, _ = torch.split(pad_mask, pad_mask.size(0) // 2, dim=0)

        p_loss = torch.nn.functional.kl_div(p, q_tec, reduction='none')
        q_loss = torch.nn.functional.kl_div(q, p_tec, reduction='none')

        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        if reduce:
            p_loss = p_loss.sum()
            q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    
    def get_align_embedding_with_label(self,model,encoder_out, align_pad, align_lengths, phone_info):
        audio = encoder_out["encoder_embedding"].transpose(0,1)
        mixseq = []
        bsz, _, fdim = audio.shape
        for i in range(bsz):
            flag = False
            word_length = align_lengths[i].item()
            seq = torch.zeros(0, fdim).cuda().type_as(audio)
            phone_idx, _ = torch.sort(phone_info[i])
            for j in range(word_length):
                flag = True
                #phone
                st, en = align_pad[i, j, 0:2]
                audio_seq = audio[i, st:en+1, :]
                #average token speech
                add=audio_seq.sum(dim=0).unsqueeze(0)
                seq = torch.cat((seq, audio_seq.sum(dim=0).unsqueeze(0)), dim=0)
            mixseq.append(seq)
            #audio result
        mixseq_length = torch.LongTensor([seq.size(0) for seq in mixseq]).cuda()
        bsz_new=mixseq_length.shape
        max_len = torch.max(mixseq_length).item()
        mixseq_pad = torch.zeros(bsz_new[0], max_len, fdim).cuda().type_as(audio)
        for i, seq in enumerate(mixseq):
            mixseq_pad[i, :seq.size(0)] = seq
        mixseq_encoder_padding_mask = lengths_to_padding_mask(mixseq_length)
    
        #mixseq_encoder_padding_mask = (~mixseq_encoder_padding_mask).float()
        #mixseq_pad = (mixseq_pad * mixseq_encoder_padding_mask.unsqueeze(-1)).sum(dim=1) / mixseq_encoder_padding_mask.sum(dim=1).unsqueeze(-1) 
        
        return mixseq_pad,mixseq_encoder_padding_mask

        
# TODO: mt_loss,encoder_output,position
     

    def forward_contrastive(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        features = nn.functional.normalize(features, dim=-1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().cuda()
        else:
            mask = mask.float().cuda()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
    

    def get_align_embedding(self,encoder_out, align_pad, align_lengths):
        audio = encoder_out["encoder_embedding"].transpose(0,1)
        audio_origin,audio_aug = torch.split(audio, audio.size(0) // 2, dim=0)
        bsz, _, fdim = audio_origin.shape
        seq = torch.zeros(0, fdim).cuda().type_as(audio)        
        origin = torch.zeros(0, fdim).cuda().type_as(audio)        
        aug = torch.zeros(0, fdim).cuda().type_as(audio)                
        for i in range(bsz): 
            word_length = align_lengths[i].item()
            for j in range(word_length):
                #phone
                st, en = align_pad[i, j, 0:2]
                audio_seq_orgin = audio_origin[i, st:en+1, :]
                audio_seq_aug = audio_aug[i, st:en+1, :]
                #average token speech
                origin = torch.cat((origin, audio_seq_orgin.sum(dim=0).unsqueeze(0)), dim=0)
                aug = torch.cat((aug, audio_seq_aug.sum(dim=0).unsqueeze(0)), dim=0)   
        seq = torch.cat((origin.unsqueeze(1), aug.unsqueeze(1)), dim=1)
        return seq
    
    def forward(self, model, sample, reduce=True):
        #alpha=0
        alpha=1
        sample["net_input"]["is_text_input"] = False
        label_smoothed_nll_loss, nll_loss = torch.tensor(0.0), torch.tensor(0.0)
        label_smoothed_nll_loss_mt, nll_loss_mt = torch.tensor(0.0), torch.tensor(0.0)
        audio, audio_lengths, source, source_lengths, prev_output_tokens, align_pad, align_lengths,phone_info,is_text_input = sample["net_input"].values()
        kl_loss,output_jsd,label_smoothed_nll_loss_mix,contrastive_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),torch.tensor(0.0)
        
        
        
        if model.training:
             #计算原始st loss + R-drop loss
            if self.tag == "embedding_fusion":
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                w2v_args = {
                    "source": torch.cat([audio, audio.clone()], 0),
                    "padding_mask": torch.cat([lengths_to_padding_mask(audio_lengths), lengths_to_padding_mask(audio_lengths).clone()], 0),
                    "mask": self.training,
                }
                
                audio,output,audio_encoder_padding_mask = model.encoder.w2v_model.extract_features(**w2v_args) # S x 2B  x H
                source, source_encoder_padding_mask = model.encoder.w2v_model.forward_text_embedding(source)   # B x S x H
                audio=audio.transpose(0,1)
                
                
                source_st,source_mix = torch.split(audio, audio.size(0) // 2, dim=0) 

                #S x B x H
                source_mix, source_encoder_padding_mask_st = model.get_mixed_input(source_mix.transpose(0,1), source.transpose(0, 1), align_pad, align_lengths, phone_info)
                audio_encoder_padding_mask_origin,_ = torch.split(audio_encoder_padding_mask, audio_encoder_padding_mask.size(0) // 2, dim=0)
                #拼接
                # B x S x H
                source_mix=source_mix.transpose(0,1)
                pad_size=source_st.shape[1]-source_mix.shape[1]
                source_mix_pad=torch.nn.functional.pad(source_mix, (0, 0, 0, pad_size, 0, 0), value=0)
                source_st_concat= torch.cat([source_st, source_mix_pad.clone()], 0)

                source_encoder_padding_mask_st = ~source_st_concat.ne(0).all(dim=-1)
                
                source_encoder_padding_mask_st_concat=source_encoder_padding_mask_st
                
                
                encoder_out = self.get_encoder_out(model,source_st_concat,source_encoder_padding_mask_st_concat)
                net_output = model.encoder.w2v_model.decoder(
                    prev_output_tokens, encoder_out=encoder_out,
                )
            
                label_smoothed_nll_loss, nll_loss, lprobs, target = self.compute_loss_with_lprobs(model, net_output, sample, reduce=reduce)
            
            #计算 st loss 与 mix loss 的kl散度
                output_jsd = self.compute_jsd_loss(lprobs, target, self.padding_idx)
                loss=label_smoothed_nll_loss+label_smoothed_nll_loss_mix+output_jsd*self.alpha
       
            if self.tag == "r_drop":
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                sample_input = sample['net_input']
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                
                }
                net_output,encoder_out = model(**sample_concat_input)
                target = model.get_targets(sample, net_output)
                align_pad, align_lengths, phone_info=torch.cat([align_pad,align_pad.clone()],0),torch.cat([align_lengths,align_lengths.clone()],0),torch.cat([phone_info,phone_info.clone()],0)
                feature_embedding = self.get_align_embedding(model,encoder_out, align_pad, align_lengths, phone_info)
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                
                contrastive_loss = self.forward_contrastive(feature_embedding,torch.cat([sample_input['source'],sample_input['source'].clone()],0))
                loss = label_smoothed_nll_loss+ kl_loss * alpha+contrastive_loss
            
            if self.tag == "embedding_contrastive":
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                sample_input = sample['net_input']
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                
                }
                net_output,encoder_out = model(**sample_concat_input)
                target = model.get_targets(sample, net_output)
                #align_pad, align_lengths, phone_info=torch.cat([align_pad,align_pad.clone()],0),torch.cat([align_lengths,align_lengths.clone()],0),torch.cat([phone_info,phone_info.clone()],0)
                feature_embedding = self.get_align_embedding(encoder_out, align_pad, align_lengths)
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
                
                contrastive_loss = self.forward_contrastive(feature_embedding)
                contrastive_loss = contrastive_loss*10
                if contrastive_loss.isnan():
                    contrastive_loss = self.contrast_tmp.clone().detach()
                else:
                    self.contrast_tmp=contrastive_loss.clone().detach()
                loss = label_smoothed_nll_loss+ kl_loss * alpha+contrastive_loss     
        
        else:
            sample_input = sample['net_input']
            sample_input = {
            'src_tokens': sample_input['audio'],
            'src_lengths': sample_input['audio_lengths'],
            'prev_output_tokens': sample_input['prev_output_tokens'],}
            
            net_output = model(**sample_input)
            label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            loss = label_smoothed_nll_loss



        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        nsentences = sample["target"].size(0)
        #loss = label_smoothed_nll_loss


        logging_output = {
            "loss": loss.data,
            "contrastive_loss": contrastive_loss.data,
            "nll_loss": label_smoothed_nll_loss.data,
            "kl_loss": kl_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        kl_loss_sum = sum(log.get("kl_loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), ntokens, round=3)
        metrics.log_scalar("nll_loss",
                           nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar("kl_loss",
                           kl_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar("contrastive_loss",
                           contrastive_loss_sum / ntokens / math.log(2), ntokens, round=3)

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )