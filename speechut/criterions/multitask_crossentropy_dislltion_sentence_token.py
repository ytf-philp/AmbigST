# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


#5.4 mt loss
#5.11 encoder contrastive

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


@register_criterion("multi_task_cross_entropy_dislltion_sentence_token")
class MultiTaskCrossEntropyWithDropAlign(LabelSmoothedCrossEntropyWithContrastiveCriterion):
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
        print("0.9* label_smoothed_nll_loss + kl_loss + 0.1 * ( contrastive_loss + contrastive_loss_sentence )")
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy,contrastive_weight,mt_weight,
                         contrastive_temperature, use_dual_ctr,ctr_dropout_rate,queue_k,top_k)
        self.negative_w = 0.5
        self.weight = 0.5 #contrastive audio weight
        self.tag = "token_sentence_level_mask"
        print(self.tag)
        self.padding_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.blank_idx = task.target_dictionary.bos()
        self.temperature = 0.05
        self.alpha=1
        self.sim = 'cosine'

        
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
    
    def compute_loss_mt(self, model, sample, reduce=True):
        net_output, _ = model(sample["net_input"]["source"], sample["net_input"]["source_lengths"],
                              sample["net_input"]["prev_output_tokens"], is_text_input=True)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss,net_output

    def compute_ctc_loss(self, model, encoder_output, target,target_lengths, reduction):
        ctc_lprobs, ctc_lens = model.get_ctc_outputs(encoder_output)
        ctc_tgt, ctc_tgt_lens = target,target_lengths
        ctc_tgt_mask = ~lengths_to_padding_mask(ctc_tgt_lens)
        ctc_tgt_flat = ctc_tgt.masked_select(ctc_tgt_mask)
        loss = F.ctc_loss(
                ctc_lprobs,
                ctc_tgt_flat,
                ctc_lens,
                ctc_tgt_lens,
                reduction=reduction,
                zero_infinity=True,
            )
        return loss
    '''def compute_kl_loss(self, model, net_output_st, net_output_mix,pad_mask_st,pad_mask_mix, ignore_index):
        net_prob_st = model.get_normalized_probs(net_output_st, log_probs=True)
        net_prob_st_tec = model.get_normalized_probs(net_output_st, log_probs=False)
        
        net_prob_mix = model.get_normalized_probs(net_output_mix, log_probs=True)
        net_prob_mix_tec = model.get_normalized_probs(net_output_mix, log_probs=False)
        kl_loss_st = F.kl_div(net_prob_st, net_prob_mix_tec, reduction="none")
        kl_loss_mix = F.kl_div(net_prob_mix, net_prob_st_tec, reduction="none")
        
        kl_loss_st.masked_fill_(pad_mask_st, 0.0)
        kl_loss_mix.masked_fill_(pad_mask_mix, 0.0)
        
        kl_loss_st = kl_loss_st.sum()
        kl_loss_mix = kl_loss_mix.sum()
        kl_loss = (kl_loss_st + kl_loss_mix) / 2.0
        return kl_loss'''
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

        loss = p_loss

        return loss
    
    def get_encoder_out(self,model, source_st,source_encoder_padding_mask_st):
        src_tokens, x, _ = model.encoder.w2v_model.convert_embeddings(
                source_st,
                source_encoder_padding_mask_st,
                mix_with_unit=False,
                use_pred_unit=False,
            )
        encoder_out = model.encoder.w2v_model.unit_encoder(
                src_tokens,
                token_embeddings=x,
                return_all_hiddens= None
            )
        encoder_out = {
            "encoder_out" : [encoder_out['encoder_out'][0]],
            "encoder_padding_mask" : [source_encoder_padding_mask_st],
        }
        return encoder_out
        
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
    
    def mask_align(self,encoder_out, encoder_pad, align_pad, align_lengths,phone_info,lp=0.6):
        
        bsz, seq, fdim = encoder_out.shape
        mixseq_pad =torch.clone(encoder_pad).cuda()
        flag = False
        seq_length=(~encoder_pad).float().sum(axis=1)    
        for i in range(bsz): 
            phone_idx, _ = torch.sort(phone_info[i])
            if phone_idx.max()<0:
                flag = True
            word_length = align_lengths[i].item()
            for j in range(word_length):
                st, en = align_pad[i, j, 0:2]
                if j in phone_idx:
                    word_prob = torch.rand(en-st+1)
                    word_sample = word_prob < lp
                    mixseq_pad[i,st:en+1]=word_sample.cuda()
                else:
                    mixseq_pad[i,st:en+1]=encoder_pad[i,st:en+1]
            if flag == True:
                    word_prob = torch.rand(seq_length[i].int())
                    word_sample = word_prob < 0.1
                    mixseq_pad[i,:seq_length[i].int()] = word_sample.cuda()
        return mixseq_pad
    
    def similarity_function(self):
        return nn.CosineSimilarity(dim=-1)
    
    def get_contrastive_loss(self, encoder_out1, encoder_out2):
        
        def _sentence_embedding(encoder_out):
            encoder_output,mask = encoder_out["encoder_out"][0],encoder_out["encoder_padding_mask"][0]
            encoder_output = encoder_output.transpose(0, 1)
            mask = (~mask).float()
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return encoder_embedding
        
        encoder_embedding1 = _sentence_embedding(encoder_out1)  # [batch, hidden_size]
        encoder_embedding2 = _sentence_embedding(encoder_out2)  # [batch, hidden_size]
        
        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2
        
        similarity_function = self.similarity_function()
        anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                  torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))
        
        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()
        
        return loss

    def compute_contrastive_sentence_mask_loss(self, out_origin,out_mix,pad_origin,pad_mix):

        def _sentence_embedding(encoder_out,pad):
            encoder_output,mask = encoder_out,pad
            mask = (~mask).float()
            encoder_embedding = (encoder_output * mask.unsqueeze(-1)).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(-1)  # [batch, hidden_size]
            return encoder_embedding
        
        #out_mix = out_mix / out_mix.norm(dim=2, keepdim=True)
        #out_origin = out_origin / out_origin.norm(dim=2, keepdim=True)
        encoder_embedding1 = _sentence_embedding(out_origin,pad_origin)  # [batch, hidden_size]
        encoder_embedding2 = _sentence_embedding(out_mix,pad_mix)  # [batch, hidden_size]
        
        #batch_size = encoder_embedding2.shape[0]
        #feature_dim = encoder_embedding2.shape[1]
        
        similarity_function = self.similarity_function()
        #anchor_dot_contrast = similarity_function(anchor_feature,
        #                                          contrast_feature.transpose(1,2))
        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2
        
        similarity_function = self.similarity_function()
        anchor_dot_contrast = similarity_function(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                  torch.transpose(contrast_feature.expand((batch_size, batch_size, feature_dim)), 0, 1))
        
        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()
        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()
        
        return loss


    def compute_contrastive_token_mask_loss(self, out_origin,out_mix,pad_origin,pad_mix):

        def _contrastive_label_smoothed_nll_loss(contrastive_scores, contrastive_labels, eps=0.0):
            '''
                contrasive_scores: bsz x seqlen x seqlen
                contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.
            '''
            
            bsz, seqlen, _ = contrastive_scores.size()
            logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
            gold = torch.arange(seqlen).view(-1,)
            gold = gold.expand(bsz, seqlen).contiguous().view(-1)
            if contrastive_scores.is_cuda:
                gold = gold.cuda(contrastive_scores.get_device())
            loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
            loss = loss.view(bsz, seqlen) * contrastive_labels
            loss = torch.sum(loss) / contrastive_labels.sum()

            _, pred = torch.max(logprobs, -1)
            correct_num = torch.eq(gold, pred).float().view(bsz, seqlen)
            correct_num = torch.sum(correct_num * contrastive_labels)
            total_num = contrastive_labels.sum()
            return loss, correct_num, total_num
        
        #mask origin pad
        pad_origin,pad_mix = pad_origin.float(), pad_mix.float()
        pad_token = pad_mix - pad_origin
        out_mix = out_mix / out_mix.norm(dim=2, keepdim=True)
        out_origin = out_origin / out_origin.norm(dim=2, keepdim=True)
        anchor_feature = out_origin
        contrast_feature = out_mix
        
        #similarity_function = self.similarity_function()
        anchor_dot_contrast = torch.matmul(anchor_feature,
                                                  contrast_feature.transpose(1, 2))/ self.temperature
        
        loss, correct_num, total_num = _contrastive_label_smoothed_nll_loss(anchor_dot_contrast, pad_token)

        
        return loss
    
# TODO: mt_loss,encoder_output,position
     
    def forward(self, model, sample, reduce=True):
        #alpha=0
        alpha=1
        sample["net_input"]["is_text_input"] = False
        label_smoothed_nll_loss, label_smoothed_nll_loss_mt, kl_loss = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        contrastive_loss,loss = torch.tensor(0.0),torch.tensor(0.0)
        audio, audio_lengths, source, source_lengths, prev_output_tokens, align_pad, align_lengths,phone_info,is_text_input = sample["net_input"].values()
        ctc_loss,output_jsd,label_smoothed_nll_loss_mix = torch.tensor(0.0), torch.tensor(0.0),torch.tensor(0.0)
        if model.training:  

            if self.tag == "encoder_ctc1":
                target_origin = sample["net_input"]['source']
                target_length = sample["net_input"]["source_lengths"]
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                sample_input = sample['net_input']
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                }
                net_output_st,encoder_out = model(**sample_concat_input)
                out=encoder_out["encoder_out"][0].transpose(0,1)
                pad=encoder_out["encoder_padding_mask"][0]    
                p,q = torch.split(out, out.size(0) // 2, dim=0)
                p_pad,q_pad = torch.split(pad, pad.size(0) // 2, dim=0)
                
                encoder_out_st = {
                    "encoder_out" : [p.transpose(0,1)],
                    "encoder_padding_mask" : [p_pad],
                }
                encoder_out_mix = {
                    "encoder_out" : [q.transpose(0,1)],
                    "encoder_padding_mask" : [q_pad],
                }
                target = model.get_targets(sample, net_output_st)
                ctc_loss = self.compute_ctc_loss(model, encoder_out_st, target_origin,target_length , reduction="sum")
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                #kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
                out=encoder_out["encoder_out"][0].transpose(0,1)
                pad=encoder_out["encoder_padding_mask"][0]
                sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                )
                nsentences = sample["target"].size(0)
                ntokens = sample["ntokens"]
                ctc_loss=ctc_loss *  nsentences / ntokens
                
                contrastive_loss = self.get_contrastive_loss(
                    encoder_out_st,
                    encoder_out_mix
                )

                contrastive_loss = contrastive_loss * ntokens / nsentences

                loss = 0.7* label_smoothed_nll_loss+contrastive_loss + 0.3* ctc_loss   

            if self.tag == "encoder_ctc_r_drop":
                target_origin = sample["net_input"]['source']
                target_length = sample["net_input"]["source_lengths"]
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                sample_input = sample['net_input']
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                }
                net_output_st,encoder_out = model(**sample_concat_input)
                target = model.get_targets(sample, net_output_st)
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                kl_loss = self.compute_kl_loss(model, net_output_st, pad_mask)
                
                out=encoder_out["encoder_out"][0].transpose(0,1)
                pad=encoder_out["encoder_padding_mask"][0]    
                p,q = torch.split(out, out.size(0) // 2, dim=0)
                p_pad,q_pad = torch.split(pad, pad.size(0) // 2, dim=0)
                
                encoder_out_st = {
                    "encoder_out" : [p.transpose(0,1)],
                    "encoder_padding_mask" : [p_pad],
                }
                encoder_out_mix = {
                    "encoder_out" : [q.transpose(0,1)],
                    "encoder_padding_mask" : [q_pad],
                }
                target = model.get_targets(sample, net_output_st)
                ctc_loss = self.compute_ctc_loss(model, encoder_out_st, target_origin,target_length , reduction="sum")
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                #kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
                out=encoder_out["encoder_out"][0].transpose(0,1)
                pad=encoder_out["encoder_padding_mask"][0]
                sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                )
                nsentences = sample["target"].size(0)
                ntokens = sample["ntokens"]
                ctc_loss=ctc_loss *  nsentences / ntokens
                
                #contrastive_loss = self.get_contrastive_loss(
                #    encoder_out_st,
                #    encoder_out_mix
                #)

                #contrastive_loss = contrastive_loss * ntokens / nsentences
                loss = 0.7* label_smoothed_nll_loss+ kl_loss + 0.3* ctc_loss   

            if self.tag == "sentence_level_mask":
                #origin input and mask input
                sample_input = sample['net_input']
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                }
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                encoder_out = model.encoder(**sample_concat_input)
                
                #split encoder
                double_out=encoder_out["encoder_out"][0].transpose(0,1)
                double_pad=encoder_out["encoder_padding_mask"][0]    
                out_origin,out_mix = torch.split(double_out, double_out.size(0) // 2, dim=0)
                pad_origin,pad_mix = torch.split(double_pad, double_pad.size(0) // 2, dim=0)
                
                
                # CTC loss
                encoder_out_st = {
                    "encoder_out" : [out_origin.transpose(0,1)],
                    "encoder_padding_mask" : [pad_origin],
                }
                target_origin = sample["net_input"]['source']
                target_length = sample["net_input"]["source_lengths"]
                
                sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
                )
                nsentences = sample["target"].size(0)
                ntokens = sample["ntokens"]

                # pad token
                pad_mix = self.mask_align(out_mix,pad_mix, align_pad, align_lengths,phone_info,lp=0.6)
                
                # CE loss
                decoder_input = {
                    "encoder_out" : [torch.cat([out_origin,out_mix],0).transpose(0,1)],
                    "encoder_padding_mask" : [torch.cat([pad_origin,pad_mix],0)],
                }
                net_output_st = model.encoder.w2v_model.decoder(
                    prev_output_tokens, encoder_out=decoder_input,
                )
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
                
                #self-dislltion
                target = model.get_targets(sample, net_output_st)
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                kl_loss = self.compute_kl_loss(model, net_output_st, pad_mask)

                #token-level contrastive
                #contrastive_loss = self.compute_contrastive_token_sentence_loss(out_origin,out_mix,pad_origin,pad_mix)
                #contrastive_loss = contrastive_loss * ntokens / nsentences
                contrastive_loss_sentence = self.compute_contrastive_sentence_mask_loss(out_origin,out_mix,pad_origin,pad_origin)
                contrastive_loss_sentence = contrastive_loss_sentence * ntokens / nsentences
                loss =  label_smoothed_nll_loss + kl_loss + contrastive_loss_sentence
        
            if self.tag == "token_level_mask":
                #origin input and mask input
                sample_input = sample['net_input']
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                }
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                encoder_out = model.encoder(**sample_concat_input)
                
                #split encoder
                double_out=encoder_out["encoder_out"][0].transpose(0,1)
                double_pad=encoder_out["encoder_padding_mask"][0]    
                out_origin,out_mix = torch.split(double_out, double_out.size(0) // 2, dim=0)
                pad_origin,pad_mix = torch.split(double_pad, double_pad.size(0) // 2, dim=0)
                

                # pad token
                pad_mix = self.mask_align(out_mix,pad_mix, align_pad, align_lengths,phone_info,lp=0.6)
                
                # CE loss
                decoder_input = {
                    "encoder_out" : [torch.cat([out_origin,out_mix],0).transpose(0,1)],
                    "encoder_padding_mask" : [torch.cat([pad_origin,pad_mix],0)],
                }
                net_output_st = model.encoder.w2v_model.decoder(
                    prev_output_tokens, encoder_out=decoder_input,
                )
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
                
                #self-dislltion
                target = model.get_targets(sample, net_output_st)
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                kl_loss = self.compute_kl_loss(model, net_output_st, pad_mask)
                
                nsentences = sample["target"].size(0)
                ntokens = sample["ntokens"]
                ctc_loss=ctc_loss *  nsentences / ntokens
                #token-level contrastive
                contrastive_loss = self.compute_contrastive_token_mask_loss(out_origin,out_mix,pad_origin,pad_mix)
                contrastive_loss = contrastive_loss * ntokens / nsentences
                
                #sentence-level contrastive
                #contrastive_loss_sentence = self.compute_contrastive_sentence_mask_loss(out_origin,out_mix,pad_origin,pad_origin)
                #contrastive_loss_sentence = contrastive_loss_sentence * ntokens / nsentences
                loss =  0.9* (label_smoothed_nll_loss + 0.5 * kl_loss) + 0.1 * contrastive_loss
            
            if self.tag == "token_sentence_level_mask":
                    #origin input and mask input
                sample_input = sample['net_input']
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                }
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                encoder_out = model.encoder(**sample_concat_input)
                
                #split encoder
                double_out=encoder_out["encoder_out"][0].transpose(0,1)
                double_pad=encoder_out["encoder_padding_mask"][0]    
                out_origin,out_mix = torch.split(double_out, double_out.size(0) // 2, dim=0)
                pad_origin,pad_mix = torch.split(double_pad, double_pad.size(0) // 2, dim=0)
                

                # pad token
                pad_mix = self.mask_align(out_mix,pad_mix, align_pad, align_lengths,phone_info,lp=0.5)
                
                # CE loss
                decoder_input = {
                    "encoder_out" : [torch.cat([out_origin,out_mix],0).transpose(0,1)],
                    "encoder_padding_mask" : [torch.cat([pad_origin,pad_mix],0)],
                }
                net_output_st = model.encoder.w2v_model.decoder(
                    prev_output_tokens, encoder_out=decoder_input,
                )
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
                
                #self-dislltion
                target = model.get_targets(sample, net_output_st)
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                kl_loss = self.compute_kl_loss(model, net_output_st, pad_mask)
                
                nsentences = sample["target"].size(0)
                ntokens = sample["ntokens"]
                ctc_loss=ctc_loss *  nsentences / ntokens
                #token-level contrastive
                contrastive_loss = self.compute_contrastive_token_mask_loss(out_origin,out_mix,pad_origin,pad_mix)
                contrastive_loss = contrastive_loss * ntokens / nsentences
                
                #sentence-level contrastive
                contrastive_loss_sentence = self.compute_contrastive_sentence_mask_loss(out_origin,out_mix,pad_origin,pad_origin)
                contrastive_loss_sentence = contrastive_loss_sentence * ntokens / nsentences
                loss =  0.9* (label_smoothed_nll_loss +  kl_loss) + 0.1 * ( contrastive_loss + contrastive_loss_sentence ) /2          
            
            if self.tag == "dislltion":
                    #origin input and mask input
                sample_input = sample['net_input']
                sample_concat_input = {
                'src_tokens': torch.cat([sample_input['audio'],sample_input['audio'].clone()],0),
                'src_lengths': torch.cat([sample_input['audio_lengths'],sample_input['audio_lengths'].clone()],0),                    
                'prev_output_tokens': torch.cat([sample_input['prev_output_tokens'], sample_input['prev_output_tokens'].clone()], 0),
                'is_text_input': sample_input["is_text_input"]
                }
                prev_output_tokens = torch.cat([prev_output_tokens, prev_output_tokens.clone()], 0)
                encoder_out = model.encoder(**sample_concat_input)
                
                #split encoder
                double_out=encoder_out["encoder_out"][0].transpose(0,1)
                double_pad=encoder_out["encoder_padding_mask"][0]    
                out_origin,out_mix = torch.split(double_out, double_out.size(0) // 2, dim=0)
                pad_origin,pad_mix = torch.split(double_pad, double_pad.size(0) // 2, dim=0)
                

                # pad token
                pad_mix = self.mask_align(out_mix,pad_mix, align_pad, align_lengths,phone_info,lp=0.5)
                
                # CE loss
                decoder_input = {
                    "encoder_out" : [torch.cat([out_origin,out_mix],0).transpose(0,1)],
                    "encoder_padding_mask" : [torch.cat([pad_origin,pad_mix],0)],
                }
                net_output_st = model.encoder.w2v_model.decoder(
                    prev_output_tokens, encoder_out=decoder_input,
                )
                sample['target'] = torch.cat([sample['target'], sample['target'].clone()], 0)
                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
                
                #self-dislltion
                target = model.get_targets(sample, net_output_st)
                pad_mask = target.unsqueeze(-1).eq(self.padding_idx)
                kl_loss = self.compute_kl_loss(model, net_output_st, pad_mask)
                
                nsentences = sample["target"].size(0)
                ntokens = sample["ntokens"]
                ctc_loss=ctc_loss *  nsentences / ntokens
                
                loss =  label_smoothed_nll_loss + 0.5 * kl_loss
            if self.tag == "origin":
                sample_input = sample['net_input']
                sample_concat_input = {
                'src_tokens': sample_input['audio'],
                'src_lengths': sample_input['audio_lengths'],                    
                'prev_output_tokens': sample_input['prev_output_tokens'],
                'is_text_input': sample_input["is_text_input"]
                }
                net_output_st,_ = model(**sample_concat_input)

                label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
                loss = label_smoothed_nll_loss 
        else:
            sample_input = sample['net_input']
            sample_input = {
            'src_tokens': sample_input['audio'],
            'src_lengths': sample_input['audio_lengths'],
            'prev_output_tokens': sample_input['prev_output_tokens'],}
            
            net_output_st = model(**sample_input)
            label_smoothed_nll_loss, nll_loss = self.compute_loss(model, net_output_st, sample, reduce=reduce)
            loss = label_smoothed_nll_loss

        sample_size = sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        nsentences = sample["target"].size(0)
        #loss = label_smoothed_nll_loss


        logging_output = {
            "loss": loss.data,
            "nll_loss": label_smoothed_nll_loss.data,
            "ctc_loss": ctc_loss.data,
            "contrastive_loss": contrastive_loss.data,
            "js_loss": kl_loss.data,
            "ntokens": sample["ntokens"],
            "avg_token": sample["ntokens"] / nsentences,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output_st, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        nll_loss_mt_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        
        avg_token_sum = sum(log.get("avg_token", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        js_loss_sum = sum(log.get("js_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss",
                           loss_sum / sample_size / math.log(2), ntokens, round=3)
        metrics.log_scalar("nll_loss",
                           nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar("ctc_loss",
                           nll_loss_mt_sum / ntokens / math.log(2), ntokens, round=3)  
        metrics.log_scalar("contrastive_loss",
                           contrastive_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar("js_loss",
                           js_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar("avg_token",
                           avg_token_sum / ntokens / math.log(2), ntokens, round=3)
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