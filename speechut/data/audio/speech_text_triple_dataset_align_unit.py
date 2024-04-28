# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import os.path as op
import re
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
import torch.nn.functional as F
from speechut.data.audio.audio_utils import (
    get_fbank, get_waveform, get_segment_waveform, read_from_stored_zip, is_npy_data,
    is_sf_audio_data, parse_path, FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform


logger = logging.getLogger(__name__)

def mix_input(audio, source, align_pad, align_lengths, prob):
    # token replace
    #audio = audio.transpose(0, 1)
    #source = source.transpose(0, 1)
    mixseq = []
    bsz, fdim = audio.shape
    if not isinstance(prob, list):
        prob = [prob for i in range(bsz)]
    for i in range(bsz):
        word_length = align_lengths[i].item()
        word_prob = torch.rand(word_length)
        word_sample = word_prob < prob[i]
        seq = torch.zeros(0, fdim).cuda().type_as(audio)
        for j in range(word_length):
            if word_sample[j]:
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
    mixseq_pad = mixseq_pad.transpose(0, 1)
    return mixseq_pad
class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "S2T data config")
        self.config = {}
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                raise Exception(f"Failed to load config from {yaml_path}: {e}")
        else:
            raise FileNotFoundError(f"{yaml_path} not found")

    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", "dict.txt")

    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("pre_tokenizer", {"tokenizer": None})

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("bpe_tokenizer", {"bpe": None})

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per audio channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input audio"""
        return self.config.get("input_channels", 1)

    @property
    def sampling_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling.
        (alpha = 1 for no resampling)"""
        return self.config.get("sampling_alpha", 1.0)

    @property
    def use_audio_input(self):
        """Needed by the dataset loader to see if the model requires
        raw audio as inputs."""
        return self.config.get("use_audio_input", False)

    @property
    def audio_root(self):
        """Audio paths in the manifest TSV can be relative and this provides
        the root path. Set this to empty string when using absolute paths."""
        return self.config.get("audio_root", "")

    def get_feature_transforms(self, split, is_train):
        """Split-specific feature transforms. Allowing train set wildcard `_train`,
        evaluation set wildcard `_eval` and general wildcard `*` for matching."""
        from copy import deepcopy

        cfg = deepcopy(self.config)
        _cur = cfg.get("transforms", {})
        cur = _cur.get(split)
        cur = _cur.get("_train") if cur is None and is_train else cur
        cur = _cur.get("_eval") if cur is None and not is_train else cur
        cur = _cur.get("*") if cur is None else cur
        cfg["transforms"] = cur
        return cfg


def get_features_from_npy_or_audio(path):
    ext = op.splitext(op.basename(path))[1]
    if ext not in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        raise ValueError(f'Unsupported file format for "{path}"')
    return np.load(path) if ext == ".npy" else get_fbank(path)


def get_features_or_waveform_from_stored_zip(
    path, byte_offset, byte_size, need_waveform=False
):
    assert path.endswith(".zip")
    data = read_from_stored_zip(path, byte_offset, byte_size)
    f = io.BytesIO(data)
    if is_npy_data(data):
        features_or_waveform = np.load(f)
    elif is_sf_audio_data(data):
        features_or_waveform = \
            get_waveform(f, always_2d=False)[0] if need_waveform else get_fbank(f)
    else:
        raise ValueError(f'Unknown file format for "{path}"')
    return features_or_waveform


def get_raw_waveform_from_audio(
        path, byte_offset, byte_size):
    return get_segment_waveform(path, byte_offset, byte_size)[0].squeeze(0)


def get_features_or_waveform(path: str, need_waveform=False):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    _path, slice_ptr = parse_path(path)
    if len(slice_ptr) == 0:
        if need_waveform:
            return get_waveform(_path, always_2d=False)
        return get_features_from_npy_or_audio(_path)
    elif len(slice_ptr) == 2:
        if _path.endswith(".zip"):
            features_or_waveform = get_features_or_waveform_from_stored_zip(
                _path, slice_ptr[0], slice_ptr[1], need_waveform=need_waveform
            )
        else:
            features_or_waveform = get_raw_waveform_from_audio(
                _path, slice_ptr[0], slice_ptr[1]
            )
    else:
        raise ValueError(f"Invalid path: {path}")

    return features_or_waveform


def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if frames[0].dim() == 1:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


class SpeechToTextDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        task: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        audio_aligns: Optional[List[str]] = None,
        text_aligns: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        phone_idx: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        unit_tokenizer=None,
    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.audio_paths, self.n_frames = audio_paths, n_frames
        self.n_samples = len(audio_paths)
        self.task = task
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert audio_aligns is None or len(audio_aligns) == self.n_samples
        assert text_aligns is None or len(text_aligns) == self.n_samples
        assert speakers is None or len(speakers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
            tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.audio_aligns, self.text_aligns = audio_aligns, text_aligns
        self.tgt_dict = tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False
        self.phone_idx = phone_idx
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, is_train_split)
        )
        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.unit_tokenizer = unit_tokenizer
        
        if is_train_split:
            assert unit_tokenizer.pad() == tgt_dict.pad()
            assert unit_tokenizer.eos() == tgt_dict.eos()
            assert unit_tokenizer.unk() == tgt_dict.unk()
        if self.task == "mt":
            self.src_texts_size = [len(self.tgt_dict.encode_line(self.tokenize_text(src_text), add_if_not_exist=False, append_eos=True)) for src_text in src_texts]

        logger.info(self.__repr__())

    def __repr__(self):
        return (
            self.__class__.__name__
            + f'(split="{self.split}", n_samples={self.n_samples}, '
            f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
            f"shuffle={self.shuffle}, transforms={self.feature_transforms})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def tokenize_text(self, text: str):
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text
    
    def load_indexed_dataset(self, text: str):
        str_list=text.split(" ")
        idx=[]
        idx.append(self.unit_tokenizer.indices.get('<s>'))
        for str in str_list:
            idx.append(self.unit_tokenizer.indices.get(str))
        idx.append(self.unit_tokenizer.indices.get('</s>'))
        idx=torch.tensor(idx)
        return idx
    
    def __getitem__(
        self, index: int
    ):
        # audio
        audio = None
        if self.task != "mt":
            audio = get_features_or_waveform(
                self.audio_paths[index], need_waveform=self.data_cfg.use_audio_input
            )
            if self.data_cfg.config["use_audio_input"]:
                audio = torch.from_numpy(audio).float()
                if self.data_cfg.standardize_audio:
                    with torch.no_grad():
                        audio = F.layer_norm(audio, audio.shape)
            else:
                if self.feature_transforms is not None:
                    audio = self.feature_transforms(audio)
                audio = torch.from_numpy(audio).float()

        # source
        #tokenized = self.tokenize_text(self.src_texts[index])
        #source = self.tgt_dict.encode_line(
        #    tokenized, add_if_not_exist=False, append_eos=False
        #).long()
        if self.src_texts is not None and self.is_train_split:
            src_text = self.load_indexed_dataset(self.src_texts[index]).long()
            if self.data_cfg.prepend_src_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                src_text = torch.cat((torch.LongTensor([lang_tag_idx]), src_text), 0)
        elif self.src_texts is not None and not self.is_train_split:
            #src_text=None
            print(index)
            tokenized = self.tokenize_text(self.src_texts[index])
            src_text = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
            if self.data_cfg.prepend_src_lang_tag:
                lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
                lang_tag_idx = self.tgt_dict.index(lang_tag)
                src_text = torch.cat((torch.LongTensor([lang_tag_idx]), src_text), 0)
        
        
        # target
        tokenized = self.tokenize_text(self.tgt_texts[index])
        target = self.tgt_dict.encode_line(
            tokenized, add_if_not_exist=False, append_eos=True
        ).long()
        if self.data_cfg.prepend_tgt_lang_tag:
            lang_tag = self.LANG_TAG_TEMPLATE.format(self.tgt_langs[index])
            lang_tag_idx = self.tgt_dict.index(lang_tag)
            target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)
        # align
        if self.audio_aligns[index] == "-1":
            audio_align = ["0,0"]
            text_align = ["0,0"]
            audio_begin = torch.tensor([int(pair.split(',')[0]) for pair in audio_align]).unsqueeze(0).long()
            audio_end = torch.tensor([int(pair.split(',')[1]) for pair in audio_align]).unsqueeze(0).long()
            text_begin = torch.tensor([int(pair.split(',')[0]) for pair in text_align]).unsqueeze(0).long()
            text_end = torch.tensor([int(pair.split(',')[1]) for pair in text_align]).unsqueeze(0).long()
            align_info = torch.cat([audio_begin, audio_end, text_begin, text_end], dim=0).transpose(0, 1)
            phone_info = [-1]
        else:
            audio_align = self.audio_aligns[index].split('|')
            text_align = self.text_aligns[index].split('|')
            audio_begin = torch.tensor([int(pair.split(',')[0]) for pair in audio_align]).unsqueeze(0).long()
            audio_end = torch.tensor([int(pair.split(',')[1]) for pair in audio_align]).unsqueeze(0).long()
            text_begin = torch.tensor([int(pair.split(',')[0]) for pair in text_align]).unsqueeze(0).long()
            text_end = torch.tensor([int(pair.split(',')[1]) for pair in text_align]).unsqueeze(0).long()
            align_info = torch.cat([audio_begin, audio_end, text_begin, text_end], dim=0).transpose(0, 1)
            if "|" in self.phone_idx[index]:
                phone_info = [int(x) for x in self.phone_idx[index].split("|")]
            else:
                phone_info = [int(self.phone_idx[index])]
        return index, audio, src_text, target, align_info, phone_info

    def __len__(self):
        return self.n_samples

    def save_to_dict(self, phone_info, indices, frames, n_frames, source, source_lengths, prev_output_tokens, align_pad, align_lengths, target, target_lengths, ntokens, nsentences):
        out = {
            "id": indices,
            "net_input": {
                "audio": frames,
                "audio_lengths": n_frames,
                "source": source,
                "source_lengths": source_lengths,
                "prev_output_tokens": prev_output_tokens,
                "align_pad": align_pad,
                "align_lengths": align_lengths,
                "phone_info": phone_info
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": nsentences,
        }
        return out

    def collater(self, samples) -> Dict:
        if len(samples) == 0:
            return {}
        # sort samples by descending number of frames FOR MT

        n_frames = torch.tensor([s.size(0) for _, s, _, _, _, _ in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        # indices and audio frames
        indices = torch.tensor([i for i, _, _, _, _, _ in samples], dtype=torch.long).index_select(0, order)
        frames = _collate_frames(
            [s for _, s, _, _, _, _ in samples], self.data_cfg.use_audio_input
        ).index_select(0, order)
        n_frames = torch.tensor(
            [s.size(0) for _, s, _, _, _, _ in samples], dtype=torch.long
        ).index_select(0, order)
        
        # source text
        source = fairseq_data_utils.collate_tokens(
            [s for _, _, s, _, _, _ in samples],
            self.tgt_dict.pad(),
            self.tgt_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        ).index_select(0, order)
        source_lengths = torch.tensor(
            [s.size(0) for _, _, s, _, _, _ in samples], dtype=torch.long
        ).index_select(0, order)
        
        # target text
        if self.task != "asr":
            target = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            ).index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for _, _, _, t, _, _ in samples], dtype=torch.long
            ).index_select(0, order)
        # previous output tokens
        prev_output_tokens = None
        ntokens = None
        if self.task == "asr":
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [s for _, _, s, _, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            ).index_select(0, order)
            ntokens = sum(s.size(0) for _, _, s, _, _, _ in samples)
        else:
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t for _, _, _, t, _, _ in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            ).index_select(0, order)
            ntokens = sum(t.size(0) for _, _, _, t, _, _ in samples)
        # align
        align_info = [a for _, _, _, _, a, _ in samples]
        align_lengths = torch.LongTensor([a.size(0) for a in align_info])
        align_maxlen = torch.max(align_lengths)
        align_pad = torch.full((len(align_info), align_maxlen, 4), -1, dtype=torch.long)
        for i in range(len(align_info)):
            tmp = torch.LongTensor(align_info[i])
            if tmp.dim() == 2:
                align_pad[i, :align_lengths[i], :] = tmp
        align_pad = align_pad.index_select(0, order)
        align_lengths = align_lengths.index_select(0, order)
        
        phone_info = [a for _, _, _, _, _, a in samples]
        if self.is_train_split:
            phone_lengths = torch.LongTensor([len(a) for a in phone_info])
            phone_maxlen = torch.max(phone_lengths)
            padded_lists = [lst + [-1] * (phone_maxlen - len(lst)) for lst in phone_info]
            phone_info = torch.LongTensor(padded_lists).index_select(0, order)
        else:
            phone_info = torch.LongTensor([a for a in phone_info]).index_select(0, order)
        # out
        out = self.save_to_dict(phone_info, indices, frames, n_frames, source, source_lengths, prev_output_tokens, align_pad, align_lengths, target, target_lengths, ntokens, len(samples))
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    def prefetch(self, indices):
        raise False


class SpeechToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_SRC_TEXT, KEY_TGT_TEXT = "src_text", "tgt_text"
    # force alignment columns
    KEY_AUDIO_ALIGN, KEY_TEXT_ALIGN = "audio_align", "text_align"
    KEY_AUDIO_PHONE = "idx"
    # optional columns
    KEY_SPEAKER = "speaker"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SPEAKER = DEFAULT_LANG = ""

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        task: str,
        is_train_split,
        samples: List[List[Dict]],
        data_cfg: S2TDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        unit_tokenizer
    ) -> SpeechToTextDataset:
        audio_paths, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        audio_aligns, text_aligns = [], []
        phone_idx = []
        speakers, src_langs, tgt_langs = [], [], []
        for s in samples:
            ids.extend([ss[cls.KEY_ID] for ss in s])
            audio_paths.extend(
                [op.join(data_cfg.audio_root, ss[cls.KEY_AUDIO]) for ss in s]
            )
            n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
            src_texts.extend([ss[cls.KEY_SRC_TEXT] for ss in s])
            tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
            audio_aligns.extend([ss[cls.KEY_AUDIO_ALIGN] for ss in s])
            text_aligns.extend([ss[cls.KEY_TEXT_ALIGN] for ss in s])
            speakers.extend([ss.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for ss in s])
            src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
            tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
            phone_idx.extend([ss.get(cls.KEY_AUDIO_PHONE) for ss in s])
            
        return SpeechToTextDataset(
            split_name,
            task,
            is_train_split,
            data_cfg,
            audio_paths,
            n_frames,
            src_texts,
            tgt_texts,
            audio_aligns,
            text_aligns,
            speakers,
            src_langs,
            tgt_langs,
            ids,
            phone_idx,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            unit_tokenizer
        )

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    @classmethod
    def from_tsv(
        cls,
        root: str,
        task: str,
        data_cfg: S2TDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        unit_tokenizer,
    ) -> SpeechToTextDataset:
        samples = []
        _splits = splits.split(",")
        for split in _splits:
            tsv_path = op.join(root, f"{split}_seg_plus.tsv")
            if not op.isfile(tsv_path):
                raise FileNotFoundError(f"Dataset not found: {tsv_path}")
            with open(tsv_path) as f:
                reader = csv.DictReader(
                    f,
                    delimiter="\t",
                    quotechar=None,
                    doublequote=False,
                    lineterminator="\n",
                    quoting=csv.QUOTE_NONE,
                )
                samples.append([dict(e) for e in reader])
                assert len(samples) > 0

        datasets = [
            cls._from_list(
                name,
                task,
                is_train_split,
                [s],
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
                unit_tokenizer
            )
            for name, s in zip(_splits, samples)
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)

if __name__ == "__main__":

    data_cfg = S2TDataConfig("/zhang_m/STEMM/data/mustc/en-de/config_ende.yaml")
    tgt_dict = Dictionary.load("/zhang_m/STEMM/data/mustc/en-de/spm_unigram10000_st.txt")
    print(tgt_dict)
    #print(data_cfg)
    dev_dataset = SpeechToTextDatasetCreator.from_tsv(
        "/zhang_m/STEMM/data/mustc/en-de",
        "mix",
        data_cfg,
        "train_homophone_align",
        tgt_dict,
        None,
        None,
        is_train_split=False,
        epoch=1,
        seed=1)

    loader = DataLoader(dev_dataset, 4, shuffle=False, collate_fn=dev_dataset.collater)
    for data in loader:
        print(data)
        result=mix_input(data["net_input"]["audio"], data["net_input"]["source"], data["net_input"]["align_pad"], data["net_input"]["align_lengths"], 0.75)
    
    for item in dev_dataset:
        print(item)
        