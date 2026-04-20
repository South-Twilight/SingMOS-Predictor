import os
import json
import logging
import math

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from .utils import calc_pitch_note, calc_pitch_histogram, f0_dio
from .utils import pad_sequence
from .utils import HOP_SIZE


def _resolve_split_name(split_info: dict, dataset_name: str) -> str:
    """Resolve dataset key with backward-compatible aliases."""
    if dataset_name in split_info:
        return dataset_name
    alias_map = {
        "singeval_p1": "singmospro-p1",
        "singeval_p2": "singmospro-p2",
        "singeval_p3": "singmospro-p3",
        "singeval_p4": "singmospro-p4",
        "singeval_p5": "singmospro-p5",
    }
    alias = alias_map.get(dataset_name)
    if alias in split_info:
        return alias
    raise KeyError(f"Dataset split key not found: {dataset_name}")


def _collect_utts(split_info: dict, datasets: list, split_key: str) -> list:
    utts = []
    if "all" in datasets:
        for dataset in split_info.keys():
            utts.extend(split_info[dataset].get(split_key, []))
    else:
        for dataset_name in datasets:
            resolved = _resolve_split_name(split_info, dataset_name)
            utts.extend(split_info[resolved].get(split_key, []))
    return utts


class MOSDataset(Dataset):
    def __init__(
        self, 
        datadir,
        utt_list, 
        score_infos=None,
        sys_info=None,
        utt_to_batch_id=None,
        use_domain_id: bool = False,
        use_judge_id: bool = False,
        use_pitch: bool = False,
        pitch_type: str = "note",
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        padding_mode: str = "repeat",  # 新增参数
    ):
        # Backward compatibility: accept legacy arg name
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.use_judge_id = use_judge_id
        self.use_pitch = use_pitch
        self.use_domain_id = use_domain_id
        self.sys_info = sys_info
        self.utt_to_batch_id = utt_to_batch_id or {}
        self.padding_mode = padding_mode  # 保存填充方式
        assert pitch_type in ["raw", "note", "histogram"], "pitch_type must be note or histogram"
        self.pitch_type = pitch_type

        score_info = {idx: score_infos[idx] for idx in utt_list if idx in score_infos}
        sysnames = {}
        uttnames = []
        wavnames = [] # judge wavname
        wavs = {}
        gt_utt_scores = {}
        domain_ids = {}
        batch_ids = {}
        for idx in score_info.keys():
            # judge 0 means the mean of judges
            mean_id = idx + "_0"
            sys_id = score_info[idx]["sys_id"]
            if sys_info[sys_id]["sample_rate"] != 16000:
                continue
            wavnames.append(mean_id)
            gt_utt_scores[mean_id] = float(score_info[idx]["score"]["mos"])
            wavs[mean_id] = os.path.join(datadir, score_info[idx]["wav"])
            sysnames[score_info[idx]["sys_id"]] = True
            domain_ids[mean_id] = sys_info[sys_id]["tag"]["domain_id"]
            batch_ids[mean_id] = self.utt_to_batch_id.get(idx, "unknown")
            uttnames.append(idx)
            if use_judge_id:
                for judge, judge_score in zip(score_info[idx]["score"]["judges"], score_info[idx]["score"]["scores"]):
                    # judge id
                    judge_id = idx + f"_{judge}"
                    wavnames.append(judge_id)
                    gt_utt_scores[judge_id] = float(judge_score)
                    wavs[judge_id] = os.path.join(datadir, score_info[idx]["wav"])
                    domain_ids[judge_id] = sys_info[sys_id]["tag"]["domain_id"]
                    batch_ids[judge_id] = self.utt_to_batch_id.get(idx, "unknown")
        self.sysnames = sorted(sysnames.keys())
        self.uttnames = sorted(uttnames)
        self.wavnames = sorted(wavnames)
        self.wavs = {k: v for k, v in sorted(wavs.items(), key=lambda x: x[0])}
        self.gt_utt_scores = {k: v for k, v in sorted(gt_utt_scores.items(), key=lambda x: x[0])}
        self.domain_ids = {k: v for k, v in sorted(domain_ids.items(), key=lambda x: x[0])}
        self.batch_ids = {k: v for k, v in sorted(batch_ids.items(), key=lambda x: x[0])}
        logging.info(f'remained utt number: {len(self.wavnames)}, remained sys number: {len(self.sysnames)}')

        
    def __getitem__(self, idx):
        """
        return value: 
            wav: wav feature, [1, L]
            gt_utt_score: gt utterance score, [1]
            gt_frame_score: gt frame score, [1]
            domain_id: domain id, [1]
            judge_id: judge id, [1]
            pitch_var: pitch variation, [1, T]
            pitch_note: pitch note, [1, T]
            pitch_histogram: pitch histogram, [1, 120, T]
            pitch: pitch, [1, T]
        """
        wavname = self.wavnames[idx]
        wav, original_sample_rate = torchaudio.load(self.wavs[wavname])
        wav = torch.mean(wav, dim=0, keepdim=True)
        
        # 检查采样率并进行重采样
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate, 
                new_freq=self.sample_rate
            )
            wav = resampler(wav)
        
        # 检查音频长度并进行截断
        max_samples = int(self.max_duration * self.sample_rate)
        if wav.shape[-1] > max_samples:
            wav = wav[..., :max_samples]
            logging.info(f"Warning: {wavname} wav length is too long, truncate to {max_samples}")
        
        ret_dict = {}

        gt_utt_score = torch.tensor([self.gt_utt_scores[wavname]], dtype=torch.float32)
        domain_id = int(self.domain_ids[wavname])
        batch_id = self.batch_ids[wavname]
        ret_dict.update(domain_id=domain_id)
        ret_dict.update(batch_id=batch_id)

        items = wavname.split('_')
        wavname = "".join(items[:-1])
        judge_id = int(items[-1])
        ret_dict.update(judge_id=judge_id)
        token_len = math.ceil(wav.shape[-1] / HOP_SIZE)
        if self.use_pitch:
            if self.pitch_type == "note":
                pitch_var, pitch_note = calc_pitch_note(
                    wav[0],
                    sampling_rate=self.sample_rate,
                    use_log_f0=False,
                    use_continuous_f0=False,
                    use_discrete_f0=True,
                )
                pitch_var = pitch_var[:token_len]
                pitch_note = pitch_note[:token_len]
                ret_dict.update(pitch_var=pitch_var)
                ret_dict.update(pitch_note=pitch_note)
            elif self.pitch_type == "histogram":
                pitch_histogram = calc_pitch_histogram(
                    wav[0],
                    sampling_rate=self.sample_rate,
                    use_log_f0=False,
                    use_continuous_f0=False,
                    use_discrete_f0=True,
                )
                pitch_histogram = pitch_histogram[:token_len]
                ret_dict.update(pitch_histogram=pitch_histogram)
            elif self.pitch_type == "raw":
                pitch = f0_dio(
                    wav[0],
                    sampling_rate=self.sample_rate,
                    use_log_f0=False,
                    use_continuous_f0=False,
                    use_discrete_f0=True,
                )
                pitch = pitch[:token_len]
                ret_dict.update(pitch=pitch)
        return wavname, wav, gt_utt_score, gt_utt_score, ret_dict
    

    def __len__(self):
        return len(self.wavnames)

    def get_sys_names(self):
        return self.sysnames
    
    def get_utt_names(self):
        return self.uttnames

    def collate_fn(self, batch):  ## zero padding
        judge_wavnames, wavs, scores, mean_scores, ret_dicts = zip(*batch)

        # padding wavs
        wavs_length = torch.tensor([seq.shape[-1] for seq in wavs])
        collate_wavs = pad_sequence(wavs, padding_mode=self.padding_mode)
        collate_scores = torch.stack(scores)
        collate_mean_scores = torch.stack(mean_scores)
        collate_ret_dict = {}

        if "judge_id" in ret_dicts[0]:
            judge_ids = [rd['judge_id'] for rd in ret_dicts]
            collate_ret_dict["judge_id"] = torch.tensor(judge_ids)
        
        if "domain_id" in ret_dicts[0]:
            domain_ids = [rd['domain_id'] for rd in ret_dicts]
            collate_ret_dict["domain_id"] = torch.tensor(domain_ids)
        
        if "batch_id" in ret_dicts[0]:
            batch_ids = [rd["batch_id"] for rd in ret_dicts]
            collate_ret_dict["batch_id"] = batch_ids

        if "pitch_var" in ret_dicts[0]:
            pitch_var = [rd['pitch_var'] for rd in ret_dicts]
            collate_ret_dict["pitch_var"] = pad_sequence(pitch_var, padding_mode=self.padding_mode)
        
        if "pitch_note" in ret_dicts[0]:
            pitch_note = [rd['pitch_note'] for rd in ret_dicts]
            collate_ret_dict["pitch_note"] = pad_sequence(pitch_note, padding_mode=self.padding_mode)

        if "pitch_histogram" in ret_dicts[0]:
            pitch_histogram = [rd['pitch_histogram'] for rd in ret_dicts]
            collate_ret_dict["pitch_histogram"] = pad_sequence(pitch_histogram, padding_mode=self.padding_mode)
        
        if "pitch" in ret_dicts[0]:
            pitch = [rd['pitch'] for rd in ret_dicts]
            collate_ret_dict["pitch"] = pad_sequence(pitch, padding_mode=self.padding_mode)
        
        return judge_wavnames, collate_wavs, wavs_length, collate_scores, collate_mean_scores, collate_ret_dict


def setup_dataloader_from_DATA(
    config: dict,
    dataset_path: str,
    train_datasets: list = ["singeval_p1"],
    valid_datasets: list = None,
    merge_diff_train: bool = True,
    return_valid_loader: bool = False,
):
    # For SingMOS dataset
    with open(f"{dataset_path}/info/split.json", "r") as f:
        split_info = json.load(f)
    with open(f"{dataset_path}/info/score.json", "r") as f:
        score_info = json.load(f)
    with open(f"{dataset_path}/info/sys_info.json", "r") as f:
        sys_info = json.load(f)
    
    utt_to_batch_id = {}
    for batch_name, payload in split_info.items():
        for split_key in ("train", "valid", "test"):
            for utt in payload.get(split_key, []):
                utt_to_batch_id[utt] = batch_name
    
    if merge_diff_train is False:
        assert len(train_datasets) == 1

    train_list = _collect_utts(split_info, train_datasets, "train")
    
    train_set = MOSDataset(
        datadir=dataset_path,
        utt_list=train_list,
        score_infos=score_info["utterance"],
        sys_info=sys_info,
        utt_to_batch_id=utt_to_batch_id,
        use_domain_id=config["model_param"]["use_domain_id"],
        use_judge_id=config["model_param"]["use_judge_id"],
        use_pitch=config["model_param"]["use_pitch"],
        pitch_type=config["model_param"]["pitch_type"],
        sample_rate=config["sample_rate"],
        max_duration=config["max_duration"],
        padding_mode=config["padding_mode"],
    )

    logging.info(f"train_set: {len(train_set)}")
    
    trainloader = DataLoader(
        train_set, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"],
        collate_fn=train_set.collate_fn,
        pin_memory=True,
    )

    if not return_valid_loader:
        return trainloader

    if valid_datasets is None:
        valid_datasets = train_datasets
    valid_list = _collect_utts(split_info, valid_datasets, "valid")

    valid_set = MOSDataset(
        datadir=dataset_path,
        utt_list=valid_list,
        score_infos=score_info["utterance"],
        sys_info=sys_info,
        utt_to_batch_id=utt_to_batch_id,
        use_domain_id=config["model_param"]["use_domain_id"],
        use_judge_id=config["model_param"]["use_judge_id"],
        use_pitch=config["model_param"]["use_pitch"],
        pitch_type=config["model_param"]["pitch_type"],
        sample_rate=config["sample_rate"],
        max_duration=config["max_duration"],
        padding_mode=config["padding_mode"],
    )
    logging.info(f"valid_set: {len(valid_set)}")

    validloader = DataLoader(
        valid_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=valid_set.collate_fn,
        pin_memory=True,
    )
    return trainloader, validloader