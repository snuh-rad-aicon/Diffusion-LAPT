import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import ants
# # add path 
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=48
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    # opt = parser.parse_args()
    opt, unknown = parser.parse_known_args()
    
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")
        
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    outdir = os.path.join(logdir, "samples", f"{opt.steps}")
    os.makedirs(outdir, exist_ok=True)
    
    config.data['params']['batch_size'] = opt.batch_size
    config.data['params']['test'] =  config.data['params']['validation']
    
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    test_dataset = data._test_dataloader().dataset
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size,
                          num_workers=data.num_workers, shuffle=False)
    
    gpu = True
    eval_mode = True
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    sampler = DDIMSampler(model)
    
    cond_stage_key = config.model.params.cond_stage_key
    sample_results = {}
    with torch.no_grad():
        with model.ema_scope():
            for test_batch in tqdm(test_loader):
                data_id = test_batch['data_id']
                slice = test_batch['slice']
                x = test_batch['mri']
                y = test_batch[cond_stage_key].to(model.device)
                batch_size = x.shape[0]
                
                c = model.get_learned_conditioning(y)
                for j in range(opt.n_samples):
                    sample, intermediates = model.sample_log(c, batch_size, ddim=True, ddim_steps=opt.steps)
                    x_sample = model.decode_first_stage(sample)
                    for i in range(batch_size):
                        if data_id[i] not in sample_results:
                            sample_results[data_id[i]] = {}
                        if j not in sample_results[data_id[i]]:
                            sample_results[data_id[i]][int(j)] = {}
                        sample_results[data_id[i]][int(j)][slice[i]] = [x_sample[i].detach().cpu().numpy(), x[i].numpy(), y[i].detach().cpu().numpy()]

    for data_id in sample_results:
        data_id_result_path = os.path.join(outdir, f'{data_id}')
        os.makedirs(data_id_result_path, exist_ok=True)
        for j in range(opt.n_samples):
            sample_result_dict = sample_results[data_id][j]
            sample = [v[0] for k, v in sample_result_dict.items()]
            sample = np.array(sample)
            sample = sample.transpose(1, 2, 3, 0)
            nifti_dwi_sample = ants.from_numpy(sample[0])
            nifti_dwi_sample.to_file(os.path.join(data_id_result_path, f'dwi_sample_{j}.nii.gz'))
            nifti_adc_sample = ants.from_numpy(sample[1])
            nifti_adc_sample.to_file(os.path.join(data_id_result_path, f'adc_sample_{j}.nii.gz'))
            