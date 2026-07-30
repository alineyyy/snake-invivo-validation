import numpy as np
import argparse
import os
import logging
from datetime import datetime


def setup_logger(output_dir, data_dir, task_id=0):
    base_name = os.path.basename(data_dir)
    tag = "unknown"
    if "tra_" in base_name and base_name.endswith(".dat"):
        tag = base_name.split("tra_")[-1].split(".dat")[0]

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    final_dir = os.path.join(output_dir, f"{date_str}_{tag}")

    os.makedirs(final_dir, exist_ok=True)

    log_path = os.path.join(final_dir, f"reconstruction_{task_id}.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Created output directory: {final_dir}")
    logging.info("Logger setup complete.")

    return final_dir, log_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct fully sampled fMRI")
    parser.add_argument("--traj_dir", type=str, required=True, help="Path to trajectory file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to k-space data file")
    parser.add_argument("--smaps_dir", type=str, required=True, help="Path to sensitivity maps file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the reconstructed data")
    parser.add_argument("--n_shot_per_frames", type=int, required=True, help="Number of shots for each frame")
    parser.add_argument("--frames_per_task", type=int, default=10, help="Number of frames to process per task")
    parser.add_argument("--reconstructor", type=str, required=True, help="Type of reconstructor")
    parser.add_argument("--restart_strategy", type=str, required=True, help="Type of restart strategy")
    parser.add_argument("--init_strategy", type=str, required=True,help="Type of init strategy")
    args = parser.parse_args()

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    frames_per_task = args.frames_per_task
    start_frame = task_id * frames_per_task
    end_frame = start_frame + frames_per_task 
    
    logging.info(f"[INFO] Task ID: {task_id}, Processing frames {start_frame} to {end_frame}")
    from siemens_loader import SiemensDataLoader
    dat_file = args.data_dir
    bin_file = args.traj_dir
    smaps_file = args.smaps_dir
    traj_args = {
        "dwell_time": 0.002,
        "raster_time": 0.01,
    }
    loader = SiemensDataLoader(dat_file, 
                               bin_file, 
                               smaps_file, 
                               n_shot_per_frames=args.n_shot_per_frames, 
                               start_idx=start_frame, 
                               **traj_args)
    sim_conf = loader.get_sim_conf()

    from snake.toolkit.reconstructors import (
    MergeGlobalReconstructor,
    SequentialReconstructor,
    )
    from snake.toolkit.reconstructors.pysap import RestartStrategy
    import numpy as np
    if args.restart_strategy == "cold":
        restart_strategy = RestartStrategy.COLD
    elif args.restart_strategy == "refine":
        restart_strategy = RestartStrategy.REFINE
    if args.reconstructor == "sequential":
        recon = SequentialReconstructor(
            max_iter_per_frame=30,
            density_compensation=True, 
            restart_strategy=restart_strategy,
            init_strategy=args.init_strategy,
            wavelet="sym4",
            optimizer="fista", 
            threshold=1e-10,
            compute_backend="numpy"
        ).reconstruct(loader)
    elif args.reconstructor == "cg":
        recon = MergeGlobalReconstructor(
            max_iter=30,
            tol=1e-10,
            density_compensation=True,
            restart_strategy=args.init_strategy,
        ).reconstruct(loader)

    final_dir, log_path = setup_logger(args.output_dir, args.data_dir, task_id)
    save_path = os.path.join(final_dir, f"reconstructed_{task_id}.npy")
    np.save(save_path, recon)
    logging.info(f"Reconstruction completed. Data saved to {save_path}")
