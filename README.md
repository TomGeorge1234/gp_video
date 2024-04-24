Generates temporal and spatial autocorrelated random videos using Gaussian processes

1. Clone this repo onto the cluster 
2. Make sure your environment has `scipy` and `numpy`
3. Run `python slurm_maker.py` to generate the slurm file
4. Run `sbatch slurm_script.sh` to submit the job to the cluster. This makes repeated calls to `gp_videos.py` which sample the a video, given some parameters, and saves it. Each parameter set will be saved as its own `.npz` file inside `\results\ddmmyyy_hhmm\<randomhash>.npz`. There are four keys inside ech  file: `results` (the N_x x N_y x N_t array of results), `L_idx` (the index identifying which length scale was used), `T_idx` (the index identifying which time scale was used), and `id` (the index identifying which "repeat" this was).
5. `gp_videos.ipynb` will process loop through the reuslts directory and process the `.npz` files into a single (N_x x N_y x N_t x N_repeat x N_L x N_T) array. This array is then saved as a `.npz` is then saved as `video_data.npz`. This script also generates an animation of the videos.

<video width="600" height="600" controls>
  <source src="video.mp4" type="video/mp4">