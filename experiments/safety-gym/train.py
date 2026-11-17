"""Install experiment-only disk logging controls, then run the stock Gym entrypoint."""
import runpy
import storage_logging

storage_logging.install_driver()
runpy.run_path('/opt/nemo-rl/examples/nemo_gym/run_grpo_nemo_gym.py', run_name='__main__')
