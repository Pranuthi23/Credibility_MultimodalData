import os 

def load_from_checkpoint(results_dir, load_fn, args):
    """Loads the model from a checkpoint.

    Args:
        load_fn: The function to load the model from a checkpoint.
    Returns:
        The loaded model.
    """
    ckpt_dir = os.path.join(results_dir, "tb", "version_0", "checkpoints")
    files = os.listdir(ckpt_dir)
    assert len(files) > 0, "Checkpoint directory is empty"
    ckpt_path = os.path.join(ckpt_dir, files[-1])
    model = load_fn(checkpoint_path=ckpt_path, args=args)
    return model


