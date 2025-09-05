# config.py

train_config = {
    "data_dir": "./data",
    "checkpoints_dir": "./checkpoints",
    "reload_checkpoint": None,

    "img_height": 32,
    "img_width": 128,
    "img_channel": 1,

    "epochs": 20,
    "train_batch_size": 64,
    "val_batch_size": 64,  # <-- добавлено
    "lr": 1e-3,
    "cpu_workers": 4,

    "show_interval": 50,
    "valid_interval": 200,
    "save_interval": 1000,

    "map_to_seq_hidden": 64,
    "rnn_hidden": 256,
    "leaky_relu": False
}
