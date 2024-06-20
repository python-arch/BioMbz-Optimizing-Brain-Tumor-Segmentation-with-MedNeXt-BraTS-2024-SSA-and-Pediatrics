import pytorch_lightning as pl

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    mode='min',
    monitor='val_loss',
    verbose=True,
    save_last=True,
    save_top_k=1,
    save_weights_only=False,
)

callback_save_last_only = pl.callbacks.ModelCheckpoint(
    monitor=None,
    verbose=True,
    save_last=True,
    save_weights_only=False,
)

lr_monitor = pl.callbacks.LearningRateMonitor(
    logging_interval='step', log_momentum=True,
)
