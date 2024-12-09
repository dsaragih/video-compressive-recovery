checkpoint_config = dict(interval=5)

log_config = dict(
    interval=100,
)
save_image_config = dict(
    interval=2000,
)
optimizer = dict(type='Adam', lr=0.0001)

loss = dict(type='MSELoss')

runner = dict(max_epochs=50)

checkpoints=None
resume=None