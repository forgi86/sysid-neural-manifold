import optax

def create_learning_rate_fn(config):
  """Creates learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=config.lr,
      transition_steps=config.warmup_iters)
  
  cosine_fn = optax.cosine_decay_schedule(
      init_value=config.lr,
      decay_steps=config.iters,
      alpha=0.1
  )
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_iters])
  return schedule_fn