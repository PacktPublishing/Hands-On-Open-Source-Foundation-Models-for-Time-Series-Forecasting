loss = DistributionLoss(distribution='Normal', level=[80, 95],
                        num_samples=N_SAMPLES, return_params=True)
model = NHITS(h=HORIZON, input_size=192, loss=loss,
              max_steps=max_steps, val_check_steps=50,
              early_stop_patience_steps=-1, learning_rate=1e-3,
              scaler_type='standard', random_seed=SEED,
              batch_size=1, windows_batch_size=64,
              n_blocks=[1, 1, 1], mlp_units=[[128, 128]] * 3,
              accelerator='cpu', devices=1, logger=False,
              enable_checkpointing=False, enable_progress_bar=False,
              alias='NHITS_Normal')
nf = NeuralForecast(models=[model], freq='h')
nf.fit(df=frame.iloc[:VAL_END], val_size=VAL_END-TRAIN_END)
