pipeline = ChronosPipeline.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION,
    device_map='cpu', torch_dtype=torch.float32)
outputs = []
for origin in ORIGINS:
    torch.manual_seed(SEED + origin)
    context = torch.tensor(frame.y.iloc[origin-512:origin].to_numpy(), dtype=torch.float32)
    # Native horizon is 64; longer requests use the documented median-chunk extension.
    pred = pipeline.predict(context, prediction_length=HORIZON,
                            num_samples=N_SAMPLES, limit_prediction_length=False,
                            temperature=1.0, top_k=50, top_p=1.0)
    samples = pred[0].detach().cpu().numpy()
