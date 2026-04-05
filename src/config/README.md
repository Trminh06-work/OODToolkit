# Model Variant Configs

Create one JSON file per model in this folder. The filename must match the model class name.

Example: `config/ResnetRegressor.json`

```json
{
  "variants": {
    "baseline": {
      "runtime_config": {
        "seed": 42,
        "metric": "rmse",
        "use_optim": false,
        "n_splits": 5
      },
      "model_params": {
        "d": 512,
        "n_res_blocks": 4,
        "dropout_rate": 0.2,
        "epochs": 100,
        "batch_size": 1024
      }
    },
    "wide_deep": {
      "runtime_config": {
        "seed": 42
      },
      "model_params": {
        "d": 768,
        "n_res_blocks": 6,
        "dropout_rate": 0.1,
        "epochs": 150,
        "batch_size": 512
      }
    }
  }
}
```

Notes:

- `runtime_config` maps to `models.base_model.ModelConfig`.
- `model_params` are passed directly to the model constructor.
- If a model has no matching JSON file, the toolkit runs a single `default` variant.
- Results are saved to `Results/<ModelName>/<variant_name>/`.
- Each variant folder also includes `_variant.json` with the exact settings that were used.
