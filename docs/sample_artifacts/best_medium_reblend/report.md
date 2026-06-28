# Numerai Research Report: reblend_walkforward_v5.2_20260628T151950Z_20260628T160150Z

## Summary

- mean_corr: 0.080321
- std_corr: 0.083930
- sharpe_like: 0.957002
- max_drawdown: 0.000000
- min_corr: 0.001655
- max_corr: 0.243436
- mean_abs_feature_exposure: 0.013631
- max_abs_feature_exposure: 0.094832
- num_eras: 12

## Fold Results

| model_name | mean_corr | sharpe_like | max_drawdown | mean_abs_feature_exposure |
| --- | ---: | ---: | ---: | ---: |
| ensemble | 0.081136 | 0.979448 | 0.000000 | 0.020476 |
| ensemble_neutralized | 0.080321 | 0.966310 | 0.000000 | 0.015597 |
| catboost_optional | 0.079392 | 0.928660 | -0.000575 | 0.018455 |
| lgbm_main | 0.070495 | 1.045938 | 0.000000 | 0.019335 |
| lgbm_alt | 0.052972 | 0.977822 | -0.001288 | 0.020490 |

## Top Feature Exposures

| feature | abs_exposure |
| --- | ---: |
| feature_aliunde_unhaunted_coacervate | 0.094832 |
| feature_anamorphic_hidden_termagant | 0.066933 |
| feature_appendicular_multidisciplinary_cuisse | 0.061532 |
| feature_australian_unstigmatized_pasch | 0.059409 |
| feature_accoutered_revolute_vexillology | 0.058473 |
| feature_bacilliform_solomonic_exotoxin | 0.056717 |
| feature_boozier_multinucleate_muley | 0.055584 |
| feature_boyish_correlate_haley | 0.054636 |
| feature_advised_subdural_toadflax | 0.053488 |
| feature_afferent_splenic_rosaniline | 0.053137 |
| feature_blindfold_putrid_grill | 0.052891 |
| feature_attenuate_several_hydrofoil | 0.051633 |
| feature_bombproof_shockable_detumescence | 0.051252 |
| feature_aery_restrainable_thai | 0.045886 |
| feature_awash_reverberative_isatin | 0.045697 |

## Notes

- This report comes from walk-forward era validation, not random splitting.
- Validation metrics are useful but still vulnerable to repeated tuning and leakage.
- Neutralization here is a practical post-processing step, not a guarantee of higher live performance.