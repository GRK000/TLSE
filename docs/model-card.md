# Model Card

## Model

The maintained baseline is a scikit-learn Random Forest trained on normalized static hand landmark features.

## Intended Use

- Recognize isolated static hand poses.
- Provide confidence-ranked predictions in local demos.
- Support experimentation with dataset balance and grouped evaluation.

## Not Recommended For

- Full Spanish Sign Language translation.
- Dynamic signs or multi-frame phrases.
- Safety-critical accessibility decisions.
- Deployment without evaluation on representative users, sessions, and camera conditions.

## Input

`depth47-v1`: normalized 2D MediaPipe hand landmarks plus fingertip depth features. The current vector length is 47.

## Output

The model returns class probabilities. TLSE Studio displays the top predictions and applies a confidence threshold before accepting a sign.

## Metrics

The API reads available JSON metrics from `reports/`. Expected fields include:

- accuracy
- macro_f1
- precision macro
- recall macro
- num_classes
- num_samples
- train/test sample counts

Synthetic demo scores are smoke-test evidence only and must not be reported as real-world sign recognition quality.

## Dataset

Training data should include class labels and, when possible, session/person/group metadata. Grouped evaluation is preferred because random frame splits can leak nearly identical frames across train and test sets.

## Limitations

- Probabilities may be uncalibrated.
- Performance depends on lighting, framing, camera quality, skin tone, background, hand orientation, and signer variation.
- MediaPipe detection can fail with occlusion, blur, or hands outside the guide region.
- The model does not understand grammar, meaning, or context.
