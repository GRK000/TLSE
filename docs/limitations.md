# Limitations

TLSE Studio recognizes isolated static hand poses. It is not a complete Spanish Sign Language translator.

## Linguistic Limitations

- It does not model LSE grammar.
- It does not understand sentence context.
- It does not represent facial expression, body posture, or discourse context.
- It does not handle dynamic signs that require temporal modeling.

## Technical Limitations

- Performance can degrade with poor lighting, motion blur, occlusion, unusual hand angles, low camera quality, or hands outside the guide.
- Accuracy may not transfer to new users, skin tones, backgrounds, cameras, or capture environments.
- Model confidence may be uncalibrated.
- The current artifact format uses trusted local pickle files.

## Evaluation Limitations

- Random frame splits can overestimate performance.
- Reliable evaluation requires group/session/person splits.
- Synthetic demo metrics only validate the software workflow, not real-world recognition quality.

## Privacy

TLSE Studio is designed for local processing by default. Future cloud features should be opt-in and explicitly documented.
