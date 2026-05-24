# Dataset Card

## Structure

The current processed dataset format is a trusted local pickle dictionary:

```python
{
    "data": numpy.ndarray,
    "labels": numpy.ndarray,
    "class_names": list[str],
    "groups": numpy.ndarray,
}
```

## Labels

Labels represent isolated static signs or hand poses. The `None` class may be used as a practical recognition state, but it is not a linguistic sign.

## Sessions And Users

Future captures should store:

- session id
- optional user id
- camera/device
- lighting notes
- capture timestamp
- split group

## Quality Checks

Dataset Studio surfaces balance, low-sample classes, session counts, and split warnings. Future quality checks should include blur, hand detection stability, lighting range, duplicate frames, and signer coverage.

## Split Strategy

Prefer group/session/person splits. Random frame splits can overestimate performance because adjacent frames from the same capture can be nearly identical.

## Risks

- Leakage from similar frames across train and test sets.
- Class imbalance that inflates accuracy.
- Bias from too few signers, skin tones, cameras, backgrounds, or lighting setups.
- Overfitting to the hand guide position or background rather than the sign.
