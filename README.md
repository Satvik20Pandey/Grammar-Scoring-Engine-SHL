**Made by - Satvik Pandey**

# Grammar Scoring Engine for Spoken English

Solution to the SHL Hiring Assessment 2026 Kaggle challenge: given a 45 to 60 second spoken answer, predict
the grammar score (0 to 5) that human raters assigned using the competition rubric.

**Public leaderboard: 0.3516 RMSE** (Pearson ≈ 0.93 with the public test labels).

## How it works

```
audio ─┬─ Whisper large-v3 ─ transcript + word timings ─┬─ fluency & complexity statistics
       │                                                └─ rubric grades from gpt-oss-120b
       └─ WavLM-large · Whisper encoder · HuBERT-large ─ per-layer mean/std + attention heads
                                         │
                       one small model per feature group (ridge, kernel ridge)
                                         │
                       non-negative linear stack of out-of-fold predictions ─ score
```

1. **Transcription.** Whisper large-v3, prompted to keep fillers and restarts, with word timestamps.
2. **Transcript features.** Speech rate, pauses, fillers, repeats, sentence length, vocabulary range, plus an
   LLM grade, error rate and coherence rating against the official rubric (cost capped, about $0.55 per pass).
3. **Audio features.** Three pretrained speech models, every layer pooled by mean and std, and small
   attention-pooling heads trained on frame sequences.
4. **Ensemble.** One regularised model per feature group, blended with a non-negative linear stacker.

## The finding that mattered

The training set contains the same speakers many times with near-identical scores, while the test speakers
are all new. Under a random split, audio models simply recognise voices: an SVR on embeddings looked like the
best model (0.51 RMSE) and collapsed to 0.90 on unseen speakers. Clips were grouped into 413 speakers by
acoustic similarity, and every model and decision was validated with speaker-grouped folds. That change alone
moved the leaderboard from 0.3695 to 0.3598, and the extra audio models then brought it to 0.3516.

| Validation (unseen speakers) | RMSE | Pearson |
|---|---|---|
| Best single model (Whisper encoder, ridge) | 0.565 | 0.83 |
| Transcript features only | 0.695 | 0.73 |
| **Stacked ensemble** | **0.535** | **0.85** |

## Repository

```
notebooks/shl_GSE_Satvik.ipynb   full pipeline, evaluation, visualisations and report
submission.csv                   final predictions for the 216 test clips
requirements.txt
```

## Running

Built for a Kaggle notebook with 2× T4 GPUs and internet enabled. Attach the competition data and add a
Kaggle secret `GROQ_API_KEY` for the LLM grading. Every heavy stage caches its output, so attaching a
previous run's output makes a rerun take minutes instead of about two hours.

Competition data is not included, as required by the competition rules.
