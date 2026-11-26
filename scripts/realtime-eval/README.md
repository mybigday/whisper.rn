# Realtime evaluation harness

Replays a WAV through `RealtimeTranscriber` on the host, with the vendored
whisper.cpp, and scores the final transcript. Positions in the transcriber are
counted in samples, so a replay gives the same slices as live capture of the
same audio.

## Build the server

```sh
cmake -S scripts/realtime-eval -B build-realtime-eval -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build-realtime-eval whisper-server vad-probs
```

`whisper-server` loads a model once and transcribes requests from stdin, using
the JSI layer's `audioCtx: 'auto'` rule and its UTF-8 token grouping.

## Run

```sh
npx ts-node --transpile-only -O '{"module":"commonjs","esModuleInterop":true,"resolveJsonModule":true}' \
  scripts/realtime-eval/run.ts \
  --server build-realtime-eval/whisper-server \
  --model example/assets/ggml-base.bin --dtw base \
  --wav example/assets/jfk.wav \
  --options '{"vadOptions":{"maxSpeechDurationS":6},"splitOverlapMs":400}' \
  --print finals
```

| flag | meaning |
|---|---|
| `--wav`, `--ref` | a 16 kHz mono PCM16 WAV, optional reference text |
| `--manifest`, `--item`, `--root` | an item of a set manifest (`items[].wav`, `items[].ref_src`), with paths relative to `--root` |
| `--options` | `RealtimeOptions` as JSON (drafts are off unless `realtimeProcessingPauseMs` is set: they do not change finals) |
| `--dtw` | alignment-heads preset for DTW token times, `-` for none |
| `--vad`, `--vad-probs` | a Silero model: segment with a VAD, from probabilities `vad-probs` computes for the file (default: next to `--server`) |
| `--hardcut` | split at the limit (`splitAtDip: false`) instead of at a dip |
| `--seconds` | only the first N seconds (drops the reference) |
| `--print finals` / `raw` | each final with its range / raw segments and tokens |
| `--log 1` | the transcriber's log on stderr |

Without `--vad` the audio is sliced by `maxSpeechDurationS` only, which
exercises the max-duration splits (and `splitOverlapMs`); with it, slices
follow the speech as in the app. The report line is
JSON: final count and end reasons, word (or CJK character) error rate against
the reference, and compute time per audio second.
