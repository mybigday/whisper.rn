package com.rnwhisper;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import android.util.Log;
import android.os.Build;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder.AudioSource;

import java.util.ArrayList;
import java.lang.StringBuilder;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;

public class WhisperContext {
  public static final String NAME = "RNWhisperContext";

  private static final int SAMPLE_RATE = 16000;
  private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
  private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
  private static final int AUDIO_SOURCE = AudioSource.VOICE_RECOGNITION;
  private static final int DEFAULT_MAX_AUDIO_SEC = 30;

  private int id;
  private ReactApplicationContext reactContext;
  private long context;
  private int jobId = -1;
  private DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

  private AudioRecord recorder = null;
  private int bufferSize;
  private int nSamplesTranscribing = 0;
  // Remember number of samples in each slice
  private ArrayList<Integer> sliceNSamples;
  // Current buffer slice index
  private int sliceIndex = 0;
  // Current transcribing slice index
  private int transcribeSliceIndex = 0;
  private boolean isUseSlices = false;
  private boolean isRealtime = false;
  private boolean isCapturing = false;
  private boolean isStoppedByAction = false;
  private boolean isTranscribing = false;
  private boolean isTdrzEnable = false;
  private Thread rootFullHandler = null;
  private Thread fullHandler = null;

  public WhisperContext(int id, ReactApplicationContext reactContext, long context) {
    this.id = id;
    this.context = context;
    this.reactContext = reactContext;
    eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
  }

  private void rewind() {
    sliceNSamples = null;
    sliceIndex = 0;
    transcribeSliceIndex = 0;
    isUseSlices = false;
    isRealtime = false;
    isCapturing = false;
    isStoppedByAction = false;
    isTranscribing = false;
    isTdrzEnable = false;
    rootFullHandler = null;
    fullHandler = null;
  }

  private boolean vad(int sliceIndex, int nSamples, int n) {
    if (isTranscribing) return true;
    return vadSimple(jobId, sliceIndex, nSamples, n);
  }

  private void finishRealtimeTranscribe(WritableMap result) {
    emitTranscribeEvent("@RNWhisper_onRealtimeTranscribeEnd", Arguments.createMap());
    finishRealtimeTranscribeJob(jobId, context, sliceNSamples.stream().mapToInt(i -> i).toArray());
  }

  public int startRealtimeTranscribe(int jobId, ReadableMap options) {
    if (isCapturing || isTranscribing) {
      return -100;
    }

    recorder = new AudioRecord(AUDIO_SOURCE, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize);

    int state = recorder.getState();
    if (state != AudioRecord.STATE_INITIALIZED) {
      recorder.release();
      return state;
    }

    rewind();

    this.jobId = jobId;

    int realtimeAudioSec = options.hasKey("realtimeAudioSec") ? options.getInt("realtimeAudioSec") : 0;
    final int audioSec = realtimeAudioSec > 0 ? realtimeAudioSec : DEFAULT_MAX_AUDIO_SEC;
    int realtimeAudioSliceSec = options.hasKey("realtimeAudioSliceSec") ? options.getInt("realtimeAudioSliceSec") : 0;
    final int audioSliceSec = realtimeAudioSliceSec > 0 && realtimeAudioSliceSec < audioSec ? realtimeAudioSliceSec : audioSec;
    isUseSlices = audioSliceSec < audioSec;

    double realtimeAudioMinSec = options.hasKey("realtimeAudioMinSec") ? options.getDouble("realtimeAudioMinSec") : 0;
    final double audioMinSec = realtimeAudioMinSec > 0.5 && realtimeAudioMinSec <= audioSliceSec ? realtimeAudioMinSec : 1;

    this.isTdrzEnable = options.hasKey("tdrzEnable") && options.getBoolean("tdrzEnable");

    createRealtimeTranscribeJob(jobId, context, options);

    sliceNSamples = new ArrayList<Integer>();
    sliceNSamples.add(0);

    isCapturing = true;
    recorder.startRecording();

    rootFullHandler = new Thread(new Runnable() {
      @Override
      public void run() {
        try {
          short[] buffer = new short[bufferSize];
          while (isCapturing) {
            try {
              int n = recorder.read(buffer, 0, bufferSize);
              if (n == 0) continue;

              int totalNSamples = 0;
              for (int i = 0; i < sliceNSamples.size(); i++) {
                totalNSamples += sliceNSamples.get(i);
              }

              int nSamples = sliceNSamples.get(sliceIndex);
              if (totalNSamples + n > audioSec * SAMPLE_RATE) {
                // Full, stop capturing
                isCapturing = false;
                if (
                  !isTranscribing &&
                  nSamples == nSamplesTranscribing &&
                  sliceIndex == transcribeSliceIndex
                ) {
                  finishRealtimeTranscribe(Arguments.createMap());
                } else if (!isTranscribing) {
                  boolean isSamplesEnough = nSamples / SAMPLE_RATE >= audioMinSec;
                  if (!isSamplesEnough || !vad(sliceIndex, nSamples, 0)) {
                    finishRealtimeTranscribe(Arguments.createMap());
                    break;
                  }
                  isTranscribing = true;
                  fullTranscribeSamples(true);
                }
                break;
              }

              // Append to buffer
              if (nSamples + n > audioSliceSec * SAMPLE_RATE) {
                Log.d(NAME, "next slice");

                sliceIndex++;
                nSamples = 0;
                sliceNSamples.add(0);
              }
              putPcmData(jobId, buffer, sliceIndex, nSamples, n);

              boolean isSpeech = vad(sliceIndex, nSamples, n);

              nSamples += n;
              sliceNSamples.set(sliceIndex, nSamples);

              boolean isSamplesEnough = nSamples / SAMPLE_RATE >= audioMinSec;
              if (!isSamplesEnough || !isSpeech) continue;

              if (!isTranscribing && nSamples > SAMPLE_RATE / 2) {
                isTranscribing = true;
                fullHandler = new Thread(new Runnable() {
                  @Override
                  public void run() {
                    fullTranscribeSamples(false);
                  }
                });
                fullHandler.start();
              }
            } catch (Exception e) {
              Log.e(NAME, "Error transcribing realtime: " + e.getMessage());
            }
          }

          if (!isTranscribing) {
            finishRealtimeTranscribe(Arguments.createMap());
          }
          if (fullHandler != null) {
            fullHandler.join(); // Wait for full transcribe to finish
          }
          recorder.stop();
        } catch (Exception e) {
          e.printStackTrace();
        } finally {
          recorder.release();
          recorder = null;
        }
      }
    });
    rootFullHandler.start();
    return state;
  }

  private void fullTranscribeSamples(boolean skipCapturingCheck) {
    int nSamplesOfIndex = sliceNSamples.get(transcribeSliceIndex);

    if (!isCapturing && !skipCapturingCheck) return;

    nSamplesTranscribing = nSamplesOfIndex;
    Log.d(NAME, "Start transcribing realtime: " + nSamplesTranscribing);

    int timeStart = (int) System.currentTimeMillis();
    int code = fullWithJob(jobId, context, transcribeSliceIndex, nSamplesTranscribing);
    int timeEnd = (int) System.currentTimeMillis();
    int timeRecording = (int) (nSamplesTranscribing / SAMPLE_RATE * 1000);

    WritableMap payload = Arguments.createMap();
    payload.putInt("code", code);
    payload.putInt("processTime", timeEnd - timeStart);
    payload.putInt("recordingTime", timeRecording);
    payload.putBoolean("isUseSlices", isUseSlices);
    payload.putInt("sliceIndex", transcribeSliceIndex);

    if (code == 0) {
      payload.putMap("data", getTextSegments(0, getTextSegmentCount(context)));
    } else if (code != -999) { // Not aborted
      payload.putString("error", "Transcribe failed with code " + code);
    }

    nSamplesOfIndex = sliceNSamples.get(transcribeSliceIndex);
    boolean isStopped = isStoppedByAction ||
      !isCapturing &&
      nSamplesTranscribing == nSamplesOfIndex &&
      sliceIndex == transcribeSliceIndex;

    if (
      // If no more samples on current slice, move to next slice
      nSamplesTranscribing == sliceNSamples.get(transcribeSliceIndex) &&
      transcribeSliceIndex != sliceIndex
    ) {
      transcribeSliceIndex++;
      nSamplesTranscribing = 0;
    }

    boolean continueNeeded = !isCapturing && nSamplesTranscribing != nSamplesOfIndex && code != -999;

    if (isStopped && !continueNeeded) {
      payload.putBoolean("isCapturing", false);
      payload.putBoolean("isStoppedByAction", isStoppedByAction);
      finishRealtimeTranscribe(payload);
    } else if (code == 0) {
      payload.putBoolean("isCapturing", true);
      emitTranscribeEvent("@RNWhisper_onRealtimeTranscribe", payload);
    } else {
      payload.putBoolean("isCapturing", true);
      emitTranscribeEvent("@RNWhisper_onRealtimeTranscribe", payload);
    }

    if (continueNeeded) {
      // If no more capturing, continue transcribing until all slices are transcribed
      fullTranscribeSamples(true);
    } else if (isStopped) {
      // No next, cleanup
      rewind();
    }
    isTranscribing = false;
  }

  private void emitTranscribeEvent(final String eventName, final WritableMap payload) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", WhisperContext.this.id);
    event.putInt("jobId", jobId);
    event.putMap("payload", payload);
    eventEmitter.emit(eventName, event);
  }

  private void emitProgress(int progress) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", WhisperContext.this.id);
    event.putInt("jobId", jobId);
    event.putInt("progress", progress);
    eventEmitter.emit("@RNWhisper_onTranscribeProgress", event);
  }

  private void emitNewSegments(WritableMap result) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", WhisperContext.this.id);
    event.putInt("jobId", jobId);
    event.putMap("result", result);
    eventEmitter.emit("@RNWhisper_onTranscribeNewSegments", event);
  }

  private static class Callback {
    WhisperContext context;
    boolean emitProgressNeeded = false;
    boolean emitNewSegmentsNeeded = false;
    int totalNNew = 0;

    public Callback(WhisperContext context, boolean emitProgressNeeded, boolean emitNewSegmentsNeeded) {
      this.context = context;
      this.emitProgressNeeded = emitProgressNeeded;
      this.emitNewSegmentsNeeded = emitNewSegmentsNeeded;
    }

    void onProgress(int progress) {
      if (!emitProgressNeeded) return;
      context.emitProgress(progress);
    }

    void onNewSegments(int nNew) {
      Log.d(NAME, "onNewSegments: " + nNew);
      totalNNew += nNew;
      if (!emitNewSegmentsNeeded) return;

      WritableMap result = context.getTextSegments(totalNNew - nNew, totalNNew);
      result.putInt("nNew", nNew);
      result.putInt("totalNNew", totalNNew);
      context.emitNewSegments(result);
    }
  }

  public WritableMap transcribeInputStream(int jobId, InputStream inputStream, ReadableMap options) throws IOException, Exception {
    if (isCapturing || isTranscribing) {
      throw new Exception("Context is already in capturing or transcribing");
    }
    rewind();
    this.jobId = jobId;
    this.isTdrzEnable = options.hasKey("tdrzEnable") && options.getBoolean("tdrzEnable");

    isTranscribing = true;
    float[] audioData = AudioUtils.decodeWaveFile(inputStream);

    boolean hasProgressCallback = options.hasKey("onProgress") && options.getBoolean("onProgress");
    boolean hasNewSegmentsCallback = options.hasKey("onNewSegments") && options.getBoolean("onNewSegments");
    int code = fullWithNewJob(
      jobId,
      context,
      // float[] audio_data,
      audioData,
      // jint audio_data_len,
      audioData.length,
      // ReadableMap options,
      options,
      // Callback callback
      hasProgressCallback || hasNewSegmentsCallback ? new Callback(this, hasProgressCallback, hasNewSegmentsCallback) : null
    );

    isTranscribing = false;
    this.jobId = -1;
    if (code != 0 && code != 999) {
      throw new Exception("Failed to transcribe the file. Code: " + code);
    }
    WritableMap result = getTextSegments(0, getTextSegmentCount(context));
    result.putBoolean("isAborted", isStoppedByAction);
    return result;
  }

  private WritableMap getTextSegments(int start, int count) {
    StringBuilder builder = new StringBuilder();

    WritableMap data = Arguments.createMap();
    WritableArray segments = Arguments.createArray();

    for (int i = 0; i < count; i++) {
      String text = getTextSegment(context, i);

      // If tdrzEnable is enabled and speaker turn is detected
      if (this.isTdrzEnable && getTextSegmentSpeakerTurnNext(context, i)) {
          text += " [SPEAKER_TURN]";
      }

      builder.append(text);

      WritableMap segment = Arguments.createMap();
      Log.d(NAME, "getTextSegments: " + text + " " + transcribeSliceIndex);
      segment.putString("text", text);
      segment.putInt("t0", getTextSegmentT0(context, i));
      segment.putInt("t1", getTextSegmentT1(context, i));
      segments.pushMap(segment);
    }
    data.putString("result", builder.toString());
    data.putArray("segments", segments);
    return data;
  }


  public boolean isCapturing() {
    return isCapturing;
  }

  public boolean isTranscribing() {
    return isTranscribing;
  }

  public void stopTranscribe(int jobId) {
    abortTranscribe(jobId);
    isCapturing = false;
    isStoppedByAction = true;
    if (rootFullHandler != null) {
      try {
        rootFullHandler.join();
      } catch (Exception e) {
        Log.e(NAME, "Error joining rootFullHandler: " + e.getMessage());
      }
      rootFullHandler = null;
    }
  }

  public void stopCurrentTranscribe() {
    stopTranscribe(this.jobId);
  }

  public String bench(int n_threads) {
    return bench(context, n_threads);
  }

  public void release() {
    stopCurrentTranscribe();
    freeContext(context);
  }

  static {
    Log.d(NAME, "Primary ABI: " + Build.SUPPORTED_ABIS[0]);
    boolean loadVfpv4 = false;
    boolean loadV8fp16 = false;
    if (isArmeabiV7a()) {
      // armeabi-v7a needs runtime detection support
      String cpuInfo = cpuInfo();
      if (cpuInfo != null) {
        Log.d(NAME, "CPU info: " + cpuInfo);
        if (cpuInfo.contains("vfpv4")) {
          Log.d(NAME, "CPU supports vfpv4");
          loadVfpv4 = true;
        }
      }
    } else if (isArmeabiV8a()) {
      // ARMv8.2a needs runtime detection support
      String cpuInfo = cpuInfo();
      if (cpuInfo != null) {
        Log.d(NAME, "CPU info: " + cpuInfo);
        if (cpuInfo.contains("fphp")) {
          Log.d(NAME, "CPU supports fp16 arithmetic");
          loadV8fp16 = true;
        }
      }
    }

    if (loadVfpv4) {
      Log.d(NAME, "Loading libwhisper_vfpv4.so");
      System.loadLibrary("whisper_vfpv4");
    } else if (loadV8fp16) {
      Log.d(NAME, "Loading libwhisper_v8fp16_va.so");
      System.loadLibrary("whisper_v8fp16_va");
    } else {
      Log.d(NAME, "Loading libwhisper.so");
      System.loadLibrary("whisper");
    }
  }

  private static boolean isArmeabiV7a() {
    return Build.SUPPORTED_ABIS[0].equals("armeabi-v7a");
  }

  private static boolean isArmeabiV8a() {
    return Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  private static String cpuInfo() {
    File file = new File("/proc/cpuinfo");
    StringBuilder stringBuilder = new StringBuilder();
    try {
      BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
      String line;
      while ((line = bufferedReader.readLine()) != null) {
          stringBuilder.append(line);
      }
      bufferedReader.close();
      return stringBuilder.toString();
    } catch (IOException e) {
      Log.w(NAME, "Couldn't read /proc/cpuinfo", e);
      return null;
    }
  }

  // JNI methods
  protected static native long initContext(String modelPath);
  protected static native long initContextWithAsset(AssetManager assetManager, String modelPath);
  protected static native long initContextWithInputStream(PushbackInputStream inputStream);
  protected static native void freeContext(long contextPtr);

  protected static native int fullWithNewJob(
    int job_id,
    long context,
    float[] audio_data,
    int audio_data_len,
    ReadableMap options,
    Callback Callback
  );
  protected static native void abortTranscribe(int jobId);
  protected static native void abortAllTranscribe();
  protected static native int getTextSegmentCount(long context);
  protected static native String getTextSegment(long context, int index);
  protected static native int getTextSegmentT0(long context, int index);
  protected static native int getTextSegmentT1(long context, int index);
  protected static native boolean getTextSegmentSpeakerTurnNext(long context, int index);

  protected static native void createRealtimeTranscribeJob(
    int job_id,
    long context,
    ReadableMap options
  );
  protected static native void finishRealtimeTranscribeJob(int job_id, long context, int[] sliceNSamples);
  protected static native boolean vadSimple(int job_id, int slice_index, int n_samples, int n);
  protected static native void putPcmData(int job_id, short[] buffer, int slice_index, int n_samples, int n);
  protected static native int fullWithJob(
    int job_id,
    long context,
    int slice_index,
    int n_samples
  );
  protected static native String bench(long context, int n_threads);
}
