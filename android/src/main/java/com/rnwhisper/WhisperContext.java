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

  private static String loadedLibrary = "";

  private static class NativeLogCallback {
    DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

    public NativeLogCallback(ReactApplicationContext reactContext) {
      this.eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    }

    void emitNativeLog(String level, String text) {
      WritableMap event = Arguments.createMap();
      event.putString("level", level);
      event.putString("text", text);
      eventEmitter.emit("@RNWhisper_onNativeLog", event);
    }
  }

  static void toggleNativeLog(ReactApplicationContext reactContext, boolean enabled) {
    if (enabled) {
      setupLog(new NativeLogCallback(reactContext));
    } else {
      unsetLog();
    }
  }

  private static final int SAMPLE_RATE = 16000;

  private int id;
  private ReactApplicationContext reactContext;
  private long context;
  private int jobId = -1;
  private DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

  private boolean isTranscribing = false;

  public WhisperContext(int id, ReactApplicationContext reactContext, long context) {
    this.id = id;
    this.context = context;
    this.reactContext = reactContext;
    eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
  }

  private void rewind() {
    isTranscribing = false;
  }

  public int getId() {
    return id;
  }

  public long getContextPtr() {
    return context;
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
    ReadableMap options;

    public Callback(WhisperContext context, ReadableMap options, boolean emitProgressNeeded, boolean emitNewSegmentsNeeded) {
      this.context = context;
      this.emitProgressNeeded = emitProgressNeeded;
      this.emitNewSegmentsNeeded = emitNewSegmentsNeeded;
      this.options = options;
    }

    void onProgress(int progress) {
      if (!emitProgressNeeded) return;
      context.emitProgress(progress);
    }

    void onNewSegments(int nNew) {
      Log.d(NAME, "onNewSegments: " + nNew);
      totalNNew += nNew;
      if (!emitNewSegmentsNeeded) return;

      WritableMap result = context.getTextSegments(options, totalNNew - nNew, totalNNew);
      result.putInt("nNew", nNew);
      result.putInt("totalNNew", totalNNew);
      context.emitNewSegments(result);
    }
  }

  public WritableMap transcribe(int jobId, float[] audioData, ReadableMap options) throws IOException, Exception {
    if (isTranscribing) {
      throw new Exception("Context is already in capturing or transcribing");
    }
    rewind();
    this.jobId = jobId;

    isTranscribing = true;

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
      hasProgressCallback || hasNewSegmentsCallback ? new Callback(this, options, hasProgressCallback, hasNewSegmentsCallback) : null
    );

    isTranscribing = false;
    this.jobId = -1;
    if (code != 0 && code != 999) {
      throw new Exception("Failed to transcribe the file. Code: " + code);
    }
    WritableMap result = getTextSegments(options, 0, getTextSegmentCount(context));
    return result;
  }

  private WritableMap getTextSegments(ReadableMap options, int start, int count) {
    StringBuilder builder = new StringBuilder();

    WritableMap data = Arguments.createMap();
    WritableArray segments = Arguments.createArray();

    for (int i = 0; i < count; i++) {
      String text = getTextSegment(context, i);

      // If tdrzEnable is enabled and speaker turn is detected
      if (options.getBoolean("tdrzEnable") && getTextSegmentSpeakerTurnNext(context, i)) {
          text += " [SPEAKER_TURN]";
      }

      builder.append(text);

      WritableMap segment = Arguments.createMap();
      segment.putString("text", text);
      segment.putInt("t0", getTextSegmentT0(context, i));
      segment.putInt("t1", getTextSegmentT1(context, i));
      segments.pushMap(segment);
    }
    data.putString("result", builder.toString());
    data.putArray("segments", segments);
    return data;
  }

  public boolean isTranscribing() {
    return isTranscribing;
  }

  public void stopTranscribe(int jobId) {
    abortTranscribe(jobId);
    isTranscribing = false;
    this.jobId = -1;
  }

  public void stopCurrentTranscribe() {
    stopTranscribe(this.jobId);
  }

  public String bench(int n_threads) {
    return bench(context, n_threads);
  }

  public void release() {
    stopCurrentTranscribe();
    freeContext(id, context);
  }

  static {
    Log.d(NAME, "Primary ABI: " + Build.SUPPORTED_ABIS[0]);

    String cpuFeatures = WhisperContext.getCpuFeatures();
    Log.d(NAME, "CPU features: " + cpuFeatures);
    boolean hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");
    Log.d(NAME, "- hasFp16: " + hasFp16);

    if (WhisperContext.isArm64V8a()) {
      if (hasFp16) {
        Log.d(NAME, "Loading librnwhisper_v8fp16_va_2.so");
        System.loadLibrary("rnwhisper_v8fp16_va_2");
        loadedLibrary = "rnwhisper_v8fp16_va_2";
      } else {
        Log.d(NAME, "Loading librnwhisper_v8.so");
        System.loadLibrary("rnwhisper_v8");
        loadedLibrary = "rnwhisper_v8";
      }
    } else if (WhisperContext.isArmeabiV7a()) {
      Log.d(NAME, "Loading librnwhisper_vfpv4.so");
      System.loadLibrary("rnwhisper_vfpv4");
      loadedLibrary = "rnwhisper_vfpv4";
    } else if (WhisperContext.isX86_64()) {
      Log.d(NAME, "Loading librnwhisper_x86_64.so");
      System.loadLibrary("rnwhisper_x86_64");
      loadedLibrary = "rnwhisper_x86_64";
    } else {
      Log.d(NAME, "ARM32 is not supported, skipping loading library");
    }
  }

  public static boolean isNativeLibraryLoaded() {
    return loadedLibrary != "";
  }

  public static boolean isArm64V8a() {
    return Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  public static boolean isArmeabiV7a() {
    return Build.SUPPORTED_ABIS[0].equals("armeabi-v7a");
  }

  public static boolean isX86_64() {
    return Build.SUPPORTED_ABIS[0].equals("x86_64");
  }

  public static String getCpuFeatures() {
    File file = new File("/proc/cpuinfo");
    StringBuilder stringBuilder = new StringBuilder();
    try {
      BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
      String line;
      while ((line = bufferedReader.readLine()) != null) {
        if (line.startsWith("Features")) {
          stringBuilder.append(line);
          break;
        }
      }
      bufferedReader.close();
      return stringBuilder.toString();
    } catch (IOException e) {
      Log.w(NAME, "Couldn't read /proc/cpuinfo", e);
      return "";
    }
  }

  public static String getLoadedLibrary() {
    return loadedLibrary;
  }

  // JNI methods
  protected static native long initContext(int contextId, String modelPath);
  protected static native long initContextWithAsset(int contextId, AssetManager assetManager, String modelPath);
  protected static native long initContextWithInputStream(int contextId, PushbackInputStream inputStream);
  protected static native void freeContext(int contextId, long contextPtr);

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

  protected static native String bench(long context, int n_threads);

  // VAD JNI methods
  protected static native long initVadContext(int contextId, String modelPath);
  protected static native long initVadContextWithAsset(int contextId, AssetManager assetManager, String modelPath);
  protected static native long initVadContextWithInputStream(int contextId, PushbackInputStream inputStream);
  protected static native void freeVadContext(int contextId, long vadContextPtr);
  protected static native boolean vadDetectSpeech(long vadContextPtr, float[] audioData, int nSamples);
  protected static native long vadGetSegmentsFromProbs(long vadContextPtr, float threshold,
                                                       int minSpeechDurationMs, int minSilenceDurationMs,
                                                       float maxSpeechDurationS, int speechPadMs,
                                                       float samplesOverlap);
  protected static native int vadGetNSegments(long segmentsPtr);
  protected static native float vadGetSegmentT0(long segmentsPtr, int index);
  protected static native float vadGetSegmentT1(long segmentsPtr, int index);
  protected static native void vadFreeSegments(long segmentsPtr);

  // Audio file loading utility for VAD
  public static float[] loadAudioFileAsFloat32(String filePath) {
    try {
      java.io.FileInputStream fis = new java.io.FileInputStream(new java.io.File(filePath));
      return AudioUtils.decodeWaveFile(fis);
    } catch (Exception e) {
      Log.e(NAME, "Failed to load audio file: " + filePath, e);
      return null;
    }
  }

  // JSI Installation
  protected static native void installJSIBindings(long runtimePtr, Object callInvokerHolder);
  protected static native void cleanupJSIBindings();
  protected static native void setupLog(NativeLogCallback logCallback);
  protected static native void unsetLog();
}
