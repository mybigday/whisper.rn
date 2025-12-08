package com.rnwhisper;

import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder.AudioSource;
import android.os.Build;
import android.util.Log;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;
import java.lang.StringBuilder;
import java.util.ArrayList;
import java.util.regex.Pattern;

public class WhisperContext {

  public static final String NAME = "RNWhisperContext";

  private static String loadedLibrary = "";
  private static final int HTP_DIR_MODE = 0755;
  private static final int HTP_FILE_MODE = 0755;
  private static final String HTP_DIR_NAME = "rnwhisper-htp";
  private static final String[] HTP_LIBS = {
    "libggml-htp-v73.so",
    "libggml-htp-v75.so",
    "libggml-htp-v79.so",
    "libggml-htp-v81.so",
  };

  private static class NativeLogCallback {

    DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

    public NativeLogCallback(ReactApplicationContext reactContext) {
      this.eventEmitter = reactContext.getJSModule(
        DeviceEventManagerModule.RCTDeviceEventEmitter.class
      );
    }

    void emitNativeLog(String level, String text) {
      WritableMap event = Arguments.createMap();
      event.putString("level", level);
      event.putString("text", text);
      eventEmitter.emit("@RNWhisper_onNativeLog", event);
    }
  }

  static void toggleNativeLog(
    ReactApplicationContext reactContext,
    boolean enabled
  ) {
    if (enabled) {
      setupLog(new NativeLogCallback(reactContext));
    } else {
      unsetLog();
    }
  }

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

  public WhisperContext(
    int id,
    ReactApplicationContext reactContext,
    long context
  ) {
    this.id = id;
    this.context = context;
    this.reactContext = reactContext;
    eventEmitter = reactContext.getJSModule(
      DeviceEventManagerModule.RCTDeviceEventEmitter.class
    );
    bufferSize = AudioRecord.getMinBufferSize(
      SAMPLE_RATE,
      CHANNEL_CONFIG,
      AUDIO_FORMAT
    );
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

  public int getId() {
    return id;
  }

  public long getContextPtr() {
    return context;
  }

  private boolean vad(int sliceIndex, int nSamples, int n) {
    if (isTranscribing) return true;
    return vadSimple(jobId, sliceIndex, nSamples, n);
  }

  private void finishRealtimeTranscribe(WritableMap result) {
    emitTranscribeEvent(
      "@RNWhisper_onRealtimeTranscribeEnd",
      Arguments.createMap()
    );
    finishRealtimeTranscribeJob(
      jobId,
      context,
      sliceNSamples
        .stream()
        .mapToInt(i -> i)
        .toArray()
    );
  }

  public int startRealtimeTranscribe(int jobId, ReadableMap options) {
    if (isCapturing || isTranscribing) {
      return -100;
    }

    recorder = new AudioRecord(
      AUDIO_SOURCE,
      SAMPLE_RATE,
      CHANNEL_CONFIG,
      AUDIO_FORMAT,
      bufferSize
    );

    int state = recorder.getState();
    if (state != AudioRecord.STATE_INITIALIZED) {
      recorder.release();
      return state;
    }

    rewind();

    this.jobId = jobId;

    int realtimeAudioSec = options.hasKey("realtimeAudioSec")
      ? options.getInt("realtimeAudioSec")
      : 0;
    final int audioSec = realtimeAudioSec > 0
      ? realtimeAudioSec
      : DEFAULT_MAX_AUDIO_SEC;
    int realtimeAudioSliceSec = options.hasKey("realtimeAudioSliceSec")
      ? options.getInt("realtimeAudioSliceSec")
      : 0;
    final int audioSliceSec = realtimeAudioSliceSec > 0 &&
      realtimeAudioSliceSec < audioSec
      ? realtimeAudioSliceSec
      : audioSec;
    isUseSlices = audioSliceSec < audioSec;

    double realtimeAudioMinSec = options.hasKey("realtimeAudioMinSec")
      ? options.getDouble("realtimeAudioMinSec")
      : 0;
    final double audioMinSec = realtimeAudioMinSec > 0.5 &&
      realtimeAudioMinSec <= audioSliceSec
      ? realtimeAudioMinSec
      : 1;

    this.isTdrzEnable =
      options.hasKey("tdrzEnable") && options.getBoolean("tdrzEnable");

    createRealtimeTranscribeJob(jobId, context, options);

    sliceNSamples = new ArrayList<Integer>();
    sliceNSamples.add(0);

    isCapturing = true;
    recorder.startRecording();

    rootFullHandler = new Thread(
      new Runnable() {
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
                    boolean isSamplesEnough =
                      nSamples / SAMPLE_RATE >= audioMinSec;
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
                  fullHandler = new Thread(
                    new Runnable() {
                      @Override
                      public void run() {
                        fullTranscribeSamples(false);
                      }
                    }
                  );
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
      }
    );
    rootFullHandler.start();
    return state;
  }

  private void fullTranscribeSamples(boolean skipCapturingCheck) {
    int nSamplesOfIndex = sliceNSamples.get(transcribeSliceIndex);

    if (!isCapturing && !skipCapturingCheck) return;

    nSamplesTranscribing = nSamplesOfIndex;
    Log.d(NAME, "Start transcribing realtime: " + nSamplesTranscribing);

    int timeStart = (int) System.currentTimeMillis();
    int code = fullWithJob(
      jobId,
      context,
      transcribeSliceIndex,
      nSamplesTranscribing
    );
    int timeEnd = (int) System.currentTimeMillis();
    int timeRecording = (int) ((nSamplesTranscribing / SAMPLE_RATE) * 1000);

    WritableMap payload = Arguments.createMap();
    payload.putInt("code", code);
    payload.putInt("processTime", timeEnd - timeStart);
    payload.putInt("recordingTime", timeRecording);
    payload.putBoolean("isUseSlices", isUseSlices);
    payload.putInt("sliceIndex", transcribeSliceIndex);

    if (code == 0) {
      payload.putMap("data", getTextSegments(0, getTextSegmentCount(context)));
    } else if (code != -999) {
      // Not aborted
      payload.putString("error", "Transcribe failed with code " + code);
    }

    nSamplesOfIndex = sliceNSamples.get(transcribeSliceIndex);
    boolean isStopped =
      isStoppedByAction ||
      (!isCapturing &&
        nSamplesTranscribing == nSamplesOfIndex &&
        sliceIndex == transcribeSliceIndex);

    if (
      // If no more samples on current slice, move to next slice
      nSamplesTranscribing == sliceNSamples.get(transcribeSliceIndex) &&
      transcribeSliceIndex != sliceIndex
    ) {
      transcribeSliceIndex++;
      nSamplesTranscribing = 0;
    }

    boolean continueNeeded =
      !isCapturing && nSamplesTranscribing != nSamplesOfIndex && code != -999;

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

  private void emitTranscribeEvent(
    final String eventName,
    final WritableMap payload
  ) {
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

    public Callback(
      WhisperContext context,
      boolean emitProgressNeeded,
      boolean emitNewSegmentsNeeded
    ) {
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

  public WritableMap transcribe(
    int jobId,
    float[] audioData,
    ReadableMap options
  ) throws IOException, Exception {
    if (isCapturing || isTranscribing) {
      throw new Exception("Context is already in capturing or transcribing");
    }
    rewind();
    this.jobId = jobId;
    this.isTdrzEnable =
      options.hasKey("tdrzEnable") && options.getBoolean("tdrzEnable");

    isTranscribing = true;

    boolean hasProgressCallback =
      options.hasKey("onProgress") && options.getBoolean("onProgress");
    boolean hasNewSegmentsCallback =
      options.hasKey("onNewSegments") && options.getBoolean("onNewSegments");
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
      hasProgressCallback || hasNewSegmentsCallback
        ? new Callback(this, hasProgressCallback, hasNewSegmentsCallback)
        : null
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
    freeContext(id, context);
  }

  static {
    // Extract HTP libraries from assets before loading native library
    try {
      Class<?> activityThread = Class.forName("android.app.ActivityThread");
      Object currentActivityThread = activityThread
        .getMethod("currentActivityThread")
        .invoke(null);
      Object app = activityThread
        .getMethod("getApplication")
        .invoke(currentActivityThread);
      android.content.Context appContext = (android.content.Context) app;

      if (appContext != null) {
        extractHtpLibrariesFromAssets(appContext);
      }
    } catch (Exception e) {
      Log.w(NAME, "Failed to extract HTP libraries", e);
    }

    Log.d(NAME, "Primary ABI: " + Build.SUPPORTED_ABIS[0]);

    String cpuFeatures = WhisperContext.getCpuFeatures();
    Log.d(NAME, "CPU features: " + cpuFeatures);
    boolean hasFp16 =
      cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");
    boolean hasDotProd = cpuFeatures.contains("asimddp");
    Log.d(NAME, "- hasFp16: " + hasFp16);
    Log.d(NAME, "- hasDotProd: " + hasDotProd);

    // Check for Hexagon support
    boolean hasHexagon = isHexagonSupported();
    Log.d(NAME, "- hasHexagon: " + hasHexagon);

    if (WhisperContext.isArm64V8a()) {
      if (hasFp16 && hasDotProd && hasHexagon) {
        Log.d(NAME, "Loading librnwhisper_jni_v8_2_hexagon.so");
        System.loadLibrary("rnwhisper_jni_v8_2_hexagon");
        loadedLibrary = "rnwhisper_jni_v8_2_hexagon";
      } else if (hasFp16) {
        Log.d(NAME, "Loading librnwhisper_jni_v8fp16_va_2.so");
        System.loadLibrary("rnwhisper_jni_v8fp16_va_2");
        loadedLibrary = "rnwhisper_jni_v8fp16_va_2";
      } else {
        Log.d(NAME, "Loading librnwhisper_jni_v8.so");
        System.loadLibrary("rnwhisper_jni_v8");
        loadedLibrary = "rnwhisper_jni_v8";
      }
    } else if (WhisperContext.isArmeabiV7a()) {
      Log.d(NAME, "Loading librnwhisper_jni_vfpv4.so");
      System.loadLibrary("rnwhisper_jni_vfpv4");
      loadedLibrary = "rnwhisper_jni_vfpv4";
    } else if (WhisperContext.isX86_64()) {
      Log.d(NAME, "Loading librnwhisper_jni_x86_64.so");
      System.loadLibrary("rnwhisper_jni_x86_64");
      loadedLibrary = "rnwhisper_jni_x86_64";
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

  private static boolean isHexagonSupported() {
    // Check SOC_MODEL on Android 12+
    if (Build.VERSION.SDK_INT >= 31) {
      String socModel = Build.SOC_MODEL;
      Log.d(NAME, "SOC Model: " + socModel);
      if (socModel != null) {
        socModel = socModel.toUpperCase();
        // SM8550 (8 Gen 2), SM8650 (8 Gen 3), SM8635 (8s Gen 3), SM8750 (8 Elite)
        if (socModel.matches(".*(SM8550|SM8650|SM8635|SM8750).*")) {
          return true;
        }
      }
    }

    // Check for supported Qualcomm platforms (Snapdragon 8 Gen 2 and newer)
    // Boards: kalama (8 Gen 2), pineapple (8 Gen 3), sun (8 Elite), lanai (8s Gen 3)
    String hardware = Build.HARDWARE.toLowerCase();
    Log.d(NAME, "Hardware: " + hardware);
    String board = Build.BOARD.toLowerCase();
    Log.d(NAME, "Board: " + board);
    if (
      hardware.matches(".*(kalama|pineapple|sun|lanai).*") ||
      board.matches(".*(kalama|pineapple|sun|lanai).*")
    ) {
      return true;
    }

    return false;
  }

  private static boolean prepareHtpDirectory(java.io.File dir, String label) {
    if (dir == null) {
      return false;
    }
    try {
      if (dir.exists()) {
        if (!dir.isDirectory()) {
          Log.w(
            NAME,
            label + " exists but is not a directory: " + dir.getAbsolutePath()
          );
          return false;
        }
      } else {
        if (!dir.mkdirs()) {
          Log.w(
            NAME,
            "Unable to create " + label + " at " + dir.getAbsolutePath()
          );
          return false;
        }
        java.io.File sanity = java.io.File.createTempFile("htp", ".tmp", dir);
        sanity.delete();
      }
    } catch (Exception e) {
      Log.w(
        NAME,
        "Unable to prepare " + label + " at " + dir.getAbsolutePath(),
        e
      );
      return false;
    }

    dir.setReadable(true, false);
    dir.setExecutable(true, false);
    dir.setWritable(true, true);

    try {
      android.system.Os.chmod(dir.getAbsolutePath(), HTP_DIR_MODE);
    } catch (Exception e) {
      Log.w(NAME, "Failed to chmod HTP directory " + dir.getAbsolutePath(), e);
    }

    return true;
  }

  private static java.io.File getPrivateHtpDir(
    android.content.Context context
  ) {
    try {
      return context.getDir(HTP_DIR_NAME, android.content.Context.MODE_PRIVATE);
    } catch (Exception e) {
      Log.w(NAME, "Unable to access private HTP directory", e);
      return null;
    }
  }

  private static java.io.File resolveHtpDirectory(
    android.content.Context context
  ) {
    java.io.File[] candidates = new java.io.File[] {
      getPrivateHtpDir(context),
      new java.io.File(context.getFilesDir(), HTP_DIR_NAME),
      context.getCodeCacheDir() != null
        ? new java.io.File(context.getCodeCacheDir(), HTP_DIR_NAME)
        : null,
      context.getCacheDir() != null
        ? new java.io.File(context.getCacheDir(), HTP_DIR_NAME)
        : null,
      context.getExternalFilesDir(null) != null
        ? new java.io.File(context.getExternalFilesDir(null), HTP_DIR_NAME)
        : null,
    };

    for (java.io.File candidate : candidates) {
      if (candidate == null) continue;
      if (prepareHtpDirectory(candidate, "HTP directory candidate")) {
        return candidate;
      }
    }

    Log.w(
      NAME,
      "Unable to provision directory for Hexagon libraries; Hexagon backend will be disabled"
    );
    return null;
  }

  private static void setHtpFilePermissions(java.io.File file) {
    file.setReadable(true, false);
    file.setExecutable(true, false);
    try {
      android.system.Os.chmod(file.getAbsolutePath(), HTP_FILE_MODE);
    } catch (Exception e) {
      Log.w(NAME, "Failed to chmod HTP library " + file.getAbsolutePath(), e);
    }
  }

  private static boolean ensureHtpLibraries(
    android.content.Context context,
    java.io.File htpDir
  ) {
    for (String libName : HTP_LIBS) {
      java.io.File outFile = new java.io.File(htpDir, libName);

      if (outFile.exists()) {
        continue;
      }

      try {
        try (
          InputStream in = context.getAssets().open("ggml-hexagon/" + libName);
          FileOutputStream out = new FileOutputStream(outFile)
        ) {
          byte[] buffer = new byte[8192];
          int read;
          while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
          }
          out.flush();
        }

        setHtpFilePermissions(outFile);
        Log.d(
          NAME,
          "Installed HTP library: " +
            libName +
            " to " +
            outFile.getAbsolutePath()
        );
      } catch (Exception e) {
        Log.w(NAME, "Could not install " + libName + " from assets", e);
        outFile.delete();
        return false;
      }
    }

    return true;
  }

  static void extractHtpLibrariesFromAssets(android.content.Context context) {
    java.io.File htpDir = resolveHtpDirectory(context);
    if (htpDir == null) {
      return;
    }

    Log.d(NAME, "Using " + htpDir.getAbsolutePath() + " for HTP libraries");

    if (!ensureHtpLibraries(context, htpDir)) {
      Log.w(
        NAME,
        "Could not install Hexagon libraries; Hexagon backend will be disabled"
      );
      return;
    }

    try {
      String htpLibPath = htpDir.getAbsolutePath();
      android.system.Os.setenv("ADSP_LIBRARY_PATH", htpLibPath, true);
      android.system.Os.setenv("WSP_GGML_HEXAGON_NDEV", "16", true);
      Log.d(NAME, "Set ADSP_LIBRARY_PATH=" + htpLibPath);
    } catch (Exception e) {
      Log.w(NAME, "Failed to set ADSP_LIBRARY_PATH", e);
    }
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

  protected static native long initContextWithAsset(
    int contextId,
    AssetManager assetManager,
    String modelPath
  );

  protected static native long initContextWithInputStream(
    int contextId,
    PushbackInputStream inputStream
  );

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

  protected static native boolean getTextSegmentSpeakerTurnNext(
    long context,
    int index
  );

  protected static native void createRealtimeTranscribeJob(
    int job_id,
    long context,
    ReadableMap options
  );

  protected static native void finishRealtimeTranscribeJob(
    int job_id,
    long context,
    int[] sliceNSamples
  );

  protected static native boolean vadSimple(
    int job_id,
    int slice_index,
    int n_samples,
    int n
  );

  protected static native void putPcmData(
    int job_id,
    short[] buffer,
    int slice_index,
    int n_samples,
    int n
  );

  protected static native int fullWithJob(
    int job_id,
    long context,
    int slice_index,
    int n_samples
  );

  protected static native String bench(long context, int n_threads);

  // VAD JNI methods
  protected static native long initVadContext(int contextId, String modelPath);

  protected static native long initVadContextWithAsset(
    int contextId,
    AssetManager assetManager,
    String modelPath
  );

  protected static native long initVadContextWithInputStream(
    int contextId,
    PushbackInputStream inputStream
  );

  protected static native void freeVadContext(
    int contextId,
    long vadContextPtr
  );

  protected static native boolean vadDetectSpeech(
    long vadContextPtr,
    float[] audioData,
    int nSamples
  );

  protected static native long vadGetSegmentsFromProbs(
    long vadContextPtr,
    float threshold,
    int minSpeechDurationMs,
    int minSilenceDurationMs,
    float maxSpeechDurationS,
    int speechPadMs,
    float samplesOverlap
  );

  protected static native int vadGetNSegments(long segmentsPtr);

  protected static native float vadGetSegmentT0(long segmentsPtr, int index);

  protected static native float vadGetSegmentT1(long segmentsPtr, int index);

  protected static native void vadFreeSegments(long segmentsPtr);

  // Audio file loading utility for VAD
  public static float[] loadAudioFileAsFloat32(String filePath) {
    try {
      java.io.FileInputStream fis = new java.io.FileInputStream(
        new java.io.File(filePath)
      );
      return AudioUtils.decodeWaveFile(fis);
    } catch (Exception e) {
      Log.e(NAME, "Failed to load audio file: " + filePath, e);
      return null;
    }
  }

  // JSI Installation
  protected static native void installJSIBindings(
    long runtimePtr,
    Object callInvokerHolder
  );

  protected static native void cleanupJSIBindings();

  protected static native void setupLog(NativeLogCallback logCallback);

  protected static native void unsetLog();
}
