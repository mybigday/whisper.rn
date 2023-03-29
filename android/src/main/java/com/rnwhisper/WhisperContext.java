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

import java.util.Random;
import java.lang.StringBuilder;
import java.io.File;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

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

  private DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;
  
  private int jobId = -1;
  private AudioRecord recorder = null;
  private int bufferSize;
  private short[] buffer16;
  private int nSamples = 0;
  private int nSamplesTranscribing = 0;
  private boolean isCapturing = false;
  private boolean isStoppedByAction = false;
  private boolean isTranscribing = false;
  private boolean isRealtime = false;
  private Thread fullHandler = null;

  public WhisperContext(int id, ReactApplicationContext reactContext, long context) {
    this.id = id;
    this.context = context;
    this.reactContext = reactContext;
    eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
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
    
    int realtimeAudioSec = options.hasKey("realtimeAudioSec") ? options.getInt("realtimeAudioSec") : 0;
    final int maxAudioSec = realtimeAudioSec > 0 ? realtimeAudioSec : DEFAULT_MAX_AUDIO_SEC;

    buffer16 = new short[maxAudioSec * SAMPLE_RATE * Short.BYTES];

    this.jobId = jobId;
    isCapturing = true;
    isStoppedByAction = false;
    isRealtime = true;
    nSamples = 0;
    nSamplesTranscribing = 0;
    fullHandler = null;
  
    recorder.startRecording();

    new Thread(new Runnable() {
      @Override
      public void run() {
        try {
          short[] buffer = new short[bufferSize];
          while (isCapturing) {
            try {
              int n = recorder.read(buffer, 0, bufferSize);
              if (n == 0) continue;

              if (nSamples + n > maxAudioSec * SAMPLE_RATE) {
                // Full, stop capturing
                isCapturing = false;
                if (!isTranscribing && nSamples == nSamplesTranscribing) {
                  emitTranscribeEvent("@RNWhisper_onRealtimeTranscribeEnd", Arguments.createMap());
                } else {
                  // wait previous handler to finish
                  fullHandler.join();
                  fullTranscribeSamples(options, true);
                }
                break;
              }

              // Append to buffer
              nSamples += n;
              for (int i = 0; i < n; i++) {
                buffer16[nSamples + i] = buffer[i];
              }

              fullTranscribeSamples(options, false);
            } catch (Exception e) {
              Log.e(NAME, "Error transcribing realtime: " + e.getMessage());
            }
          }
          if (!isTranscribing) {
            emitTranscribeEvent("@RNWhisper_onRealtimeTranscribeEnd", Arguments.createMap());
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
    }).start();
    return state;
  }

  private void fullTranscribeSamples(ReadableMap options, boolean skipCapturingCheck) {
    if (!isTranscribing && nSamples > SAMPLE_RATE / 2) {
      isTranscribing = true;
      fullHandler = new Thread(new Runnable() {
        @Override
        public void run() {
          if (!isCapturing && !skipCapturingCheck) return;

          nSamplesTranscribing = nSamples;

          // convert I16 to F32
          float[] nSamplesBuffer32 = new float[nSamplesTranscribing];
          for (int i = 0; i < nSamplesTranscribing; i++) {
            nSamplesBuffer32[i] = buffer16[i] / 32768.0f;
          }

          Log.d(NAME, "Start transcribing realtime: " + nSamplesTranscribing);

          int timeStart = (int) System.currentTimeMillis();
          int code = full(jobId, options, nSamplesBuffer32, nSamplesTranscribing);
          int timeEnd = (int) System.currentTimeMillis();
          int timeRecording = (int) (nSamplesTranscribing / SAMPLE_RATE * 1000);

          WritableMap payload = Arguments.createMap();
          payload.putInt("code", code);
          payload.putInt("processTime", timeEnd - timeStart);
          payload.putInt("recordingTime", timeRecording);

          if (code == 0) {
            payload.putMap("data", getTextSegments());
          } else {
            payload.putString("error", "Transcribe failed with code " + code);
          }

          if (isStoppedByAction || !isCapturing && nSamplesTranscribing == nSamples) {
            payload.putBoolean("isCapturing", false);
            payload.putBoolean("isStoppedByAction", isStoppedByAction);
            emitTranscribeEvent("@RNWhisper_onRealtimeTranscribeEnd", payload);
          } else if (code == 0) {
            payload.putBoolean("isCapturing", true);
            emitTranscribeEvent("@RNWhisper_onRealtimeTranscribe", payload);
          } else {
            payload.putBoolean("isCapturing", true);
            emitTranscribeEvent("@RNWhisper_onRealtimeTranscribe", payload);
          }
          isTranscribing = false;
        }
      });
      fullHandler.start();
    }
  }

  private void emitTranscribeEvent(final String eventName, final WritableMap payload) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", WhisperContext.this.id);
    event.putInt("jobId", jobId);
    event.putMap("payload", payload);
    eventEmitter.emit(eventName, event);
  }

  public WritableMap transcribeFile(int jobId, String filePath, ReadableMap options) throws IOException, Exception {
    this.jobId = jobId;
    isTranscribing = true;
    float[] audioData = decodeWaveFile(new File(filePath));
    int code = full(jobId, options, audioData, audioData.length);
    isTranscribing = false;
    this.jobId = -1;
    if (code != 0) {
      throw new Exception("Failed to transcribe the file. Code: " + code);
    }
    return getTextSegments();
  }

  private int full(int jobId, ReadableMap options, float[] audioData, int audioDataLen) {
    return fullTranscribe(
      jobId,
      context,
      // jboolean realtime,
      isRealtime,
      // float[] audio_data,
      audioData,
      // jint audio_data_len,
      audioDataLen,
      // jint n_threads,
      options.hasKey("maxThreads") ? options.getInt("maxThreads") : -1,
      // jint max_context,
      options.hasKey("maxContext") ? options.getInt("maxContext") : -1,

      // jint word_thold,
      options.hasKey("wordThold") ? options.getInt("wordThold") : -1,
      // jint max_len,
      options.hasKey("maxLen") ? options.getInt("maxLen") : -1,
      // jboolean token_timestamps,
      options.hasKey("tokenTimestamps") ? options.getBoolean("tokenTimestamps") : false,
  
      // jint offset,
      options.hasKey("offset") ? options.getInt("offset") : -1,
      // jint duration,
      options.hasKey("duration") ? options.getInt("duration") : -1,
      // jfloat temperature,
      options.hasKey("temperature") ? (float) options.getDouble("temperature") : -1.0f,
      // jfloat temperature_inc,
      options.hasKey("temperatureInc") ? (float) options.getDouble("temperatureInc") : -1.0f,
      // jint beam_size,
      options.hasKey("beamSize") ? options.getInt("beamSize") : -1,
      // jint best_of,
      options.hasKey("bestOf") ? options.getInt("bestOf") : -1,
      // jboolean speed_up,
      options.hasKey("speedUp") ? options.getBoolean("speedUp") : false,
      // jboolean translate,
      options.hasKey("translate") ? options.getBoolean("translate") : false,
      // jstring language,
      options.hasKey("language") ? options.getString("language") : "auto",
      // jstring prompt
      options.hasKey("prompt") ? options.getString("prompt") : null
    );
  }

  private WritableMap getTextSegments() {
    Integer count = getTextSegmentCount(context);
    StringBuilder builder = new StringBuilder();

    WritableMap data = Arguments.createMap();
    WritableArray segments = Arguments.createArray();
    for (int i = 0; i < count; i++) {
      String text = getTextSegment(context, i);
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


  public boolean isCapturing() {
    return isCapturing;
  }

  public boolean isTranscribing() {
    return isTranscribing;
  }

  public void stopTranscribe(int jobId) {
    abortTranscribe(jobId);
    isCapturing = false;
    isTranscribing = false;
    isStoppedByAction = true;
  }

  public void stopCurrentTranscribe() {
    stopTranscribe(this.jobId);
  }

  public void release() {
    stopCurrentTranscribe();
    freeContext(context);
  }

  public static float[] decodeWaveFile(File file) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (InputStream inputStream = new FileInputStream(file)) {
      byte[] buffer = new byte[1024];
      int bytesRead;
      while ((bytesRead = inputStream.read(buffer)) != -1) {
        baos.write(buffer, 0, bytesRead);
      }
    }
    ByteBuffer byteBuffer = ByteBuffer.wrap(baos.toByteArray());
    byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
    byteBuffer.position(44);
    ShortBuffer shortBuffer = byteBuffer.asShortBuffer();
    short[] shortArray = new short[shortBuffer.limit()];
    shortBuffer.get(shortArray);
    float[] floatArray = new float[shortArray.length];
    for (int i = 0; i < shortArray.length; i++) {
      floatArray[i] = ((float) shortArray[i]) / 32767.0f;
      floatArray[i] = Math.max(floatArray[i], -1f);
      floatArray[i] = Math.min(floatArray[i], 1f);
    }
    return floatArray;
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

  protected static native long initContext(String modelPath);
  protected static native int fullTranscribe(
    int job_id,
    long context,
    boolean realtime,
    float[] audio_data,
    int audio_data_len,
    int n_threads,
    int max_context,
    int word_thold,
    int max_len,
    boolean token_timestamps,
    int offset,
    int duration,
    float temperature,
    float temperature_inc,
    int beam_size,
    int best_of,
    boolean speed_up,
    boolean translate,
    String language,
    String prompt
  );
  protected static native void abortTranscribe(int jobId);
  protected static native void abortAllTranscribe();
  protected static native int getTextSegmentCount(long context);
  protected static native String getTextSegment(long context, int index);
  protected static native int getTextSegmentT0(long context, int index);
  protected static native int getTextSegmentT1(long context, int index);
  protected static native void freeContext(long contextPtr);
}