package com.rnwhisper;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReadableMap;

import android.util.Log;
import android.os.Build;
import android.content.res.AssetManager;

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
  private long context;

  public WhisperContext(long context) {
    this.context = context;
  }

  public WritableMap transcribe(int jobId, String filePath, ReadableMap options) throws IOException, Exception {
    int code = fullTranscribe(
      jobId,
      context,
      decodeWaveFile(new File(filePath)),
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
    if (code != 0) {
      throw new Exception("Transcription failed with code " + code);
    }
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

  public void abortTranscribeBy(int jobId) {
    abortTranscribe(jobId);
  }

  public void release() {
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
    float[] audio_data,
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
  protected static native int getTextSegmentCount(long context);
  protected static native String getTextSegment(long context, int index);
  protected static native int getTextSegmentT0(long context, int index);
  protected static native int getTextSegmentT1(long context, int index);
  protected static native void freeContext(long contextPtr);
}