package com.rnwhisper;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.module.annotations.ReactModule;
import android.util.Log;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

@ReactModule(name = RNWhisperModule.NAME)
public class RNWhisperModule extends NativeRNWhisperSpec {
  public static final String NAME = "RNWhisper";

  private RNWhisper rnwhisper;

  public RNWhisperModule(ReactApplicationContext reactContext) {
    super(reactContext);
    rnwhisper = new RNWhisper(reactContext);
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  @Override
  public HashMap<String, Object> getTypedExportedConstants() {
    return rnwhisper.getTypedExportedConstants();
  }

  @ReactMethod
  public void initContext(final ReadableMap options, final Promise promise) {
    Log.d("RNWhisperModule", "Calling initContext with options: " + options);
    rnwhisper.initContext(options, promise);
  }

  @ReactMethod
  public void transcribeFile(double id, double jobId, String filePath, ReadableMap options, Promise promise) {
    Log.d("RNWhisperModule", "Calling transcribeFile with id: " + id + ", jobId: " + jobId + ", filePath: " + filePath);
    rnwhisper.transcribeFile(id, jobId, filePath, options, promise);
  }

  @ReactMethod
  public void transcribeData(double id, double jobId, String dataBase64, ReadableMap options, Promise promise) {
    Log.d("RNWhisperModule", "Calling transcribeData with id: " + id + ", jobId: " + jobId + ", dataBase64: " + dataBase64.substring(0, Math.min(dataBase64.length(), 100)) + "...");
    rnwhisper.transcribeData(id, jobId, dataBase64, options, promise);
  }


  @ReactMethod
  public void startRealtimeTranscribe(double id, double jobId, ReadableMap options, Promise promise) {
    rnwhisper.startRealtimeTranscribe(id, jobId, options, promise);
  }

  @ReactMethod
  public void pauseRealtimeTranscribe(double contextId, Promise promise) {
      rnwhisper.pauseRealtimeTranscribe(contextId, promise);
  }
  
  @ReactMethod
  public void finalizeWavFile(String filePath, Promise promise) {
    try {
      WavWriter wavWriter = new WavWriter();
      boolean success = wavWriter.finalizeExternalWav(filePath);
      if (success) {
        promise.resolve(true);
      } else {
        promise.reject("file_error", "Failed to finalize WAV");
      }
    } catch (Exception e) {
      promise.reject("file_error", e.getMessage());
    }
  }
  
  @ReactMethod
  public void resumeRealtimeTranscribe(double contextId, Promise promise) {
      rnwhisper.resumeRealtimeTranscribe(contextId, promise);
  }

  @ReactMethod
  public void abortTranscribe(double contextId, double jobId, Promise promise) {
    rnwhisper.abortTranscribe(contextId, jobId, promise);
  }

  @ReactMethod
  public void bench(double id, double nThreads, Promise promise) {
    rnwhisper.bench(id, nThreads, promise);
  }

  @ReactMethod
  public void releaseContext(double id, Promise promise) {
    rnwhisper.releaseContext(id, promise);
  }

  @ReactMethod
  public void releaseAllContexts(Promise promise) {
    rnwhisper.releaseAllContexts(promise);
  }

  /*
   * iOS Specific methods, left here for make the turbo module happy:
   */

  @ReactMethod
  public void getAudioSessionCurrentCategory(Promise promise) {
    promise.resolve(null);
  }

  @ReactMethod
  public void getAudioSessionCurrentMode(Promise promise) {
    promise.resolve(null);
  }
  @ReactMethod
  public void setAudioSessionCategory(String category, ReadableArray options, Promise promise) {
    promise.resolve(null);
  }
  @ReactMethod
  public void setAudioSessionMode(String mode, Promise promise) {
    promise.resolve(null);
  }
  @ReactMethod
  public void setAudioSessionActive(boolean active, Promise promise) {
    promise.resolve(null);
  }
}
