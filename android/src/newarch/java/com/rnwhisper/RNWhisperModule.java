package com.rnwhisper;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.module.annotations.ReactModule;

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
    rnwhisper.initContext(options, promise);
  }

  @ReactMethod
  public void transcribeFile(double id, double jobId, String filePath, ReadableMap options, Promise promise) {
    rnwhisper.transcribeFile(id, jobId, filePath, options, promise);
  }

  @ReactMethod
  public void startRealtimeTranscribe(double id, double jobId, ReadableMap options, Promise promise) {
    rnwhisper.startRealtimeTranscribe(id, jobId, options, promise);
  }

  @ReactMethod
  public void abortTranscribe(double contextId, double jobId, Promise promise) {
    rnwhisper.abortTranscribe(contextId, jobId, promise);
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
