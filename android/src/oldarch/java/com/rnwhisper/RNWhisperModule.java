package com.rnwhisper;

import androidx.annotation.NonNull;
import android.util.Log;
import android.os.Build;
import android.os.Handler;
import android.os.AsyncTask;
import android.media.AudioRecord;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.module.annotations.ReactModule;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

@ReactModule(name = RNWhisperModule.NAME)
public class RNWhisperModule extends ReactContextBaseJavaModule implements LifecycleEventListener {
  public static final String NAME = "RNWhisper";

  private ReactApplicationContext reactContext;
  private Downloader downloader;

  public RNWhisperModule(ReactApplicationContext reactContext) {
    super(reactContext);
    reactContext.addLifecycleEventListener(this);
    this.reactContext = reactContext;
    this.downloader = new Downloader(reactContext);
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  private HashMap<Integer, WhisperContext> contexts = new HashMap<>();

  private int getResourceIdentifier(String filePath) {
    int identifier = reactContext.getResources().getIdentifier(
      filePath,
      "drawable",
      reactContext.getPackageName()
    );
    if (identifier == 0) {
      identifier = reactContext.getResources().getIdentifier(
        filePath,
        "raw",
        reactContext.getPackageName()
      );
    }
    return identifier;
  }

  @ReactMethod
  public void initContext(final ReadableMap options, final Promise promise) {
    new AsyncTask<Void, Void, Integer>() {
      private Exception exception;

      @Override
      protected Integer doInBackground(Void... voids) {
        try {
          String modelPath = options.getString("filePath");
          boolean isBundleAsset = options.getBoolean("isBundleAsset");

          String modelFilePath = modelPath;
          if (!isBundleAsset && (modelPath.startsWith("http://") || modelPath.startsWith("https://"))) {
            modelFilePath = downloader.downloadFile(modelPath);
          }

          long context;
          int resId = getResourceIdentifier(modelFilePath);
          if (resId > 0) {
            context = WhisperContext.initContextWithInputStream(
              new PushbackInputStream(reactContext.getResources().openRawResource(resId))
            );
          } else if (isBundleAsset) {
            context = WhisperContext.initContextWithAsset(reactContext.getAssets(), modelFilePath);
          } else {
            context = WhisperContext.initContext(modelFilePath);
          }
          if (context == 0) {
            throw new Exception("Failed to initialize context");
          }
          int id = Math.abs(new Random().nextInt());
          WhisperContext whisperContext = new WhisperContext(id, reactContext, context);
          contexts.put(id, whisperContext);
          return id;
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(Integer id) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(id);
      }
    }.execute();
  }

  @ReactMethod
  public void transcribeFile(int id, int jobId, String filePath, ReadableMap options, Promise promise) {
    final WhisperContext context = contexts.get(id);
    if (context == null) {
      promise.reject("Context not found");
      return;
    }
    if (context.isCapturing()) {
      promise.reject("The context is in realtime transcribe mode");
      return;
    }
    if (context.isTranscribing()) {
      promise.reject("Context is already transcribing");
      return;
    }
    new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          String waveFilePath = filePath;

          if (filePath.startsWith("http://") || filePath.startsWith("https://")) {
            waveFilePath = downloader.downloadFile(filePath);
          }

          int resId = getResourceIdentifier(waveFilePath);
          if (resId > 0) {
            return context.transcribeInputStream(
              (int) jobId,
              reactContext.getResources().openRawResource(resId),
              options
            );
          }

          return context.transcribeInputStream(
            (int) jobId,
            new FileInputStream(new File(waveFilePath)),
            options
          );
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(WritableMap data) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(data);
      }
    }.execute();
  }

  @ReactMethod
  public void startRealtimeTranscribe(int id, int jobId, ReadableMap options, Promise promise) {
    final WhisperContext context = contexts.get(id);
    if (context == null) {
      promise.reject("Context not found");
      return;
    }
    if (context.isCapturing()) {
      promise.reject("Context is already in capturing");
      return;
    }
    int state = context.startRealtimeTranscribe(jobId, options);
    if (state == AudioRecord.STATE_INITIALIZED) {
      promise.resolve(null);
      return;
    }
    promise.reject("Failed to start realtime transcribe. State: " + state);
  }

  @ReactMethod
  public void abortTranscribe(int contextId, int jobId, Promise promise) {
    WhisperContext context = contexts.get(contextId);
    if (context == null) {
      promise.reject("Context not found");
      return;
    }
    context.stopTranscribe(jobId);
  }

  @ReactMethod
  public void releaseContext(int id, Promise promise) {
    new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          WhisperContext context = contexts.get(id);
          if (context == null) {
            throw new Exception("Context " + id + " not found");
          }
          context.release();
          contexts.remove(id);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
      }
    }.execute();
  }

  @ReactMethod
  public void releaseAllContexts(Promise promise) {
    new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          onHostDestroy();
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
      }
    }.execute();
  }

  @Override
  public void onHostResume() {
  }

  @Override
  public void onHostPause() {
  }

  @Override
  public void onHostDestroy() {
    WhisperContext.abortAllTranscribe();
    for (WhisperContext context : contexts.values()) {
      context.release();
    }
    contexts.clear();
    downloader.clearCache();
  }
}
