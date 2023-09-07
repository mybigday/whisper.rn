package com.rnwhisper;

import androidx.annotation.NonNull;
import android.util.Log;
import android.os.Build;
import android.os.Handler;
import android.os.AsyncTask;
import android.media.AudioRecord;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableMap;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

public class RNWhisper implements LifecycleEventListener {
  public static final String NAME = "RNWhisper";

  private ReactApplicationContext reactContext;
  private Downloader downloader;

  public RNWhisper(ReactApplicationContext reactContext) {
    reactContext.addLifecycleEventListener(this);
    this.reactContext = reactContext;
    this.downloader = new Downloader(reactContext);
  }

  public HashMap<String, Object> getTypedExportedConstants() {
    HashMap<String, Object> constants = new HashMap<>();

    // iOS only constants, put for passing type checks
    constants.put("useCoreML", false);
    constants.put("coreMLAllowFallback", false);

    return constants;
  }

  private HashMap<AsyncTask, String> tasks = new HashMap<>();

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

  public void initContext(final ReadableMap options, final Promise promise) {
    AsyncTask task = new AsyncTask<Void, Void, Integer>() {
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
        tasks.remove(this);
      }
    }.execute();
    tasks.put(task, "initContext");
  }

  public void transcribeFile(double id, double jobId, String filePath, ReadableMap options, Promise promise) {
    final WhisperContext context = contexts.get((int) id);
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
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
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
        tasks.remove(this);
      }
    }.execute();
    tasks.put(task, "transcribeFile-" + id);
  }

  public void startRealtimeTranscribe(double id, double jobId, ReadableMap options, Promise promise) {
    final WhisperContext context = contexts.get((int) id);
    if (context == null) {
      promise.reject("Context not found");
      return;
    }
    if (context.isCapturing()) {
      promise.reject("Context is already in capturing");
      return;
    }
    int state = context.startRealtimeTranscribe((int) jobId, options);
    if (state == AudioRecord.STATE_INITIALIZED) {
      promise.resolve(null);
      return;
    }
    promise.reject("Failed to start realtime transcribe. State: " + state);
  }

  public void abortTranscribe(double id, double jobId, Promise promise) {
    WhisperContext context = contexts.get((int) id);
    if (context == null) {
      promise.reject("Context not found");
      return;
    }
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          context.stopTranscribe((int) jobId);
          AsyncTask completionTask = null;
          for (AsyncTask task : tasks.keySet()) {
            if (tasks.get(task).equals("transcribeFile-" + id)) {
              task.get();
              break;
            }
          }
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
        tasks.remove(this);
      }
    }.execute();
    tasks.put(task, "abortTranscribe-" + id);
  }

  public void releaseContext(double id, Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          WhisperContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context " + id + " not found");
          }
          context.stopCurrentTranscribe();
          AsyncTask completionTask = null;
          for (AsyncTask task : tasks.keySet()) {
            if (tasks.get(task).equals("transcribeFile-" + contextId)) {
              task.get();
              break;
            }
          }
          context.release();
          contexts.remove(contextId);
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
        tasks.remove(this);
      }
    }.execute();
    tasks.put(task, "releaseContext-" + id);
  }

  public void releaseAllContexts(Promise promise) {
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
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
        tasks.remove(this);
      }
    }.execute();
    tasks.put(task, "releaseAllContexts");
  }

  @Override
  public void onHostResume() {
  }

  @Override
  public void onHostPause() {
  }

  @Override
  public void onHostDestroy() {
    for (WhisperContext context : contexts.values()) {
      context.stopCurrentTranscribe();
    }
    for (AsyncTask task : tasks.keySet()) {
      try {
        task.get();
      } catch (Exception e) {
        Log.e(NAME, "Failed to wait for task", e);
      }
    }
    for (WhisperContext context : contexts.values()) {
      context.release();
    }
    WhisperContext.abortAllTranscribe(); // graceful abort
    contexts.clear();
    downloader.clearCache();
  }
}
