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
import com.facebook.react.bridge.Arguments;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
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
        WritableMap result = Arguments.createMap();
        result.putInt("contextId", id);
        result.putBoolean("gpu", false);
        result.putString("reasonNoGPU", "Currently not supported");
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "initContext");
  }

  private AsyncTask transcribe(WhisperContext context, double jobId, final float[] audioData, final ReadableMap options, Promise promise) {
    Log.d("RNWhisper", "Starting transcription for jobId: " + jobId);
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        Log.d("RNWhisper", "Transcribing audio data for jobId: " + jobId);
        try {
          Log.d("RNWhisper", "Starting transcription for jobId: " + jobId + " with " + audioData.length + " samples.");
          return context.transcribe(
            (int) jobId,
            audioData,
            options
          );
        } catch (Exception e) {
          exception = e;
          Log.e("RNWhisper", "Error during transcription: " + e.getMessage());
          return null;
        }
      }

      @Override
      protected void onPostExecute(WritableMap data) {
        if (exception != null) {
          Log.e("RNWhisper", "Error during transcription: " + exception.getMessage());
          promise.reject(exception);
          return;
        }
        Log.d("RNWhisper", "Transcription completed for jobId: " + jobId);
        promise.resolve(data);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    return task;
  }

 public void transcribeFile(double id, double jobId, String filePathOrBase64, ReadableMap options, Promise promise) {
    final WhisperContext context = contexts.get((int) id);
    if (context == null) {
      Log.e("RNWhisper", "Context not found for id: " + id);
      promise.reject("Context not found");
      return;
    }
    if (context.isCapturing()) {
      Log.e("RNWhisper", "The context is in realtime transcribe mode");
      promise.reject("The context is in realtime transcribe mode");
      return;
    }
    if (context.isTranscribing()) {
      Log.e("RNWhisper", "Context is already transcribing");
      promise.reject("Context is already transcribing");
      return;
    }

    String waveFilePath = filePathOrBase64;
    try {
      Log.d("RNWhisper", "Transcribing file at path: " + waveFilePath);
      if (filePathOrBase64.startsWith("http://") || filePathOrBase64.startsWith("https://")) {
        waveFilePath = downloader.downloadFile(filePathOrBase64);
        Log.d("RNWhisper", "Downloaded file to: " + waveFilePath);
      }

      float[] audioData;
      int resId = getResourceIdentifier(waveFilePath);
      if (resId > 0) {
        audioData = AudioUtils.decodeWaveFile(reactContext.getResources().openRawResource(resId));
      } else if (filePathOrBase64.startsWith("data:audio/wav;base64,")) {
        audioData = AudioUtils.decodeWaveData(filePathOrBase64);
      } else {
        audioData = AudioUtils.decodeWaveFile(new FileInputStream(new File(waveFilePath)));
      }
      Log.d("RNWhisper", "Decoded audio data size: " + audioData.length);


      AsyncTask task = transcribe(context, jobId, audioData, options, promise);
      tasks.put(task, "transcribeFile-" + id);
    } catch (Exception e) {
      Log.e("RNWhisper", "Error transcribing file: " + e.getMessage());
      promise.reject(e);
    }
  }
  public void transcribeData(double id, double jobId, String dataBase64, ReadableMap options, Promise promise) {
    final WhisperContext context = contexts.get((int) id);
    if (context == null) {
      Log.e("RNWhisper", "Context not found for id: " + id);
      promise.reject("Context not found");
      return;
    }
    if (context.isCapturing()) {
      Log.e("RNWhisper", "The context is in realtime transcribe mode");
      promise.reject("The context is in realtime transcribe mode");
      return;
    }
    if (context.isTranscribing()) {
      Log.e("RNWhisper", "Context is already transcribing");
      promise.reject("Context is already transcribing");
      return;
    }

    try {
      Log.d("RNWhisper", "Transcribing data with base64: " + dataBase64.substring(0, Math.min(dataBase64.length(), 100)) + "...");
      float[] audioData = AudioUtils.decodePcmData(dataBase64);
      AsyncTask task = transcribe(context, jobId, audioData, options, promise);
      tasks.put(task, "transcribeData-" + id);
    } catch (Exception e) {
      Log.e("RNWhisper", "Error transcribing data: " + e.getMessage());
      promise.reject(e);
    }
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

  public void pauseRealtimeTranscribe(double contextId, Promise promise) {
      WhisperContext context = contexts.get((int) contextId);
      if (context != null) {
          context.pauseRealtimeTranscribe();  // Delegate to WhisperContext
          promise.resolve(null);
      } else {
          promise.reject("Context not found");
      }
  }

  // Method to resume transcription by contextId
  public void resumeRealtimeTranscribe(double contextId, Promise promise) {
      WhisperContext context = contexts.get((int) contextId);
      if (context != null) {
          context.resumeRealtimeTranscribe();  // Delegate to WhisperContext
          promise.resolve(null);
      } else {
          promise.reject("Context not found");
      }
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
            if (tasks.get(task).equals("transcribeFile-" + id) || tasks.get(task).equals("transcribeData-" + id)) {
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
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "abortTranscribe-" + id);
  }

  public void bench(double id, double nThreads, Promise promise) {
    final WhisperContext context = contexts.get((int) id);
    if (context == null) {
      promise.reject("Context not found");
      return;
    }
    promise.resolve(context.bench((int) nThreads));
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
            if (tasks.get(task).equals("transcribeFile-" + contextId) || tasks.get(task).equals("transcribeData-" + contextId)) {
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
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
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
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
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
