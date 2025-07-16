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
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;
import com.facebook.react.turbomodule.core.interfaces.CallInvokerHolder;
import com.facebook.react.bridge.Arguments;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.PushbackInputStream;

import com.facebook.react.common.LifecycleState;
import com.facebook.react.modules.core.DeviceEventManagerModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;
import com.facebook.react.bridge.ReactContext;

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
  private HashMap<Integer, WhisperVadContext> vadContexts = new HashMap<>();

  // JSI helper method to check if context exists
  public boolean hasContext(int contextId) {
    return contexts.containsKey(contextId);
  }

  public void installJSIBindings(Promise promise) {
    if (!WhisperContext.isNativeLibraryLoaded()) {
      promise.reject("Native library not loaded");
      return;
    }

    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          CallInvokerHolderImpl callInvokerHolder = JSCallInvokerResolver.getJSCallInvokerHolder(reactContext);
          long runtimePtr = JSCallInvokerResolver.getJavaScriptContextHolder(reactContext);

          WhisperContext.installJSIBindings(runtimePtr, callInvokerHolder);
          android.util.Log.i("RNWhisperModule", "JSI bindings installed successfully");
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
    tasks.put(task, "installJSIBindings");
  }

  public void toggleNativeLog(boolean enabled, Promise promise) {
    if (!WhisperContext.isNativeLibraryLoaded()) {
      promise.reject("Native library not loaded");
      return;
    }

    new AsyncTask<Void, Void, Boolean>() {
      private Exception exception;

      @Override
      protected Boolean doInBackground(Void... voids) {
        try {
          WhisperContext.toggleNativeLog(reactContext, enabled);
          return true;
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Boolean result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(result);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
  }

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
    AsyncTask task = new AsyncTask<Void, Void, WhisperContext>() {
      private Exception exception;

      @Override
      protected WhisperContext doInBackground(Void... voids) {
        try {
          String modelPath = options.getString("filePath");
          boolean isBundleAsset = options.getBoolean("isBundleAsset");

          String modelFilePath = modelPath;
          if (!isBundleAsset && (modelPath.startsWith("http://") || modelPath.startsWith("https://"))) {
            modelFilePath = downloader.downloadFile(modelPath);
          }

          int id = Math.abs(new Random().nextInt());
          long context;
          int resId = getResourceIdentifier(modelFilePath);
          if (resId > 0) {
            context = WhisperContext.initContextWithInputStream(
              id,
              new PushbackInputStream(reactContext.getResources().openRawResource(resId))
            );
          } else if (isBundleAsset) {
            context = WhisperContext.initContextWithAsset(id, reactContext.getAssets(), modelFilePath);
          } else {
            context = WhisperContext.initContext(id, modelFilePath);
          }
          if (context == 0) {
            throw new Exception("Failed to initialize context");
          }
          WhisperContext whisperContext = new WhisperContext(id, reactContext, context);
          contexts.put(id, whisperContext);
          return whisperContext;
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(WhisperContext context) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        WritableMap result = Arguments.createMap();
        result.putInt("contextId", context.getId());
        result.putDouble("contextPtr", (double) context.getContextPtr());
        result.putBoolean("gpu", false);
        result.putString("reasonNoGPU", "Currently not supported");
        promise.resolve(result);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "initContext");
  }

  private AsyncTask transcribe(WhisperContext context, double jobId, final float[] audioData, final ReadableMap options, Promise promise) {
    AsyncTask task = new AsyncTask<Void, Void, WritableMap>() {
      private Exception exception;

      @Override
      protected WritableMap doInBackground(Void... voids) {
        try {
          return context.transcribe(
            (int) jobId,
            audioData,
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
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    return task;
  }

  public void transcribeFile(double id, double jobId, String filePathOrBase64, ReadableMap options, Promise promise) {
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

    String waveFilePath = filePathOrBase64;
    try {
      if (filePathOrBase64.startsWith("http://") || filePathOrBase64.startsWith("https://")) {
        waveFilePath = downloader.downloadFile(filePathOrBase64);
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

      AsyncTask task = transcribe(context, jobId, audioData, options, promise);
      tasks.put(task, "transcribeFile-" + id);
    } catch (Exception e) {
      promise.reject(e);
    }
  }

  public void transcribeData(double id, double jobId, String dataBase64, ReadableMap options, Promise promise) {
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

    float[] audioData = AudioUtils.decodePcmData(dataBase64);
    AsyncTask task = transcribe(context, jobId, audioData, options, promise);

    tasks.put(task, "transcribeData-" + id);
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
          releaseAllContexts();
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

  public void initVadContext(final ReadableMap options, final Promise promise) {
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

          int id = Math.abs(new Random().nextInt());
          long vadContext;
          int resId = getResourceIdentifier(modelFilePath);
          if (resId > 0) {
            vadContext = WhisperContext.initVadContextWithInputStream(
              id,
              new PushbackInputStream(reactContext.getResources().openRawResource(resId))
            );
          } else if (isBundleAsset) {
            vadContext = WhisperContext.initVadContextWithAsset(id, reactContext.getAssets(), modelFilePath);
          } else {
            vadContext = WhisperContext.initVadContext(id, modelFilePath);
          }
          if (vadContext == 0) {
            throw new Exception("Failed to initialize VAD context");
          }
          WhisperVadContext whisperVadContext = new WhisperVadContext(id, reactContext, vadContext);
          vadContexts.put(id, whisperVadContext);
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
    tasks.put(task, "initVadContext");
  }

  public void vadDetectSpeech(double id, String audioDataBase64, ReadableMap options, Promise promise) {
    final WhisperVadContext vadContext = vadContexts.get((int) id);
    if (vadContext == null) {
      promise.reject("VAD context not found");
      return;
    }

    AsyncTask task = new AsyncTask<Void, Void, WritableArray>() {
      private Exception exception;

      @Override
      protected WritableArray doInBackground(Void... voids) {
        try {
          float[] audioData = AudioUtils.decodePcmData(audioDataBase64);
          return vadContext.detectSpeechWithAudioData(audioData, audioData.length, options);
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(WritableArray segments) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(segments);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "vadDetectSpeech-" + id);
  }

  public void vadDetectSpeechFile(double id, String filePathOrBase64, ReadableMap options, Promise promise) {
    final WhisperVadContext vadContext = vadContexts.get((int) id);
    if (vadContext == null) {
      promise.reject("VAD context not found");
      return;
    }

    AsyncTask task = new AsyncTask<Void, Void, WritableArray>() {
      private Exception exception;

      @Override
      protected WritableArray doInBackground(Void... voids) {
        try {
          // Handle file processing like transcribeFile does
          String filePath = filePathOrBase64;
          if (filePathOrBase64.startsWith("http://") || filePathOrBase64.startsWith("https://")) {
            filePath = downloader.downloadFile(filePathOrBase64);
          }

          float[] audioData;
          int resId = getResourceIdentifier(filePath);
          if (resId > 0) {
            audioData = AudioUtils.decodeWaveFile(reactContext.getResources().openRawResource(resId));
          } else if (filePathOrBase64.startsWith("data:audio/wav;base64,")) {
            audioData = AudioUtils.decodeWaveData(filePathOrBase64);
          } else {
            audioData = AudioUtils.decodeWaveFile(new java.io.FileInputStream(new java.io.File(filePath)));
          }

          if (audioData == null) {
            throw new Exception("Failed to load audio file: " + filePathOrBase64);
          }

          return vadContext.detectSpeechWithAudioData(audioData, audioData.length, options);
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(WritableArray segments) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(segments);
        tasks.remove(this);
      }
    }.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
    tasks.put(task, "vadDetectSpeechFile-" + id);
  }

  public void releaseVadContext(double id, Promise promise) {
    final int contextId = (int) id;
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          WhisperVadContext vadContext = vadContexts.get(contextId);
          if (vadContext == null) {
            throw new Exception("VAD context " + id + " not found");
          }
          vadContext.release();
          vadContexts.remove(contextId);
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
    tasks.put(task, "releaseVadContext-" + id);
  }

  public void releaseAllVadContexts(Promise promise) {
    AsyncTask task = new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          releaseAllVadContexts();
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
    tasks.put(task, "releaseAllVadContexts");
  }

  @Override
  public void onHostResume() {
  }

  @Override
  public void onHostPause() {
  }

  private void releaseAllContexts() {
    for (WhisperContext context : contexts.values()) {
      context.stopCurrentTranscribe();
    }
    WhisperContext.abortAllTranscribe(); // graceful abort
    for (WhisperContext context : contexts.values()) {
      context.release();
    }
    contexts.clear();
  }

  private void releaseAllVadContexts() {
    for (WhisperVadContext vadContext : vadContexts.values()) {
      vadContext.release();
    }
    vadContexts.clear();
  }

  @Override
  public void onHostDestroy() {
    for (AsyncTask task : tasks.keySet()) {
      try {
        task.get();
      } catch (Exception e) {
        Log.e(NAME, "Failed to wait for task", e);
      }
    }
    downloader.clearCache();
    releaseAllContexts();
    releaseAllVadContexts();
  }
}
