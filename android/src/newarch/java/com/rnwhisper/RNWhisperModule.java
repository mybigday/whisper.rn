package com.rnwhisper;

import androidx.annotation.NonNull;

import android.content.res.AssetManager;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;

import java.util.HashMap;

@ReactModule(name = RNWhisper.NAME)
public class RNWhisperModule extends NativeRNWhisperSpec {
  public static final String NAME = RNWhisper.NAME;

  private final ReactApplicationContext context;

  public RNWhisperModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.context = reactContext;
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  @Override
  public HashMap<String, Object> getConstants() {
    HashMap<String, Object> constants = new HashMap<>();
    constants.put("useCoreML", false);
    constants.put("coreMLAllowFallback", false);
    return constants;
  }

  @Override
  public void install(Promise promise) {
    try {
      boolean loaded = RNWhisper.loadNative(context);
      if (!loaded) {
        promise.resolve(false);
        return;
      }

      long jsContextPointer = context.getJavaScriptContextHolder().get();
      CallInvokerHolderImpl holder =
        (CallInvokerHolderImpl) context.getCatalystInstance().getJSCallInvokerHolder();
      AssetManager assetManager = context.getAssets();

      if (jsContextPointer == 0 || holder == null || assetManager == null) {
        promise.resolve(false);
        return;
      }

      installJSIBindings(
        jsContextPointer,
        holder,
        context.getApplicationContext(),
        assetManager
      );
      promise.resolve(true);
    } catch (UnsatisfiedLinkError error) {
      promise.resolve(false);
    } catch (Exception error) {
      promise.resolve(false);
    }
  }

  private native void installJSIBindings(
    long jsContextPointer,
    CallInvokerHolderImpl callInvokerHolder,
    Object applicationContext,
    Object assetManager
  );

  private native void cleanupJSIBindings();

  @Override
  public void invalidate() {
    try {
      cleanupJSIBindings();
    } catch (UnsatisfiedLinkError ignored) {
    }
    super.invalidate();
  }
}
