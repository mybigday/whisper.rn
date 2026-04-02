package com.rnwhisper;

import android.os.Build;
import android.util.Log;

import com.facebook.react.bridge.ReactApplicationContext;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class RNWhisper {
  public static final String NAME = "RNWhisper";
  private static final String TAG = "RNWhisper";
  private static boolean libsLoaded = false;

  private static boolean tryLoadLibrary(String library) {
    try {
      System.loadLibrary(library);
      Log.d(TAG, "Loaded native library: " + library);
      return true;
    } catch (UnsatisfiedLinkError error) {
      Log.w(TAG, "Unable to load native library " + library, error);
      return false;
    }
  }

  public static synchronized boolean loadNative(ReactApplicationContext context) {
    if (libsLoaded) {
      return true;
    }

    if (Build.SUPPORTED_ABIS.length == 0) {
      Log.w(TAG, "No supported ABIs reported by the runtime");
      return false;
    }

    String cpuFeatures = getCpuFeatures();
    boolean hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");

    try {
      if (isArm64V8a()) {
        if (hasFp16 && tryLoadLibrary("rnwhisper_v8fp16_va_2")) {
          libsLoaded = true;
          return true;
        }

        if (tryLoadLibrary("rnwhisper_v8")) {
          libsLoaded = true;
          return true;
        }
      } else if (isArmeabiV7a()) {
        if (tryLoadLibrary("rnwhisper_vfpv4")) {
          libsLoaded = true;
          return true;
        }
      } else if (isX86_64()) {
        if (tryLoadLibrary("rnwhisper_x86_64")) {
          libsLoaded = true;
          return true;
        }
      }

      if (tryLoadLibrary("rnwhisper")) {
        libsLoaded = true;
      }
    } catch (UnsatisfiedLinkError error) {
      Log.e(TAG, "Failed to load RNWhisper native library", error);
      libsLoaded = false;
    }

    return libsLoaded;
  }

  private static boolean isArm64V8a() {
    return Build.SUPPORTED_ABIS.length > 0
      && Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  private static boolean isArmeabiV7a() {
    return Build.SUPPORTED_ABIS.length > 0
      && Build.SUPPORTED_ABIS[0].equals("armeabi-v7a");
  }

  private static boolean isX86_64() {
    return Build.SUPPORTED_ABIS.length > 0
      && Build.SUPPORTED_ABIS[0].equals("x86_64");
  }

  private static String getCpuFeatures() {
    File file = new File("/proc/cpuinfo");
    StringBuilder builder = new StringBuilder();
    try (BufferedReader bufferedReader = new BufferedReader(new FileReader(file))) {
      String line;
      while ((line = bufferedReader.readLine()) != null) {
        if (line.startsWith("Features")) {
          builder.append(line);
          break;
        }
      }
      return builder.toString();
    } catch (IOException error) {
      Log.w(TAG, "Couldn't read /proc/cpuinfo", error);
      return "";
    }
  }
}
