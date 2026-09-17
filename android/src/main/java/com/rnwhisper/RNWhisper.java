package com.rnwhisper;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import com.facebook.react.bridge.ReactApplicationContext;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.regex.Pattern;

public class RNWhisper {
  public static final String NAME = "RNWhisper";
  private static final String TAG = "RNWhisper";
  private static boolean libsLoaded = false;

  // Hexagon (Qualcomm NPU). The DSP-side kernels ship as app assets
  // (android/build.gradle syncRNWhisperHtpAssets) and are copied to an
  // app-private directory that ADSP_LIBRARY_PATH points at, since the DSP
  // loader reads them from the filesystem rather than from the APK.
  private static final String HTP_DIR_NAME = "rnwhisper-htp";
  private static final int HTP_FILE_MODE = 0755;
  private static final String[] HTP_LIBS = {
    "libggml-htp-v73.so",
    "libggml-htp-v75.so",
    "libggml-htp-v79.so",
    "libggml-htp-v81.so"
  };
  // Same heuristics as llama.rn: SoCs whose HTP is known to work with ggml-hexagon,
  // plus a generic Snapdragon 8-series match when the device also reports Qualcomm.
  private static final Pattern QUALCOMM_HINT_PATTERN =
    Pattern.compile("(adreno|qcom|qualcomm|snapdragon)", Pattern.CASE_INSENSITIVE);
  private static final Pattern KNOWN_HEXAGON_SOC_PATTERN =
    Pattern.compile("\\b(SM8450|SM8550|SM8635|SM8650|SM8750|SM8845|SM8850)\\b");
  private static final Pattern SNAPDRAGON_8_SERIES_SOC_PATTERN = Pattern.compile("\\bSM8\\d{3}\\b");
  private static final Pattern SNAPDRAGON_8_SERIES_NAME_PATTERN = Pattern.compile("SNAPDRAGON\\s*8");
  private static final Pattern HEXAGON_CODENAME_PATTERN = Pattern.compile("(taro|kalama|pineapple|sun|lanai)");

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
        if (hasFp16 && isHexagonSupported() && prepareHexagon(context)
            && tryLoadLibrary("rnwhisper_v8fp16_va_2_hexagon")) {
          libsLoaded = true;
          return true;
        }

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

  // Extracts the HTP libraries and points ADSP_LIBRARY_PATH at them. Returns
  // false when the app does not bundle them, in which case the CPU variant is
  // loaded instead.
  private static boolean prepareHexagon(Context context) {
    File htpDir;
    try {
      htpDir = context.getDir(HTP_DIR_NAME, Context.MODE_PRIVATE);
    } catch (Exception error) {
      Log.w(TAG, "Unable to create the HTP directory; using CPU", error);
      return false;
    }

    for (String libName : HTP_LIBS) {
      File outFile = new File(htpDir, libName);
      try (InputStream in = context.getAssets().open("ggml-hexagon/" + libName);
           FileOutputStream out = new FileOutputStream(outFile)) {
        byte[] buffer = new byte[64 * 1024];
        int read;
        while ((read = in.read(buffer)) != -1) {
          out.write(buffer, 0, read);
        }
      } catch (IOException error) {
        Log.w(TAG, "HTP library " + libName + " not bundled in assets/ggml-hexagon; using CPU");
        return false;
      }
      outFile.setReadable(true, false);
      outFile.setExecutable(true, false);
      try {
        android.system.Os.chmod(outFile.getAbsolutePath(), HTP_FILE_MODE);
      } catch (Exception error) {
        Log.w(TAG, "Failed to chmod " + outFile.getAbsolutePath(), error);
      }
    }

    try {
      android.system.Os.setenv("ADSP_LIBRARY_PATH", htpDir.getAbsolutePath(), true);
      // whisper.cpp uses a single GPU-class device, so one HTP session is enough.
      android.system.Os.setenv("GGML_HEXAGON_DEVICES", "1", false);
    } catch (Exception error) {
      Log.w(TAG, "Failed to set ADSP_LIBRARY_PATH; using CPU", error);
      return false;
    }
    Log.d(TAG, "HTP libraries extracted to " + htpDir.getAbsolutePath());
    return true;
  }

  private static String lowerOrEmpty(String value) {
    return value == null ? "" : value.toLowerCase(Locale.ROOT);
  }

  private static String upperOrEmpty(String value) {
    return value == null ? "" : value.toUpperCase(Locale.ROOT);
  }

  private static boolean hasQualcommDeviceHint() {
    StringBuilder hints = new StringBuilder();
    hints
      .append(lowerOrEmpty(Build.HARDWARE)).append(' ')
      .append(lowerOrEmpty(Build.BOARD)).append(' ')
      .append(lowerOrEmpty(Build.MANUFACTURER)).append(' ')
      .append(lowerOrEmpty(Build.BRAND)).append(' ')
      .append(lowerOrEmpty(Build.MODEL));

    if (Build.VERSION.SDK_INT >= 31) {
      hints.append(' ')
        .append(lowerOrEmpty(Build.SOC_MANUFACTURER)).append(' ')
        .append(lowerOrEmpty(Build.SOC_MODEL));
    }

    return QUALCOMM_HINT_PATTERN.matcher(hints.toString()).find();
  }

  private static boolean isHexagonSupported() {
    boolean hasQualcommHint = hasQualcommDeviceHint();

    if (Build.VERSION.SDK_INT >= 31) {
      String socModel = upperOrEmpty(Build.SOC_MODEL);
      if (!socModel.isEmpty()) {
        if (KNOWN_HEXAGON_SOC_PATTERN.matcher(socModel).find()) {
          return true;
        }
        if (hasQualcommHint &&
            (SNAPDRAGON_8_SERIES_SOC_PATTERN.matcher(socModel).find() ||
             SNAPDRAGON_8_SERIES_NAME_PATTERN.matcher(socModel).find())) {
          return true;
        }
      }
    }

    String hardwareHints = lowerOrEmpty(Build.HARDWARE) + " " + lowerOrEmpty(Build.BOARD);
    return hasQualcommHint && HEXAGON_CODENAME_PATTERN.matcher(hardwareHints).find();
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
