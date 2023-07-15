package com.rnwhisper;

import android.content.Context;

import java.io.BufferedInputStream;
import java.io.FileOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URL;
import java.net.URLConnection;

/**
 * NOTE: This is simple downloader,
 * the main purpose is supported load assets on RN Debug mode,
 * so it's a very crude implementation.
 * 
 * If you want to use file download in production to load model / audio files,
 * I would recommend using react-native-fs or expo-file-system to manage the files.
 */
public class Downloader {
  private static Context context;

  public Downloader(Context context) {
    this.context = context;
  }

  private String getDir() {
    String dir = context.getCacheDir().getAbsolutePath() + "/rnwhisper_debug_assets/";
    File file = new File(dir);
    if (!file.exists()) {
      file.mkdirs();
    }
    return dir;
  }

  private boolean fileExists(String filename) {
    File file = new File(getDir() + filename);
    return file.exists();
  }

  public String downloadFile(String urlPath) throws Exception {
    String filename = urlPath.substring(urlPath.lastIndexOf('/') + 1);
    if (filename.contains("?")) {
      filename = filename.substring(0, filename.indexOf("?"));
    }
    String filepath = getDir() + filename;
    if (fileExists(filename)) {
      return filepath;
    }
    try {
      URL url = new URL(urlPath);
      URLConnection connection = url.openConnection();
      connection.connect();
      InputStream input = new BufferedInputStream(url.openStream());
      OutputStream output = new FileOutputStream(filepath);
      byte data[] = new byte[1024];
      int count;
      while ((count = input.read(data)) != -1) {
        output.write(data, 0, count);
      }
      output.flush();
      output.close();
      input.close();
    } catch (Exception e) {
      throw e;
    }
    return filepath;
  }

  private void deleteFile(File fileOrDir) {
    if (fileOrDir.isDirectory()) {
      for (File child : fileOrDir.listFiles()) {
        deleteFile(child);
      }
    }
    fileOrDir.delete();
  }

  public void clearCache() {
    deleteFile(new File(getDir()));
  }
}