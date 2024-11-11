package com.rnwhisper;

import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Base64;

import java.util.Arrays;

public class AudioUtils {
  private static final String NAME = "RNWhisperAudioUtils";

  private static float[] bufferToFloatArray(byte[] buffer, Boolean cutHeader) {
    ByteBuffer byteBuffer = ByteBuffer.wrap(buffer);
    byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
    ShortBuffer shortBuffer = byteBuffer.asShortBuffer();
    short[] shortArray = new short[shortBuffer.limit()];
    shortBuffer.get(shortArray);
    if (cutHeader) {
      shortArray = Arrays.copyOfRange(shortArray, 44, shortArray.length);
    }
    float[] floatArray = new float[shortArray.length];
    for (int i = 0; i < shortArray.length; i++) {
      floatArray[i] = ((float) shortArray[i]) / 32767.0f;
      floatArray[i] = Math.max(floatArray[i], -1f);
      floatArray[i] = Math.min(floatArray[i], 1f);
    }
    return floatArray;
  }

  public static float[] decodeWaveFile(InputStream inputStream) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    byte[] buffer = new byte[1024];
    int bytesRead;
    while ((bytesRead = inputStream.read(buffer)) != -1) {
      baos.write(buffer, 0, bytesRead);
    }
    return bufferToFloatArray(baos.toByteArray(), true);
  }

  public static float[] decodeWaveData(String dataBase64) throws IOException {
    return bufferToFloatArray(Base64.getDecoder().decode(dataBase64), true);
  }

  public static float[] decodePcmData(String dataBase64) {
    return bufferToFloatArray(Base64.getDecoder().decode(dataBase64), false);
  }
}
