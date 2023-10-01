package com.rnwhisper;

import android.util.Log;

import java.util.ArrayList;
import java.lang.StringBuilder;
import java.io.IOException;
import java.io.FileReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

public class AudioUtils {
  private static final String NAME = "RNWhisperAudioUtils";

  private static final int SAMPLE_RATE = 16000;

  private static byte[] shortToByte(short[] shortInts) {
    int j = 0;
    int length = shortInts.length;
    byte[] byteData = new byte[length * 2];
    for (int i = 0; i < length; i++) {
      byteData[j++] = (byte) (shortInts[i] >>> 8);
      byteData[j++] = (byte) (shortInts[i] >>> 0);
    }
    return byteData;
  }

  public static byte[] concatShortBuffers(ArrayList<short[]> buffers) {
    int totalLength = 0;
    for (int i = 0; i < buffers.size(); i++) {
      totalLength += buffers.get(i).length;
    }
    byte[] result = new byte[totalLength * 2];
    int offset = 0;
    for (int i = 0; i < buffers.size(); i++) {
      byte[] bytes = shortToByte(buffers.get(i));
      System.arraycopy(bytes, 0, result, offset, bytes.length);
      offset += bytes.length;
    }

    return result;
  }

  private static byte[] removeTrailingZeros(byte[] audioData) {
    int i = audioData.length - 1;
    while (i >= 0 && audioData[i] == 0) {
      --i;
    }
    byte[] newData = new byte[i + 1];
    System.arraycopy(audioData, 0, newData, 0, i + 1);
    return newData;
  }

  public static void saveWavFile(byte[] rawData, String audioOutputFile) throws IOException {
    Log.d(NAME, "call saveWavFile");
    rawData = removeTrailingZeros(rawData);
    DataOutputStream output = null;
    try {
      output = new DataOutputStream(new FileOutputStream(audioOutputFile));
      // WAVE header
      // see http://ccrma.stanford.edu/courses/422/projects/WaveFormat/
      output.writeBytes("RIFF"); // chunk id
      output.writeInt(Integer.reverseBytes(36 + rawData.length)); // chunk size
      output.writeBytes("WAVE"); // format
      output.writeBytes("fmt "); // subchunk 1 id
      output.writeInt(Integer.reverseBytes(16)); // subchunk 1 size
      output.writeShort(Short.reverseBytes((short) 1)); // audio format (1 = PCM)
      output.writeShort(Short.reverseBytes((short) 1)); // number of channels
      output.writeInt(Integer.reverseBytes(SAMPLE_RATE)); // sample rate
      output.writeInt(Integer.reverseBytes(SAMPLE_RATE * 2)); // byte rate
      output.writeShort(Short.reverseBytes((short) 2)); // block align
      output.writeShort(Short.reverseBytes((short) 16)); // bits per sample
      output.writeBytes("data"); // subchunk 2 id
      output.writeInt(Integer.reverseBytes(rawData.length)); // subchunk 2 size
      // Audio data (conversion big endian -> little endian)
      short[] shorts = new short[rawData.length / 2];
      ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts);
      ByteBuffer bytes = ByteBuffer.allocate(shorts.length * 2);
      for (short s : shorts) {
        bytes.putShort(s);
      }
      Log.d(NAME, "writing audio file: " + audioOutputFile);
      output.write(bytes.array());
    } finally {
      if (output != null) {
        output.close();
      }
    }
  }

  public static float[] decodeWaveFile(InputStream inputStream) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    byte[] buffer = new byte[1024];
    int bytesRead;
    while ((bytesRead = inputStream.read(buffer)) != -1) {
      baos.write(buffer, 0, bytesRead);
    }
    ByteBuffer byteBuffer = ByteBuffer.wrap(baos.toByteArray());
    byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
    byteBuffer.position(44);
    ShortBuffer shortBuffer = byteBuffer.asShortBuffer();
    short[] shortArray = new short[shortBuffer.limit()];
    shortBuffer.get(shortArray);
    float[] floatArray = new float[shortArray.length];
    for (int i = 0; i < shortArray.length; i++) {
      floatArray[i] = ((float) shortArray[i]) / 32767.0f;
      floatArray[i] = Math.max(floatArray[i], -1f);
      floatArray[i] = Math.min(floatArray[i], 1f);
    }
    return floatArray;
  }
}