package com.rnwhisper;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class WavWriter {
    private FileOutputStream fos;
    private int sampleRate;
    private short channels;
    private short bitsPerSample;
    private int totalSamples = 0;
    private String filePath;
    private boolean isOpen = false;

    public boolean initialize(String filePath, int sampleRate, short channels, short bitsPerSample) {
        try {
            this.filePath = filePath;
            this.sampleRate = sampleRate;
            this.channels = channels;
            this.bitsPerSample = bitsPerSample;
            fos = new FileOutputStream(filePath);
            // Write placeholder header (44 bytes)
            fos.write(new byte[44]);
            isOpen = true;
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public synchronized boolean appendSamples(short[] samples, int count) {
        if (!isOpen) return false;
        try {
            ByteBuffer bb = ByteBuffer.allocate(count * 2);
            bb.order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < count; i++) {
                bb.putShort(samples[i]);
            }
            fos.write(bb.array());
            totalSamples += count;
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public void flush() {
        try {
            if (isOpen) {
                fos.flush();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    // public boolean appendSamples(short[] samples, int count) {
    //     if (!isOpen) return false;
    //     try {
    //         ByteBuffer bb = ByteBuffer.allocate(count * 2);
    //         bb.order(ByteOrder.LITTLE_ENDIAN);
    //         for (int i = 0; i < count; i++) {
    //             bb.putShort(samples[i]);
    //         }
    //         fos.write(bb.array());
    //         totalSamples += count;
    //         return true;
    //     } catch (IOException e) {
    //         e.printStackTrace();
    //         return false;
    //     }
    // }

    public boolean finalizeWav() {
        if (!isOpen) return false;
        try {
            fos.flush();
            fos.close();
            isOpen = false;
            int byteRate = sampleRate * channels * bitsPerSample / 8;
            int dataSize = totalSamples * channels * bitsPerSample / 8;
            int chunkSize = 36 + dataSize;
            RandomAccessFile raf = new RandomAccessFile(filePath, "rw");
            ByteBuffer header = ByteBuffer.allocate(44);
            header.order(ByteOrder.LITTLE_ENDIAN);
            header.put("RIFF".getBytes());
            header.putInt(chunkSize);
            header.put("WAVE".getBytes());
            header.put("fmt ".getBytes());
            header.putInt(16); // PCM
            header.putShort((short) 1); // AudioFormat
            header.putShort(channels);
            header.putInt(sampleRate);
            header.putInt(byteRate);
            header.putShort((short)(channels * bitsPerSample / 8));
            header.putShort(bitsPerSample);
            header.put("data".getBytes());
            header.putInt(dataSize);
            raf.seek(0);
            raf.write(header.array());
            raf.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    
  public boolean finalizeExternalWav(String filePath) throws IOException {
    RandomAccessFile raf = new RandomAccessFile(filePath, "rw");
    long fileSize = raf.length();
    if (fileSize < 44) {
      raf.close();
      return false;
    }

    int dataSize = (int)(fileSize - 44);
    int chunkSize = 36 + dataSize;

    // Update RIFF chunk size
    raf.seek(4);
    raf.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(chunkSize).array());

    // Update data chunk size
    raf.seek(40);
    raf.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(dataSize).array());

    raf.close();
    return true;
  }
}
