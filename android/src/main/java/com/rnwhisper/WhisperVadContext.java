package com.rnwhisper;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReactApplicationContext;

import android.util.Log;
import android.content.res.AssetManager;
import android.util.Base64;

import java.io.PushbackInputStream;

public class WhisperVadContext {
    public static final String NAME = "RNWhisperVadContext";

    private int id;
    private ReactApplicationContext reactContext;
    private long vadContext;

    public WhisperVadContext(int id, ReactApplicationContext reactContext, long vadContext) {
        this.id = id;
        this.vadContext = vadContext;
        this.reactContext = reactContext;
    }

    public WritableArray detectSpeech(String audioDataBase64, ReadableMap options) throws Exception {
        if (vadContext == 0) {
            throw new Exception("VAD context is null");
        }

        // Decode base64 audio data to float array
        byte[] audioBytes = Base64.decode(audioDataBase64, Base64.DEFAULT);
        int numSamples = audioBytes.length / 4; // 4 bytes per float
        float[] audioData = new float[numSamples];

        for (int i = 0; i < numSamples; i++) {
            int intBits = (audioBytes[i * 4] & 0xFF) |
                         ((audioBytes[i * 4 + 1] & 0xFF) << 8) |
                         ((audioBytes[i * 4 + 2] & 0xFF) << 16) |
                         ((audioBytes[i * 4 + 3] & 0xFF) << 24);
            audioData[i] = Float.intBitsToFloat(intBits);
        }

        return processVadDetection(audioData, numSamples, options);
    }

    public WritableArray detectSpeechFile(String filePathOrBase64, ReadableMap options) throws Exception {
        if (vadContext == 0) {
            throw new Exception("VAD context is null");
        }

        // Follow the same pattern as transcribeFile
        String filePath = filePathOrBase64;

        // Handle HTTP downloads
        if (filePathOrBase64.startsWith("http://") || filePathOrBase64.startsWith("https://")) {
            // Note: This would require access to the downloader, but for now we'll throw an error
            throw new Exception("HTTP URLs not supported in VAD file detection. Please download the file first.");
        }

        float[] audioData;

        // Check for resource identifier (bundled assets)
        int resId = getResourceIdentifier(filePath);
        if (resId > 0) {
            audioData = AudioUtils.decodeWaveFile(reactContext.getResources().openRawResource(resId));
        } else if (filePathOrBase64.startsWith("data:audio/wav;base64,")) {
            // Handle base64 WAV data
            audioData = AudioUtils.decodeWaveData(filePathOrBase64);
        } else {
            // Handle regular file path
            audioData = AudioUtils.decodeWaveFile(new java.io.FileInputStream(new java.io.File(filePath)));
        }

        if (audioData == null) {
            throw new Exception("Failed to load audio file: " + filePathOrBase64);
        }

        return processVadDetection(audioData, audioData.length, options);
    }

    public WritableArray detectSpeechWithAudioData(float[] audioData, ReadableMap options) throws Exception {
        if (vadContext == 0) {
            throw new Exception("VAD context is null");
        }

        return processVadDetection(audioData, audioData.length, options);
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

    private WritableArray processVadDetection(float[] audioData, int numSamples, ReadableMap options) throws Exception {
        // Run VAD detection using WhisperContext static methods
        boolean speechDetected = WhisperContext.vadDetectSpeech(vadContext, audioData, numSamples);
        if (!speechDetected) {
            return Arguments.createArray();
        }

        // Set VAD parameters from options
        float threshold = options.hasKey("threshold") ? (float) options.getDouble("threshold") : 0.5f;
        int minSpeechDurationMs = options.hasKey("minSpeechDurationMs") ? options.getInt("minSpeechDurationMs") : 250;
        int minSilenceDurationMs = options.hasKey("minSilenceDurationMs") ? options.getInt("minSilenceDurationMs") : 100;
        float maxSpeechDurationS = options.hasKey("maxSpeechDurationS") ? (float) options.getDouble("maxSpeechDurationS") : 30.0f;
        int speechPadMs = options.hasKey("speechPadMs") ? options.getInt("speechPadMs") : 30;
        float samplesOverlap = options.hasKey("samplesOverlap") ? (float) options.getDouble("samplesOverlap") : 0.1f;

        // Get segments from VAD using WhisperContext static methods
        long segments = WhisperContext.vadGetSegmentsFromProbs(vadContext, threshold, minSpeechDurationMs,
                                               minSilenceDurationMs, maxSpeechDurationS,
                                               speechPadMs, samplesOverlap);
        if (segments == 0) {
            return Arguments.createArray();
        }

        // Convert segments to WritableArray using WhisperContext static methods
        WritableArray result = Arguments.createArray();
        int nSegments = WhisperContext.vadGetNSegments(segments);

        for (int i = 0; i < nSegments; i++) {
            float t0 = WhisperContext.vadGetSegmentT0(segments, i);
            float t1 = WhisperContext.vadGetSegmentT1(segments, i);

            WritableMap segment = Arguments.createMap();
            segment.putDouble("t0", t0);
            segment.putDouble("t1", t1);
            result.pushMap(segment);
        }

        // Clean up using WhisperContext static methods
        WhisperContext.vadFreeSegments(segments);

        return result;
    }

    public void release() {
        if (vadContext != 0) {
            WhisperContext.freeVadContext(vadContext);
            vadContext = 0;
        }
    }
}
