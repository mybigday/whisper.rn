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

    public WritableArray detectSpeechWithAudioData(float[] audioData, int numSamples, ReadableMap options) throws Exception {
        if (vadContext == 0) {
            throw new Exception("VAD context is null");
        }

        return processVadDetection(audioData, numSamples, options);
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
            WhisperContext.freeVadContext(id, vadContext);
            vadContext = 0;
        }
    }
}
