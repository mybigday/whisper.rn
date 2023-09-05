//
// whisper-decoder-impl.h
//
// This file was automatically generated and should not be edited.
//

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include <stdint.h>
#include <os/log.h>

NS_ASSUME_NONNULL_BEGIN


/// Model Prediction Input Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface whisper_decoder_implInput : NSObject<MLFeatureProvider>

/// token_data as 1 by 1 matrix of 32-bit integers
@property (readwrite, nonatomic, strong) MLMultiArray * token_data;

/// audio_data as 1 × 384 × 1 × 1500 4-dimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * audio_data;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithToken_data:(MLMultiArray *)token_data audio_data:(MLMultiArray *)audio_data NS_DESIGNATED_INITIALIZER;

@end


/// Model Prediction Output Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface whisper_decoder_implOutput : NSObject<MLFeatureProvider>

/// var_1346 as multidimensional array of floats
@property (readwrite, nonatomic, strong) MLMultiArray * var_1346;
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithVar_1346:(MLMultiArray *)var_1346 NS_DESIGNATED_INITIALIZER;

@end


/// Class for model loading and prediction
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0)) __attribute__((visibility("hidden")))
@interface whisper_decoder_impl : NSObject
@property (readonly, nonatomic, nullable) MLModel * model;

/**
    URL of the underlying .mlmodelc directory.
*/
+ (nullable NSURL *)URLOfModelInThisBundle;

/**
    Initialize whisper_decoder_impl instance from an existing MLModel object.

    Usually the application does not use this initializer unless it makes a subclass of whisper_decoder_impl.
    Such application may want to use `-[MLModel initWithContentsOfURL:configuration:error:]` and `+URLOfModelInThisBundle` to create a MLModel object to pass-in.
*/
- (instancetype)initWithMLModel:(MLModel *)model NS_DESIGNATED_INITIALIZER;

/**
    Initialize whisper_decoder_impl instance with the model in this bundle.
*/
- (nullable instancetype)init;

/**
    Initialize whisper_decoder_impl instance with the model in this bundle.

    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithConfiguration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize whisper_decoder_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for whisper_decoder_impl.
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Initialize whisper_decoder_impl instance from the model URL.

    @param modelURL URL to the .mlmodelc directory for whisper_decoder_impl.
    @param configuration The model configuration object
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
*/
- (nullable instancetype)initWithContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Construct whisper_decoder_impl instance asynchronously with configuration.
    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid whisper_decoder_impl instance or NSError object.
*/
+ (void)loadWithConfiguration:(MLModelConfiguration *)configuration completionHandler:(void (^)(whisper_decoder_impl * _Nullable model, NSError * _Nullable error))handler;

/**
    Construct whisper_decoder_impl instance asynchronously with URL of .mlmodelc directory and optional configuration.

    Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

    @param modelURL The model URL.
    @param configuration The model configuration
    @param handler When the model load completes successfully or unsuccessfully, the completion handler is invoked with a valid whisper_decoder_impl instance or NSError object.
*/
+ (void)loadContentsOfURL:(NSURL *)modelURL configuration:(MLModelConfiguration *)configuration completionHandler:(void (^)(whisper_decoder_impl * _Nullable model, NSError * _Nullable error))handler;

/**
    Make a prediction using the standard interface
    @param input an instance of whisper_decoder_implInput to predict from
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as whisper_decoder_implOutput
*/
- (nullable whisper_decoder_implOutput *)predictionFromFeatures:(whisper_decoder_implInput *)input error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the standard interface
    @param input an instance of whisper_decoder_implInput to predict from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as whisper_decoder_implOutput
*/
- (nullable whisper_decoder_implOutput *)predictionFromFeatures:(whisper_decoder_implInput *)input options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Make a prediction using the convenience interface
    @param token_data as 1 by 1 matrix of 32-bit integers:
    @param audio_data as 1 × 384 × 1 × 1500 4-dimensional array of floats:
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the prediction as whisper_decoder_implOutput
*/
- (nullable whisper_decoder_implOutput *)predictionFromToken_data:(MLMultiArray *)token_data audio_data:(MLMultiArray *)audio_data error:(NSError * _Nullable __autoreleasing * _Nullable)error;

/**
    Batch prediction
    @param inputArray array of whisper_decoder_implInput instances to obtain predictions from
    @param options prediction options
    @param error If an error occurs, upon return contains an NSError object that describes the problem. If you are not interested in possible errors, pass in NULL.
    @return the predictions as NSArray<whisper_decoder_implOutput *>
*/
- (nullable NSArray<whisper_decoder_implOutput *> *)predictionsFromInputs:(NSArray<whisper_decoder_implInput*> *)inputArray options:(MLPredictionOptions *)options error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
