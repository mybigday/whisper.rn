package com.rnwhisper;

import androidx.annotation.OptIn;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.common.annotations.FrameworkAPI;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;

public class JSCallInvokerResolver {

  @OptIn(markerClass = FrameworkAPI.class)
  public static CallInvokerHolderImpl getJSCallInvokerHolder(ReactApplicationContext context) {
    try {
      var method = context.getClass().getMethod("getJSCallInvokerHolder");
      return (CallInvokerHolderImpl) method.invoke(context);
    } catch (Exception ignored) {
      // In newer implementations, the method is in CatalystInstance, continue.
    }
    try {
      var catalystInstance = context.getClass().getMethod("getCatalystInstance").invoke(context);
      assert catalystInstance != null;
      var method = catalystInstance.getClass().getMethod("getJSCallInvokerHolder");
      return (CallInvokerHolderImpl) method.invoke(catalystInstance);
    } catch (Exception e) {
      throw new RuntimeException("Failed to get JSCallInvokerHolder", e);
    }
  }

  @OptIn(markerClass = FrameworkAPI.class)
  public static long getJavaScriptContextHolder(ReactApplicationContext context) {
    try {
      var method = context.getClass().getMethod("getJavaScriptContextHolder");
      var holder = method.invoke(context);
      var getMethod = holder.getClass().getMethod("get");
      return (Long) getMethod.invoke(holder);
    } catch (Exception e) {
      throw new RuntimeException("Failed to get JavaScriptContextHolder", e);
    }
  }
}

