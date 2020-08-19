// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.android.tools.r8;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * Class for accessing package private members of R8, which is used in the Bazel integration, but
 * are not in the public API.
 */
public class CompatDxSupport {
  public static void run(D8Command command) throws CompilationFailedException {
    try {
      // bazel can point to both an full r8.jar and to the shrunken r8.jar (r8lib.jar), as the
      // r8.jar is currently referenced from the Android SDK build-tools, where different versions
      // have either full or shrunken jar. From build-tools version 30.0.1 the shrunken jar is
      // shipped. If the full jar is used some additional internal APIs are used for additional
      // configuration of "check main dex" and "minimal main dex" flags which are not
      // supported in the public D8 API.
      Class<?> androidAppClass = Class.forName("com.android.tools.r8.utils.AndroidApp");
      Class<?> internalOptionsClass = Class.forName("com.android.tools.r8.utils.InternalOptions");
      runOnFullJar(command, androidAppClass, internalOptionsClass);
    } catch (ClassNotFoundException e) {
      D8.run(command);
    }
  }

  public static void runOnFullJar(
      D8Command command, Class<?> androidAppClass, Class<?> internalOptionsClass) {
    Method getInputAppMethod;
    Method getInternalOptionsMethod;
    Method runForTestingMethod;
    try {
      getInputAppMethod = BaseCommand.class.getDeclaredMethod("getInputApp");
      getInternalOptionsMethod = D8Command.class.getDeclaredMethod("getInternalOptions");
      runForTestingMethod =
          D8.class.getDeclaredMethod("runForTesting", androidAppClass, internalOptionsClass);
    } catch (NoSuchMethodException e) {
      throw new AssertionError("Unsupported r8.jar", e);
    }

    try {
      // Use reflection for:
      //   <code>AndroidApp app = command.getInputApp();</code>
      //   <code>InternalOptions options = command.getInternalOptions();</code>
      // as bazel might link to a shrunken r8.jar which does not have these APIs.
      Object app = getInputAppMethod.invoke(command);
      Object options = getInternalOptionsMethod.invoke(command);
      // DX allows --multi-dex without specifying a main dex list for legacy devices.
      // That is broken, but for CompatDX we do the same to not break existing builds
      // that are trying to transition.
      try {
        Field enableMainDexListCheckField = internalOptionsClass.getField("enableMainDexListCheck");
        try {
          // Use reflection for:
          //   <code>options.enableMainDexListCheck = false;</code>
          // as bazel might link to an old r8.jar which does not have this field.
          enableMainDexListCheckField.setBoolean(options, false);
        } catch (IllegalAccessException e) {
          throw new AssertionError("Unsupported r8.jar", e);
        }
      } catch (NoSuchFieldException e) {
        // Ignore if bazel is linking to an old r8.jar.
      }

      runForTestingMethod.invoke(null, app, options);
    } catch (ReflectiveOperationException e) {
      // This is an unsupported r8.jar.
      throw new AssertionError("Unsupported r8.jar", e);
    }
  }

  public static void enableDesugarBackportStatics(D8Command.Builder builder) {
    // Use reflection for:
    //   <code>builder.enableDesugarBackportStatics();</code>
    // as bazel might link to an old r8.jar which does not have this field.
    try {
      Method enableDesugarBackportStatics =
          builder.getClass().getMethod("enableDesugarBackportStatics");
      try {
        enableDesugarBackportStatics.invoke(builder);
      } catch (ReflectiveOperationException e) {
        throw new AssertionError("Unsupported r8.jar", e);
      }
    } catch (NoSuchMethodException e) {
      // Ignore if bazel is linking to an old r8.jar.
    }
  }

  private CompatDxSupport() {}
}
