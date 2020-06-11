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

import com.android.tools.r8.utils.AndroidApp;
import com.android.tools.r8.utils.InternalOptions;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * Class for accessing package private members of R8, which is used in the Bazel integration, but
 * are not in the public API.
 */
public class CompatDxSupport {
  public static void run(D8Command command, boolean minimalMainDex)
      throws CompilationFailedException {
    AndroidApp app = command.getInputApp();
    InternalOptions options = command.getInternalOptions();
    // DX allows --multi-dex without specifying a main dex list for legacy devices.
    // That is broken, but for CompatDX we do the same to not break existing builds
    // that are trying to transition.
    try {
      // Use reflection for:
      //   <code>options.enableMainDexListCheck = false;</code>
      // as bazel might link to an old r8.jar which does not have this field.
      Field enableMainDexListCheck = options.getClass().getField("enableMainDexListCheck");
      try {
        enableMainDexListCheck.setBoolean(options, false);
      } catch (IllegalAccessException e) {
        throw new AssertionError(e);
      }
    } catch (NoSuchFieldException e) {
      // Ignore if bazel is linking to an old r8.jar.
    }

    // DX has a minimal main dex flag. In compat mode only do minimal main dex
    // if the flag is actually set.
    options.minimalMainDex = minimalMainDex;

    D8.runForTesting(app, options);
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
        throw new AssertionError(e);
      }
    } catch (NoSuchMethodException e) {
      // Ignore if bazel is linking to an old r8.jar.
    }
  }

  private CompatDxSupport() {}
}
