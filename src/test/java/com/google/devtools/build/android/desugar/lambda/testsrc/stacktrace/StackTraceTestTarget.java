/*
 * Copyright 2021 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.lambda.testsrc.stacktrace;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

/** Test data class for testing the stack trace related behaviors from lambda desugaring. */
public class StackTraceTestTarget {

  private StackTraceTestTarget() {}

  public static List<String> getStackTraceFileNames() {
    Throwable throwable = new Throwable();
    List<String> sourceFileNames = new ArrayList<>();
    for (StackTraceElement stackTraceElement : throwable.getStackTrace()) {
      sourceFileNames.add(stackTraceElement.getFileName());
    }
    return sourceFileNames;
  }

  public static List<String> getStackTraceFileNamesThroughLambda() throws Exception {
    Callable<List<String>> stackTraceElementsProvider = () -> getStackTraceFileNames();
    return stackTraceElementsProvider.call();
  }

  public static List<String> getStackTraceFileNamesThroughNestedLambda() throws Exception {
    Callable<Callable<List<String>>> stackTraceElementsProviderProvider =
        () -> () -> getStackTraceFileNames();
    return stackTraceElementsProviderProvider.call().call();
  }
}
