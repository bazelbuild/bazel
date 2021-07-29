// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.buildjar.javac;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.protobuf.ByteString;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.util.Context;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/** A subclass of the JavacFileManager that handles only boot classpaths */
@VisibleForTesting
class BootClassPathCachingFileManager extends JavacFileManager {

  private final Map<String, ByteString> bootJarsAndDigest = new HashMap<>();

  /** Create a JavacFileManager using a given context and BlazeJavacArguments */
  public BootClassPathCachingFileManager(Context context, BlazeJavacArguments arguments) {
    super(context, false, UTF_8);

    for (Path bootClassPath : arguments.bootClassPath()) {
      bootJarsAndDigest.put(
          bootClassPath.toString(), arguments.inputsAndDigest().get(bootClassPath.toString()));
    }
  }

  /**
   * Checks if this instance or a new instance is needed for the {@code BlazeJavacArguments}. An
   * update is needed if the bootclasspath in {@code BlazeJavacArguments} have changed its digest.
   */
  @VisibleForTesting
  boolean needsUpdate(BlazeJavacArguments arguments) {
    for (Path bootClassPath : arguments.bootClassPath()) {
      ByteString currDigest = arguments.inputsAndDigest().get(bootClassPath.toString());

      if (currDigest == null) {
        return true;
      }

      ByteString oldDigest = bootJarsAndDigest.putIfAbsent(bootClassPath.toString(), currDigest);

      if (oldDigest != null && !oldDigest.equals(currDigest)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Checks if the bootClassPaths in {@code BlazeJavacArguments} can benefit from the {@code
   * BootCachingFileManager}. Arguments are not valid if missing boot classpath or at least one
   * digest
   */
  @VisibleForTesting
  static boolean areArgumentsValid(BlazeJavacArguments arguments) {
    if (arguments.bootClassPath().isEmpty()) {
      return false;
    }

    for (Path bootClassPath : arguments.bootClassPath()) {
      ByteString currDigest = arguments.inputsAndDigest().get(bootClassPath.toString());
      if (currDigest == null) {
        return false;
      }
    }

    return true;
  }
}
