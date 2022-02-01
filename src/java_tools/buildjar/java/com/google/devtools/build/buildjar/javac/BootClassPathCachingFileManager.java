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

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import com.sun.tools.javac.file.JavacFileManager;
import com.sun.tools.javac.util.Context;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import javax.tools.StandardLocation;

/** A subclass of the JavacFileManager that handles only boot classPaths. */
@VisibleForTesting
class BootClassPathCachingFileManager extends JavacFileManager {

  private final Key key;

  /** Create a JavacFileManager using a given context and BlazeJavacArguments. */
  public BootClassPathCachingFileManager(Context context, Key key) {
    super(context, false, UTF_8);
    this.key = key;
    try {
      this.setLocationFromPaths(
          StandardLocation.PLATFORM_CLASS_PATH, key.bootJarsAndDigest().keySet());
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  public Key getKey() {
    return key;
  }

  /**
   * Checks if the bootClassPaths in {@code BlazeJavacArguments} can benefit from the {@code
   * BootCachingFileManager}. Arguments are not valid if missing boot classpath or at least one
   * digest.
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

  /**
   * A key based on the combination of the bootClassPaths and their digest. This key is unique for
   * each {@code BootClassPathCachingFileManager}.
   */
  @AutoValue
  public abstract static class Key {

    static Key create(BlazeJavacArguments arguments) {
      ImmutableMap.Builder<Path, ByteString> bootClasspathsBuilder = ImmutableMap.builder();
      for (Path bootClassPath : arguments.bootClassPath()) {
        bootClasspathsBuilder.put(
            bootClassPath, arguments.inputsAndDigest().get(bootClassPath.toString()));
      }
      return new AutoValue_BootClassPathCachingFileManager_Key(
          bootClasspathsBuilder.buildOrThrow());
    }

    abstract ImmutableMap<Path, ByteString> bootJarsAndDigest();
  }
}
