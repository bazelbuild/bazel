// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.jni;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.flogger.GoogleLogger;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.util.OS;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.annotation.Nullable;

/** Generic code to interact with the platform-specific JNI code bundle. */
public final class JniLoader {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  @Nullable private static final Throwable JNI_LOAD_ERROR;

  static {
    Throwable jniLoadError;
    try {
      switch (OS.getCurrent()) {
        case LINUX:
        case FREEBSD:
        case OPENBSD:
        case UNKNOWN:
          loadLibrary("main/native/libunix_jni.so");
          break;

        case DARWIN:
          loadLibrary("main/native/libunix_jni.dylib");
          break;

        case WINDOWS:
          try {
            // TODO(jmmv): This is here only for the bootstrapping process, which builds the JNI
            // library and passes a -Djava.library.path to the JVM to find it. I'm sure that this
            // can be replaced by properly bundling the library as a resource in the JAR. For some
            // strange reason that I haven't fully understood yet, this also must come first.
            System.loadLibrary("windows_jni");
          } catch (UnsatisfiedLinkError e) {
            try {
              loadLibrary("main/native/windows/windows_jni.dll");
            } catch (IOException e2) {
              e2.addSuppressed(e);
              throw e2;
            }
          }
          break;
      }
      jniLoadError = null;
    } catch (IOException | UnsatisfiedLinkError e) {
      logger.atWarning().withCause(e).log("Failed to load JNI library");
      jniLoadError = e;
    }
    JNI_LOAD_ERROR = jniLoadError;
  }

  /**
   * Loads a resource as a shared library.
   *
   * @param resourceName the name of the shared library to load, specified as a slash-separated
   *     relative path within the JAR with at least two components
   * @throws IOException if the resource cannot be extracted or loading the library fails for any
   *     other reason
   */
  private static void loadLibrary(String resourceName) throws IOException {
    Path dir = null;
    Path tempFile = null;
    try {
      dir = Files.createTempDirectory("bazel-jni.");
      int slash = resourceName.lastIndexOf('/');
      checkArgument(slash != -1, "resourceName must contain two path components");
      tempFile = dir.resolve(resourceName.substring(slash + 1));

      ClassLoader loader = JniLoader.class.getClassLoader();
      try (InputStream resource = loader.getResourceAsStream(resourceName)) {
        if (resource == null) {
          throw new UnsatisfiedLinkError("Resource " + resourceName + " not in JAR");
        }
        try (OutputStream diskFile = new FileOutputStream(tempFile.toString())) {
          ByteStreams.copy(resource, diskFile);
        }
      }

      System.load(tempFile.toString());

      // Remove the temporary file now that we have loaded it. If we keep it short-lived, we can
      // avoid the file system from persisting it to disk, avoiding an unnecessary cost.
      //
      // Unfortunately, we cannot do this on Windows because the DLL remains open and we don't have
      // a way to specify FILE_SHARE_DELETE in the System.load() call.
      if (OS.getCurrent() != OS.WINDOWS) {
        Files.delete(tempFile);
        tempFile = null;
        Files.delete(dir);
        dir = null;
      }
    } catch (IOException e) {
      try {
        if (tempFile != null) {
          Files.deleteIfExists(tempFile);
        }
        if (dir != null) {
          Files.delete(dir);
        }
      } catch (IOException e2) {
        // Nothing else we can do. Rely on "delete on exit" to try clean things up later on.
      }
      throw e;
    }
  }

  private JniLoader() {}

  /**
   * Triggers the load of the JNI bundle in a platform-independent basis.
   *
   * <p>This does <b>not</b> fail if the JNI bundle cannot be loaded because there are scenarios in
   * which we want to run Bazel without JNI (e.g. during bootstrapping) or are able to fall back to
   * an alternative implementation (e.g. in some filesystem implementations).
   *
   * <p>Callers can check if the JNI bundle was successfully loaded via {@link #isJniAvailable()}
   * and obtain the load error via {@link #getJniLoadError()}.
   */
  public static void loadJni() {}

  /** Returns whether the JNI bundle was successfully loaded. */
  public static boolean isJniAvailable() {
    return JNI_LOAD_ERROR == null;
  }

  /** Returns the exception thrown while loading the JNI bundle, if it failed. */
  @Nullable
  public static Throwable getJniLoadError() {
    return JNI_LOAD_ERROR;
  }
}
