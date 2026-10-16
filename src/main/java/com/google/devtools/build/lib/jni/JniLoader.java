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
        case LINUX, FREEBSD, OPENBSD, UNKNOWN -> {
          loadLibrary("main/native/libunix_jni.so");
        }
        case DARWIN -> {
          loadLibrary("main/native/libunix_jni.dylib");
        }
        case WINDOWS -> {
          loadLibrary("main/native/windows/windows_jni.dll");
        }
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
        if (dir != null) {
          dir.toFile().deleteOnExit();
        }
        if (tempFile != null) {
          tempFile.toFile().deleteOnExit();
        }
      }
      throw e;
    }
  }

  private JniLoader() {}

  /**
   * Ensures that the JNI library has been loaded.
   *
   * <p>If the JNI library cannot be loaded, this method returns normally, but the error can be
   * later retrieved via {@link #getJniLoadError()}. This makes it possible for this method to be
   * called during static initialization, while delaying the failure to a later stage where we're in
   * a better position to display an error message (see {@link BlazeRuntime#main()}).
   */
  public static void loadJni() {
    // No-op: loading occurs in the static initializer.
  }

  /**
   * Ensures that the JNI library has been loaded and returns the exception thrown while loading it,
   * if any.
   */
  @Nullable
  public static Throwable getJniLoadError() {
    return JNI_LOAD_ERROR;
  }
}
