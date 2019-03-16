// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.testutil;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Some static utility functions for testing.
 */
public class TestUtils {
  public static final ThreadPoolExecutor POOL =
    (ThreadPoolExecutor) Executors.newFixedThreadPool(10);

  public static final UUID ZERO_UUID = UUID.fromString("00000000-0000-0000-0000-000000000000");

  /**
   * Wait until the {@link System#currentTimeMillis} / 1000 advances.
   *
   * This method takes 0-1000ms to run, 500ms on average.
   */
  public static void advanceCurrentTimeSeconds() throws InterruptedException {
    long currentTimeSeconds = System.currentTimeMillis() / 1000;
    do {
      Thread.sleep(50);
    } while (currentTimeSeconds == System.currentTimeMillis() / 1000);
  }

  public static ThreadPoolExecutor getPool() {
    return POOL;
  }

  public static String tmpDir() {
    return tmpDirFile().getAbsolutePath().replaceAll("\\\\", "/");
  }

  static String getUserValue(String key) {
    String value = System.getProperty(key);
    if (value == null) {
      value = System.getenv(key);
    }
    return value;
  }

  public static File tmpDirFile() {
    File tmpDir;

    // Flag value specified in environment?
    String tmpDirStr = getUserValue("TEST_TMPDIR");

    if (tmpDirStr != null && tmpDirStr.length() > 0) {
      tmpDir = new File(tmpDirStr);
    } else {
      // Fallback default $TEMP/$USER/tmp/$TESTNAME
      String baseTmpDir = System.getProperty("java.io.tmpdir");
      tmpDir = new File(baseTmpDir).getAbsoluteFile();

      // .. Add username
      String username = System.getProperty("user.name");
      username = username.replace('/', '_');
      username = username.replace('\\', '_');
      username = username.replace('\000', '_');
      tmpDir = new File(tmpDir, username);
      tmpDir = new File(tmpDir, "tmp");
    }

    // Ensure tmpDir exists
    if (!tmpDir.isDirectory()) {
      tmpDir.mkdirs();
    }
    return tmpDir;
  }

  public static File makeTempDir() throws IOException {
    File dir = File.createTempFile(TestUtils.class.getName(), ".temp", tmpDirFile());
    if (!dir.delete()) {
      throw new IOException("Cannot remove a temporary file " + dir);
    }
    if (!dir.mkdir()) {
      throw new IOException("Cannot create a temporary directory " + dir);
    }
    return dir;
  }

  public static int getRandomSeed() {
    // Default value if not running under framework
    int randomSeed = 301;

    // Value specified in environment by framework?
    String value = getUserValue("TEST_RANDOM_SEED");
    if ((value != null) && (value.length() > 0)) {
      try {
        randomSeed = Integer.parseInt(value);
      } catch (NumberFormatException e) {
        // throw new AssertionError("TEST_RANDOM_SEED must be an integer");
        throw new RuntimeException("TEST_RANDOM_SEED must be an integer", e);
      }
    }

    return randomSeed;
  }

  public static byte[] serializeObject(Object obj) throws IOException {
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
    try (ObjectOutputStream objectStream = new ObjectOutputStream(outputStream)) {
      objectStream.writeObject(obj);
    }
    return outputStream.toByteArray();
  }

  public static Object deserializeObject(byte[] buf) throws IOException, ClassNotFoundException {
    try (ObjectInputStream inStream = new ObjectInputStream(new ByteArrayInputStream(buf))) {
      return inStream.readObject();
    }
  }

  /**
   * Timeouts for asserting that an arbitrary event occurs eventually.
   *
   * <p>In general, it's not appropriate to use a small constant timeout for an arbitrary
   * computation since there is no guarantee that a snippet of code will execute within a given
   * amount of time - you are at the mercy of the jvm, your machine, and your OS. In theory we
   * could try to take all of these factors into account but instead we took the simpler and
   * obviously correct approach of not having timeouts.
   *
   * <p>If a test that uses these timeout values is failing due to a "timeout" at the
   * 'blaze test' level, it could be because of a legitimate deadlock that would have been caught
   * if the timeout values below were small. So you can rule out such a deadlock by changing these
   * values to small numbers (also note that the --test_timeout blaze flag may be useful).
   */
  public static final long WAIT_TIMEOUT_MILLISECONDS = Long.MAX_VALUE;
  public static final long WAIT_TIMEOUT_SECONDS = WAIT_TIMEOUT_MILLISECONDS / 1000;
}
