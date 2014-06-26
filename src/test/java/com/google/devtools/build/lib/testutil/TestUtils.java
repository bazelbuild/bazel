// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.SkyframeMode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

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
  /**
   * A list of all embedded binaries that go into the regular Blaze binary. This is used to
   * fake a list of these because the usual method of scanning the directory tree cannot be used,
   * since we don't have one in tests.
   */
  public static final ImmutableList<String> EMBEDDED_TOOLS = ImmutableList.of(
      "build-runfiles",
      "client_info",
      "process-wrapper",
      "alarm");

  public static final ThreadPoolExecutor POOL =
    (ThreadPoolExecutor) Executors.newFixedThreadPool(10);

  public static final UUID ZERO_UUID = UUID.fromString("00000000-0000-0000-0000-000000000000");

  private TestUtils() {}

  public static boolean sanityChecksEnabled() {
    return false;
  }

  /** Returns the skyframe mode the test class should be run in. */
  public static SkyframeMode skyframeMode(Class<?> clazz) {
    String skyframeProperty = System.getProperty("blaze.skyframe");
    SkyframeMode skyframeMin = Suite.getSkyframeMin(clazz);
    SkyframeMode skyframeMax = Suite.getSkyframeMax(clazz);
    if (skyframeProperty == null) {
      // This most likely means the test is not being run via 'blaze test', e.g. it's being run in
      // an IDE without blaze integration, such as Intellij.
      return skyframeMin;
    }
    SkyframeMode skyframe = SkyframeMode.valueOf(skyframeProperty);
    if (!skyframe.atLeast(skyframeMin) || !skyframe.atMost(skyframeMax)) {
      // This most likely means the test is being run through an inappropriate blaze test target,
      // so we at least try to run with a sensible skyframe mode.
      return skyframeMin;
    }
    return skyframe;
  }

  /** Creates an empty file, along with all its parent directories. */
  public static void makeEmptyFile(Path path) throws IOException {
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.createEmptyFile(path);
  }

  /**
   * Changes the mtime of the file "path", which must exist.  No guarantee is
   * made about the new mtime except that it is different from the previous one.
   *
   * @throws IOException if the mtime could not be read or set.
   */
  public static void changeModtime(Path path)
    throws IOException {
    long prevMtime = path.getLastModifiedTime();
    long newMtime = prevMtime;
    do {
      newMtime += 1000;
      path.setLastModifiedTime(newMtime);
    } while (path.getLastModifiedTime() == prevMtime);
  }

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

  /**
   * Writes a FilesetRule to a String array.
   *
   * @param name the name of the rule.
   * @param out the output directory.
   * @param entries The FilesetEntry entries.
   * @return the String array of the rule.  One String for each line.
   */
  public static String[] createFilesetRule(String name, String out, String... entries) {
    return new String[] {
        String.format("Fileset(name = '%s', out = '%s',", name, out),
                      "        entries = [" +  Joiner.on(", ").join(entries) + "])"
    };
  }

  public static ThreadPoolExecutor getPool() {
    return POOL;
  }

  public static String tmpDir() {
    return tmpDirFile().getAbsolutePath();
  }

  private static String getUserValue(String key) {
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

  public static File undeclaredOutputDir() {
    String dir = System.getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (dir != null) {
      return new File(dir);
    }

    return tmpDirFile();
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

  public static String srcDir() {
    return runfilesDir();
  }

  public static String runfilesDir() {
    File runfilesDir;

    String runfilesDirStr = getUserValue("TEST_SRCDIR");
    if (runfilesDirStr != null && runfilesDirStr.length() > 0) {
      runfilesDir = new File(runfilesDirStr);
    } else {
      // Goal is to find the google3 directory, so we check current
      // directory, then keep backing up until we see google3.
      File dir = new File("");
      while (dir != null) {
        dir = dir.getAbsoluteFile();

        File google3 = new File(dir, "google3");
        if (google3.exists()) {
          return dir.getAbsolutePath();
        }

        dir = dir.getParentFile();
      }

      // Fallback default $CWD/.. works if CWD is //depot/google3
      runfilesDir = new File("").getAbsoluteFile().getParentFile();
    }

    return runfilesDir.getAbsolutePath();
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
        throw new RuntimeException("TEST_RANDOM_SEED must be an integer");
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
}
