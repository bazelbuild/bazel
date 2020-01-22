// Copyright 2019 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.common.hash.Hashing;
import com.google.common.hash.HashingInputStream;
import com.google.common.io.ByteStreams;
import com.google.common.io.Files;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.net.URL;
import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A utility class for running Java tests using persistent workers. */
final class PersistentTestRunner {

  private PersistentTestRunner() {
    // utility class; should not be instantiated
  }

  /**
   * Runs new tests in the same process. Communicates with bazel using the worker's protocol. Reads
   * a {@link WorkRequest} sent by bazel and sends back a {@link WorkResponse}.
   *
   * <p>Uses three different classloaders:
   *
   * <ul>
   *   <li>A classloader for direct dependencies: loads the test's classes and the first two layers
   *       of its dependencies.
   *   <li>A classloader for transitive dependencies: loads the remaining classes from the
   *       transitive dependencies.
   *   <li>The system classloader: initialized at JVM startup time; loads the test runner's classes.
   * </ul>
   *
   * <p>The classloaders have a child-parent relationship: direct CL -> transitive CL -> system CL.
   *
   * <p>The direct dependencies classloader is the current thread's classloader.
   *
   * <p>The transitive dependencies classloader checks if a class was already loaded by the direct
   * dependencies classloader if it did not succeed in loading the class itself. This is required
   * for classes loaded by the transitive classloader that reference classes in the direct
   * dependencies classloader.
   *
   * <p>The default class loading logic in {@link ClassLoader} applies in all other cases.
   *
   * <p>The direct and transitive classloaders are rebuilt before every test run only if the
   * combined hash of the jars to be loaded has changed.
   */
  static int runPersistentTestRunner(
      String suitClassName, String workspacePrefix, SuiteTestRunner suiteTestRunner) {
    PrintStream originalStdOut = System.out;
    PrintStream originalStdErr = System.err;

    // TODO(elenairina): Remove this variable after cl/282553936 is released.
    String legacyTestRuntimeClasspathFile = System.getenv("TEST_RUNTIME_CLASSPATH_FILE");

    String directClasspathFile = System.getenv("TEST_DIRECT_CLASSPATH_FILE");
    String transitiveClasspathFile = System.getenv("TEST_TRANSITIVE_CLASSPATH_FILE");
    String absolutePathPrefix = System.getenv("JAVA_RUNFILES") + File.separator + workspacePrefix;

    PersistentTestRunnerClassLoader transitiveDepsClassLoader = null;
    PersistentTestRunnerClassLoader directDepsClassLoader = null;

    // Reading the work requests and solving them in sequence is not a problem because Bazel creates
    // up to --worker_max_instances (defaults to 4) instances per worker key.
    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);

        if (request == null) {
          // null is only returned when the stream reaches EOF
          break;
        }

        if (legacyTestRuntimeClasspathFile != null) {
          // Re-use the same classloader variable in the legacy case for simplicity.
          directDepsClassLoader =
              maybeRecreateClassLoader(
                  getFilesWithAbsolutePathPrefixFromFile(
                      legacyTestRuntimeClasspathFile, absolutePathPrefix),
                  ClassLoader.getSystemClassLoader(),
                  null);
        } else {
          transitiveDepsClassLoader =
              maybeRecreateClassLoader(
                  getFilesWithAbsolutePathPrefixFromFile(
                      transitiveClasspathFile, absolutePathPrefix),
                  ClassLoader.getSystemClassLoader(),
                  transitiveDepsClassLoader);

          directDepsClassLoader =
              maybeRecreateClassLoader(
                  getFilesWithAbsolutePathPrefixFromFile(directClasspathFile, absolutePathPrefix),
                  transitiveDepsClassLoader,
                  directDepsClassLoader);
          transitiveDepsClassLoader.setChild(directDepsClassLoader);
        }

        Thread.currentThread().setContextClassLoader(directDepsClassLoader);

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream printStream = new PrintStream(outputStream, true);
        System.setOut(printStream);
        System.setErr(printStream);
        String[] arguments = request.getArgumentsList().toArray(new String[0]);
        int exitCode = -1;
        try {
          exitCode =
              suiteTestRunner.runTestsInSuite(
                  suitClassName, arguments, directDepsClassLoader, /* resolve= */ true);
        } finally {
          System.setOut(originalStdOut);
          System.setErr(originalStdErr);
        }

        WorkResponse response =
            WorkResponse.newBuilder()
                .setOutput(outputStream.toString())
                .setExitCode(exitCode)
                .build();

        response.writeDelimitedTo(System.out);
        System.out.flush();
      } catch (IOException e) {
        e.printStackTrace();
        return 1;
      }
    }
    return 0;
  }

  /**
   * Returns a {@link Class} with the given name. Loads the class for the given classloader, or from
   * the system classloader if none is specified.
   */
  static Class<?> getTestClass(String name, ClassLoader classLoader, boolean resolve) {
    if (name == null) {
      return null;
    }

    try {
      if (classLoader == null) {
        return Class.forName(name);
      }
      return Class.forName(name, resolve, classLoader);
    } catch (ClassNotFoundException e) {
      return null;
    }
  }

  /**
   * Returns a classloader that loads the given jars, only if the combined hash of their content is
   * different than the one for the given previous classloader.
   *
   * <p>Returns previousClassLoader if the hashes are the same.
   *
   * <p>Needs to be called before every test run to avoid having stale classes. A class already
   * loaded in a classloader can not be unloaded. To overcome this a new classloader has to be
   * created at every test run.
   */
  private static PersistentTestRunnerClassLoader maybeRecreateClassLoader(
      List<File> runtimeJars,
      ClassLoader parent,
      @Nullable PersistentTestRunnerClassLoader previousClassLoader)
      throws IOException {
    HashCode combinedHash = getCombinedHashForFiles(runtimeJars);
    if (previousClassLoader != null && combinedHash.equals(previousClassLoader.getChecksum())) {
      return previousClassLoader;
    }

    return new PersistentTestRunnerClassLoader(
        convertFileListToURLArray(runtimeJars), parent, combinedHash);
  }

  private static HashCode getCombinedHashForFiles(List<File> files) throws IOException {
    Hasher hasher = Hashing.sha256().newHasher();
    for (File file : files) {
      hasher.putBytes(getFileHash(file).asBytes());
    }
    return hasher.hash();
  }

  private static HashCode getFileHash(File file) throws IOException {
    InputStream inputStream = new FileInputStream(file);
    HashingInputStream hashingStream = new HashingInputStream(Hashing.sha256(), inputStream);
    ByteStreams.copy(hashingStream, ByteStreams.nullOutputStream());
    return hashingStream.hash();
  }

  private static URL[] convertFileListToURLArray(List<File> jars) throws IOException {
    URL[] urls = new URL[jars.size()];
    for (int i = 0; i < jars.size(); i++) {
      urls[i] = jars.get(i).toURI().toURL();
    }
    return urls;
  }

  private static List<File> getFilesWithAbsolutePathPrefixFromFile(
      String runtimeClasspathFilename, String absolutePathPrefix) throws IOException {
    return Files.readLines(new File(runtimeClasspathFilename), UTF_8).stream()
        .map(entry -> new File(absolutePathPrefix + entry))
        .collect(Collectors.toList());
  }

  static boolean isPersistentTestRunner() {
    return Boolean.parseBoolean(System.getenv("PERSISTENT_TEST_RUNNER"));
  }
}
