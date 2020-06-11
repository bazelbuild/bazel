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
import static java.util.stream.Collectors.toList;

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
import java.net.URLClassLoader;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

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
   *   <li>A classloader for direct dependencies: loads the classes of the test target and its
   *       direct dependencies, *excluding* those dependencies that are also among the rest of the
   *       transitive dependencies
   *   <li>A classloader for transitive dependencies: loads the remaining classes from the
   *       transitive dependencies.
   *   <li>The system classloader: initialized at JVM startup time; loads the test runner's classes.
   * </ul>
   *
   * <p>The classloaders have a child-parent relationship: direct CL -> transitive CL -> system CL.
   *
   * <p>The direct dependencies classloader is the current thread's classloader.
   *
   * <p>For example, given the following dependency graph:
   *
   *                TestTarget
   *                /  |     \
   *               *   *     *
   *              a    b     c
   *              |   *      |
   *              *  /       *
   *               d         e
   *               |
   *               *
   *               f
   *
   * <p>the classloaders load the following: - the direct classloader loads the classes of
   * TestTarget, a and c, which are direct deps - the transitive classloader loads the classes of b,
   * d, e and f, which are transitive deps Note that b is loaded by the transitive classloader even
   * if it is a direct dependency, because b is also a dependency of d, which is a 2nd level
   * dependency of TestTarget.
   *
   * <p>Excluding the direct dependencies that are also present lower in the dependency tree from
   * direct dependencies classloader presents two advantages: 1. reduces the number of classes
   * loaded by directDepsClassLoader, making it faster to create/load 2. avoids additional custom
   * logic for loading classes, since we are now sure that the transitive dependencies classLoader
   * doesn't reference anything from its child classloader
   *
   * <p>The direct and transitive classloaders are rebuilt before every test run only if the
   * combined hash of the jars to be loaded has changed. If the transitive classloader has to be
   * rebuilt, the direct classloader will also be rebuilt to preserve correctness.
   */
  static int runPersistentTestRunner(
      String suitClassName, String workspacePrefix, SuiteTestRunner suiteTestRunner) {
    PrintStream originalStdOut = System.out;
    PrintStream originalStdErr = System.err;

    String directClasspathFile = System.getenv("TEST_DIRECT_CLASSPATH_FILE");
    String transitiveClasspathFile = System.getenv("TEST_TRANSITIVE_CLASSPATH_FILE");
    String absolutePathPrefix = System.getenv("JAVA_RUNFILES") + File.separator + workspacePrefix;

    // Loads the classes of the test target and its direct dependencies *excluding* those
    // dependencies that are also among the rest of the transitive dependencies.
    URLClassLoader directDepsClassLoader = null;
    // Loads all the classes in the transitive dependencies, excluding those loaded
    // by directDepsClassLoader
    URLClassLoader transitiveDepsClassLoader = null;

    HashCode previousTransitiveCombinedHash = null;
    HashCode previousDirectCombinedHash = null;

    // Reading the work requests and solving them in sequence is not a problem because Bazel creates
    // up to --worker_max_instances (defaults to 4) instances per worker key.
    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);

        if (request == null) {
          // null is only returned when the stream reaches EOF
          break;
        }

        Set<File> transitiveDeps =
            getFilesWithAbsolutePathPrefixFromFile(transitiveClasspathFile, absolutePathPrefix);
        Set<File> directDeps =
            getFilesWithAbsolutePathPrefixFromFile(directClasspathFile, absolutePathPrefix);
        // Filter out duplicated dependencies from the directDeps, to ensure that the
        // transitiveDepsClassLoader doesn't reference anything from directDepsClassLoader.
        directDeps = filterOutDupedDeps(directDeps, transitiveDeps);

        HashCode transitiveCombinedHash = getCombinedHashForFiles(transitiveDeps);
        HashCode directCombinedHash = getCombinedHashForFiles(directDeps);

        boolean recreateTransitiveClassloader =
            transitiveDepsClassLoader == null
                || !transitiveCombinedHash.equals(previousTransitiveCombinedHash);

        // if the parent needs to be re-created, the child needs to be recreated also
        boolean recreateDirectClassloader =
            recreateTransitiveClassloader
                || directDepsClassLoader == null
                || !directCombinedHash.equals(previousDirectCombinedHash);

        previousTransitiveCombinedHash = transitiveCombinedHash;
        previousDirectCombinedHash = directCombinedHash;

        if (recreateTransitiveClassloader) {
          transitiveDepsClassLoader =
              new URLClassLoader(
                  convertFileListToURLArray(transitiveDeps), ClassLoader.getSystemClassLoader());
        }

        if (recreateDirectClassloader) {
          directDepsClassLoader =
              new URLClassLoader(convertFileListToURLArray(directDeps), transitiveDepsClassLoader);
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

  private static HashCode getCombinedHashForFiles(Set<File> files) {
    List<HashCode> hashesAsBytes =
        files.stream()
            .parallel()
            .map(file -> getFileHash(file))
            .filter(Objects::nonNull)
            .collect(toList());

    // Update the hasher separately because Hasher.putBytes() is not safe for parallel operations
    Hasher hasher = Hashing.sha256().newHasher();
    for (HashCode hash : hashesAsBytes) {
      hasher.putBytes(hash.asBytes());
    }
    return hasher.hash();
  }

  private static HashCode getFileHash(File file) {
    try {
      InputStream inputStream;
      inputStream = new FileInputStream(file);
      HashingInputStream hashingStream = new HashingInputStream(Hashing.sha256(), inputStream);
      ByteStreams.copy(hashingStream, ByteStreams.nullOutputStream());
      return hashingStream.hash();
    } catch (IOException e) {
      // Throwing RuntimeException to fail the whole build, and still benefit from using parallel
      // streams in getCombinedHashForFiles().
      throw new RuntimeException(e);
    }
  }

  private static URL[] convertFileListToURLArray(Set<File> jars) throws IOException {
    URL[] urls = new URL[jars.size()];
    int it = 0;
    for (File jar : jars) {
      urls[it++] = jar.toURI().toURL();
    }
    return urls;
  }

  private static Set<File> getFilesWithAbsolutePathPrefixFromFile(
      String runtimeClasspathFilename, String absolutePathPrefix) throws IOException {
    return Files.readLines(new File(runtimeClasspathFilename), UTF_8).stream()
        .map(entry -> new File(absolutePathPrefix + entry))
        .collect(Collectors.toSet());
  }

  private static Set<File> filterOutDupedDeps(Set<File> smallSet, Set<File> largeSet) {
    return smallSet.stream()
        .parallel()
        .filter(f -> !largeSet.contains(f))
        .collect(Collectors.toSet());
  }

  static boolean isPersistentTestRunner() {
    return Boolean.parseBoolean(System.getenv("PERSISTENT_TEST_RUNNER"));
  }
}
