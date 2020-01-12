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

import com.google.common.io.Files;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.List;

/** A utility class for running Java tests using persistent workers. */
final class PersistentTestRunner {

  private PersistentTestRunner() {
    // utility class; should not be instantiated
  }

  /**
   * Runs new tests in the same process. Communicates with bazel using the worker's protocol. Reads
   * a {@link WorkRequest} sent by bazel and sends back a {@link WorkResponse}.
   *
   * <p>Before running a test it creates a new classloader loading the test and its dependencies'
   * classes. A class already loaded in a classloader can not be unloaded. To overcome this issue a
   * new classloader has to be created at every test run.
   */
  static int runPersistentTestRunner(
      String suitClassName, String workspacePrefix, SuiteTestRunner suiteTestRunner) {
    PrintStream originalStdOut = System.out;
    PrintStream originalStdErr = System.err;

    String testRuntimeClasspathFile = System.getenv("TEST_RUNTIME_CLASSPATH_FILE");
    String javaRunfilesPath = System.getenv("JAVA_RUNFILES");

    // Reading the work requests and solving them in sequence is not a problem because Bazel creates
    // up to --worker_max_instances (defaults to 4) instances per worker key.
    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);

        if (request == null) {
          // null is only returned when the stream reaches EOF
          break;
        }

        URLClassLoader testRunnerClassLoader =
            recreateClassLoader(testRuntimeClasspathFile, javaRunfilesPath, workspacePrefix);

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream printStream = new PrintStream(outputStream, true);
        System.setOut(printStream);
        System.setErr(printStream);
        String[] arguments = request.getArgumentsList().toArray(new String[0]);
        int exitCode = -1;
        try {
          exitCode =
              suiteTestRunner.runTestsInSuite(
                  suitClassName, arguments, testRunnerClassLoader, /* resolve= */ true);
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
   * Returns a classloader containing the jars read from the given runtime classpath file.
   *
   * <p>Sets the classloader of the current thread to the newly created classloader.
   *
   * <p>Sets the classloader used to load the test classes and their dependencies.
   *
   * <p>Needs to be called before every test run to avoid having stale classes. A class already
   * loaded in a classloader can not be unloaded. To overcome this a new classloader has to be
   * created at every test run.
   */
  private static URLClassLoader recreateClassLoader(
      String runtimeClasspathFilename, String javaRunfilesPath, String workspacePrefix)
      throws IOException {
    URLClassLoader classLoader =
        new URLClassLoader(
            createURLsFromRelativePathsInFile(
                runtimeClasspathFilename, javaRunfilesPath, workspacePrefix));
    Thread.currentThread().setContextClassLoader(classLoader);
    return classLoader;
  }

  private static URL[] createURLsFromRelativePathsInFile(
      String runtimeClasspathFilename, String javaRunfilesPath, String workspacePrefix)
      throws IOException {
    List<String> testRuntimeClasspath = Files.readLines(new File(runtimeClasspathFilename), UTF_8);
    ArrayList<URL> urlList = new ArrayList<>();
    for (String classPathEntry : testRuntimeClasspath) {
      urlList.add(
          new File(javaRunfilesPath + File.separator + workspacePrefix + classPathEntry)
              .toURI()
              .toURL());
    }
    URL[] urls = new URL[urlList.size()];
    return urlList.toArray(urls);
  }

  static boolean isPersistentTestRunner() {
    return Boolean.parseBoolean(System.getenv("PERSISTENT_TEST_RUNNER"));
  }
}
