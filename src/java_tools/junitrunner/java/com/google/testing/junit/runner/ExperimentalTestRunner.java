// Copyright 2017 The Bazel Authors. All Rights Reserved.
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

import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.testing.junit.runner.internal.StackTraces;
import com.google.testing.junit.runner.junit4.JUnit4InstanceModules.Config;
import com.google.testing.junit.runner.junit4.JUnit4InstanceModules.SuiteClass;
import com.google.testing.junit.runner.junit4.JUnit4Runner;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.TimeUnit;

/**
 * Like {@link BazelTestRunner} but runs the tests in their own classloader.
 */
public class ExperimentalTestRunner {
  /**
   * If no arguments are passed on the command line, use this System property to
   * determine which test suite to run.
   */
  static final String TEST_SUITE_PROPERTY_NAME = "bazel.test_suite";

  private static URL[] classpaths = null;
  private static URLClassLoader targetClassLoader;

  private ExperimentalTestRunner() {
    // utility class; should not be instantiated
  }

  /**
   * Takes as arguments the classes or packages to test.
   *
   * <p>To help just run one test or method in a suite, the test suite
   * may be passed in via system properties (-Dbazel.test_suite).
   * An empty args parameter means to run all tests in the suite.
   * A non-empty args parameter means to run only the specified tests/methods.
   *
   * <p>Return codes:
   * <ul>
   * <li>Test runner failure, bad arguments, etc.: exit code of 2</li>
   * <li>Normal test failure: exit code of 1</li>
   * <li>All tests pass: exit code of 0</li>
   * </ul>
   */
  public static void main(String[] args) {
    PrintStream stderr = System.err;

    String suiteClassName = System.getProperty(TEST_SUITE_PROPERTY_NAME);

    if ("true".equals(System.getenv("PERSISTENT_TEST_RUNNER"))) {
      System.exit(runPersistentTestRunner(suiteClassName));
    }

    System.out.println("WARNING: RUNNING EXPERIMENTAL TEST RUNNER");

    if (!checkTestSuiteProperty(suiteClassName)) {
      System.exit(2);
    }

    int exitCode;
    try {
      exitCode = runTestsInSuite(suiteClassName, args);
    } catch (Throwable e) {
      // An exception was thrown by the runner. Print the error to the output stream so it will be logged
      // by the executing strategy, and return a failure, so this process can gracefully shut down.
      e.printStackTrace(System.err);
      exitCode = 1;
    }

    System.err.printf("%nExperimentalTestRunner exiting with a return value of %d%n", exitCode);
    System.err.println("JVM shutdown hooks (if any) will run now.");
    System.err.println("The JVM will exit once they complete.");
    System.err.println();

    printStackTracesIfJvmExitHangs(stderr);

    DateFormat format = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    Date shutdownTime = new Date();
    String formattedShutdownTime = format.format(shutdownTime);
    System.err.printf("-- JVM shutdown starting at %s --%n%n", formattedShutdownTime);
    System.exit(exitCode);
  }

  /**
   * Ensures that the bazel.test_suite in argument is not {@code null} or print error and
   * explanation.
   *
   * @param testSuiteProperty system property to check
   */
  private static boolean checkTestSuiteProperty(String testSuiteProperty) {
    if (testSuiteProperty == null) {
      System.err.printf(
          "Error: The test suite Java system property %s is required but missing.%n",
          TEST_SUITE_PROPERTY_NAME);
      System.err.println();
      System.err.println("This property is set automatically when running with Bazel like such:");
      System.err.printf("  java -D%s=[test-suite-class] %s%n",
          TEST_SUITE_PROPERTY_NAME, ExperimentalTestRunner.class.getName());
      System.err.printf("  java -D%s=[test-suite-class] -jar [deploy-jar]%n",
          TEST_SUITE_PROPERTY_NAME);
      System.err.println("E.g.:");
      System.err.printf("  java -D%s=org.example.testing.junit.runner.SmallTests %s%n",
          TEST_SUITE_PROPERTY_NAME, ExperimentalTestRunner.class.getName());
      System.err.printf("  java -D%s=org.example.testing.junit.runner.SmallTests "
              + "-jar SmallTests_deploy.jar%n",
          TEST_SUITE_PROPERTY_NAME);
      return false;
    }
    return true;
  }

  private static int runTestsInSuite(String suiteClassName, String[] args) {
    Class<?> suite = getTargetSuiteClass(suiteClassName);

    if (suite == null) {
      // No class found corresponding to the system property passed in from Bazel
      if (args.length == 0 && suiteClassName != null) {
        System.err.printf("Class not found: [%s]%n", suiteClassName);
        return 2;
      }
    }

    JUnit4Runner runner =
        JUnit4Bazel.builder()
            .suiteClass(new SuiteClass(suite))
            .config(new Config(args))
            .build()
            .runner();

    // Some frameworks such as Mockito use the Thread's context classloader.
    Thread.currentThread().setContextClassLoader(targetClassLoader);

    int result = 1;
    try {
      result = runner.run().wasSuccessful() ? 0 : 1;
    } catch (RuntimeException e) {
      System.err.println("Test run failed with exception");
      e.printStackTrace();
    }
    return result;
  }

  /**
   * Run in a loop awaiting instructions for the next test run.
   *
   * @param suiteClassName name of the class which is passed on to JUnit to determine the test suite
   * @return 0 when we encounter an EOF from input, or non-zero values if we encounter an
   *     unrecoverable error.
   */
  private static int runPersistentTestRunner(String suiteClassName) {
    PrintStream originalStdOut = System.out;
    PrintStream originalStdErr = System.err;

    while (true) {
      try {
        WorkRequest request = WorkRequest.parseDelimitedFrom(System.in);

        if (request == null) {
          break;
        }
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PrintStream printStream = new PrintStream(outputStream, true);
        System.setOut(printStream);
        System.setErr(printStream);
        String[] arguments = request.getArgumentsList().toArray(new String[0]);
        int exitCode = -1;
        try {
          exitCode = runTestsInSuite(suiteClassName, arguments);
        } finally {
          System.setOut(originalStdOut);
          System.setErr(originalStdErr);
        }

        WorkResponse response =
            WorkResponse
                .newBuilder()
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
   * Get the actual Test Suite class corresponding to the given name.
   */
  private static Class<?> getTargetSuiteClass(String suiteClassName) {
    if (suiteClassName == null) {
      return null;
    }

    try {
      targetClassLoader = new URLClassLoader(getClasspaths());
      Class<?> targetSuiteClass = targetClassLoader.loadClass(suiteClassName);
      System.out.printf(
          "Running test suites for class: %s, created by classLoader: %s%n",
          targetSuiteClass, targetSuiteClass.getClassLoader());
      return targetSuiteClass;
    } catch (ClassNotFoundException | IOException e) {
      System.err.println("Exception in loading class:" + e.getMessage());
      return null;
    }
  }

  /**
   * Used to get the classpaths which should be used to load the classes of the test target.
   *
   * @throws MalformedURLException when we are unable to create a given classpath.
   * @return array of URLs containing the classpaths or null if classpaths could not be located.
   */
  private static URL[] getClasspaths() throws IOException {
    if (classpaths != null) {
      return classpaths;
    }

    String selfLocation = System.getenv("SELF_LOCATION");
    // TODO(kush): Get this to work for windows style paths.
    String classpathFileLocation = selfLocation + "_classpaths_file";
    Path file = Paths.get(classpathFileLocation);
    byte[] classPathFileBytes = null;
    try {
      classPathFileBytes = Files.readAllBytes(file);
    } catch (IOException e) {
      System.err.println("exception in reading file:" + e.getMessage());
      throw e;
    }

    String classloaderPrefixPath = System.getenv("CLASSLOADER_PREFIX_PATH");
    String workingDir = System.getProperty("user.dir");
    String[] targetClassPaths = new String(classPathFileBytes, StandardCharsets.UTF_8).split(":");
    classpaths = new URL[targetClassPaths.length];

    String locationPrefix = "file://" + workingDir + "/" + classloaderPrefixPath;

    for (int index = 0; index < targetClassPaths.length; index++) {
      try {
        classpaths[index] = new URL(locationPrefix + targetClassPaths[index]);
      } catch (MalformedURLException e) {
        System.err.println("Unable to create URL for:" + targetClassPaths[index]);
        throw e;
      }
    }
    return classpaths;
  }

  /**
   * Prints out stack traces if the JVM does not exit quickly. This can help detect shutdown hooks
   * that are preventing the JVM from exiting quickly.
   *
   * @param out Print stream to use
   */
  private static void printStackTracesIfJvmExitHangs(final PrintStream out) {
    Thread thread = new Thread(new Runnable() {
      @Override
      public void run() {
        sleepUninterruptibly(5);
        out.println("JVM still up after five seconds. Dumping stack traces for all threads.");
        StackTraces.printAll(out);
      }
    }, "ExperimentalTestRunner: Print stack traces if JVM exit hangs");

    thread.setDaemon(true);
    thread.start();
  }

  /**
   * Invokes SECONDS.{@link TimeUnit#sleep(long) sleep(sleepForSeconds)} uninterruptibly.
   */
  private static void sleepUninterruptibly(long sleepForSeconds) {
    boolean interrupted = false;
    try {
      long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(sleepForSeconds);
      while (true) {
        try {
          // TimeUnit.sleep() treats negative timeouts just like zero.
          TimeUnit.NANOSECONDS.sleep(end - System.nanoTime());
          return;
        } catch (InterruptedException e) {
          interrupted = true;
        }
      }
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }
}
