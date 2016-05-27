// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.inject.AbstractModule;
import com.google.inject.Guice;
import com.google.inject.Injector;
import com.google.inject.Provides;
import com.google.inject.Singleton;
import com.google.testing.junit.runner.internal.StackTraces;
import com.google.testing.junit.runner.internal.Stderr;
import com.google.testing.junit.runner.internal.Stdout;
import com.google.testing.junit.runner.junit4.JUnit4Runner;
import com.google.testing.junit.runner.junit4.JUnit4RunnerModule;
import com.google.testing.junit.runner.model.AntXmlResultWriter;
import com.google.testing.junit.runner.model.XmlResultWriter;

import org.joda.time.DateTime;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import java.io.PrintStream;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * A class to run JUnit tests in a controlled environment.
 *
 * <p>Currently sets up a security manager to catch undesirable behaviour;
 * System.exit. Also has nice command line options - run with "-help" for
 * details.
 *
 * <p>This class traps writes to <code>System.err.println()</code> and
 * <code>System.out.println()</code> including the output of failed tests in
 * the error report.
 *
 * <p>It also traps SIGTERM signals to make sure that the test report is
 * written when the signal is closed by the unit test framework for running
 * over time.
 */
public class BazelTestRunner {
  /**
   * If no arguments are passed on the command line, use this System property to
   * determine which test suite to run.
   */
  static final String TEST_SUITE_PROPERTY_NAME = "bazel.test_suite";

  private BazelTestRunner() {
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
  public static void main(String args[]) {
    PrintStream stderr = System.err;

    String suiteClassName = System.getProperty(TEST_SUITE_PROPERTY_NAME);

    if (!checkTestSuiteProperty(suiteClassName)) {
      System.exit(2);
    }

    int exitCode = runTestsInSuite(suiteClassName, args);

    System.err.printf("%nBazelTestRunner exiting with a return value of %d%n", exitCode);
    System.err.println("JVM shutdown hooks (if any) will run now.");
    System.err.println("The JVM will exit once they complete.");
    System.err.println();

    printStackTracesIfJvmExitHangs(stderr);

    DateTimeFormatter formatter = DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss");
    DateTime shutdownTime = new DateTime();
    String formattedShutdownTime = formatter.print(shutdownTime);
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
          TEST_SUITE_PROPERTY_NAME, BazelTestRunner.class.getName());
      System.err.printf("  java -D%s=[test-suite-class] -jar [deploy-jar]%n",
          TEST_SUITE_PROPERTY_NAME);
      System.err.println("E.g.:");
      System.err.printf("  java -D%s=org.example.testing.junit.runner.SmallTests %s%n",
          TEST_SUITE_PROPERTY_NAME, BazelTestRunner.class.getName());
      System.err.printf("  java -D%s=org.example.testing.junit.runner.SmallTests "
              + "-jar SmallTests_deploy.jar%n",
          TEST_SUITE_PROPERTY_NAME);
      return false;
    }
    return true;
  }

  private static int runTestsInSuite(String suiteClassName, String[] args) {
    Class<?> suite = getTestClass(suiteClassName);

    if (suite == null) {
      // No class found corresponding to the system property passed in from Bazel
      if (args.length == 0 && suiteClassName != null) {
        System.err.printf("Class not found: [%s]%n", suiteClassName);
        return 2;
      }
    }

    Injector injector = Guice.createInjector(
        new BazelTestRunnerModule(suite, ImmutableList.copyOf(args)));

    JUnit4Runner runner = injector.getInstance(JUnit4Runner.class);
    return runner.run().wasSuccessful() ? 0 : 1;
  }

  private static Class<?> getTestClass(String name) {
    if (name == null) {
      return null;
    }

    try {
      return Class.forName(name);
    } catch (ClassNotFoundException e) {
      return null;
    }
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
        Uninterruptibles.sleepUninterruptibly(5, TimeUnit.SECONDS);
        out.println("JVM still up after five seconds. Dumping stack traces for all threads.");
        StackTraces.printAll(out);
      }
    }, "BazelTestRunner: Print stack traces if JVM exit hangs");

    thread.setDaemon(true);
    thread.start();
  }

  static class BazelTestRunnerModule extends AbstractModule {
    final Class<?> suite;
    final List<String> args;

    BazelTestRunnerModule(Class<?> suite, List<String> args) {
      this.suite = suite;
      this.args = args;
    }

    @Override
    protected void configure() {
      install(JUnit4RunnerModule.create(suite, args));
      bind(XmlResultWriter.class).to(AntXmlResultWriter.class);
    }

    @Provides @Singleton @Stdout
    PrintStream provideStdoutStream() {
      return System.out;
    }

    @Provides @Singleton @Stderr
    PrintStream provideStderrStream() {
      return System.err;
    }
  };
}
