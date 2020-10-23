// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.outputfilter;

import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the {@code --output_filter} option. */
@RunWith(JUnit4.class)
public class OutputFilterTest extends BuildIntegrationTestCase {
  private EventCollector stderr = new EventCollector(EventKind.STDERR);

  // Cast warnings are silenced by default.
  private void enableCastWarnings() throws Exception {
    addOptions("--javacopt=\"-Xlint:cast\"");
  }

  // Deprecation warnings are silenced by default.
  private void enableDeprecationWarnings() throws Exception {
    addOptions("--javacopt=\"-Xlint:deprecation\"");
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(new OutputFilteringModule());
  }

  @Before
  public final void writeFiles() throws Exception  {
    write("java/a/BUILD",
        "java_library(name = 'a',",
        "            srcs = ['A.java']," +
        "            deps = ['//java/b'])");
    write("java/a/A.java",
        "package a;",
        "public class A {",
        "  public static void a() { b.B.b(); }",
        "}");
    write("java/b/BUILD",
        "java_library(name = 'b',",
        "             srcs = ['B.java']," +
        "             deps = ['//java/c'])");
    write("java/b/B.java",
        "package b;",
        "public class B {",
        "  @Deprecated public static void b() {}",
        "  public static void x() { c.C.c(); }",
        "}");
    write("java/c/BUILD",
        "java_library(name = 'c',",
        "             srcs = ['C.java'])");
    write("java/c/C.java",
        "package c;",
        "public class C {",
        "  @Deprecated public static void c() {}",
        "}");
    write("java/d/BUILD",
        "java_library(name = 'd',",
        "             srcs = ['D.java'],",
        "             deps = ['//java/e'])");
    write(
        "java/d/D.java",
        "package d;",
        "import java.lang.Integer;",
        "import java.util.ArrayList;",
        "public class D {",
        "  public static void d() {",
        "    int i = (int) 0;",
        "    e.E.e();",
        "  }",
        "}");
    write("java/e/BUILD",
        "java_library(name = 'e',",
        "             srcs = ['E.java'])");
    write(
        "java/e/E.java",
        "package e;",
        "import java.lang.Integer;",
        "import java.util.LinkedList;",
        "public class E {",
        "  public static void e() {",
        "    int i = (int) 0;",
        "  }",
        "}");
    write("javatests/a/BUILD",
        "java_library(name = 'a',",
        "             srcs = ['ATest.java']," +
        "             deps = ['//java/a', '//javatests/b'])");
    write("javatests/a/ATest.java",
        "package a;",
        "public class ATest {",
        "  public static void aTest() { a.A.a(); }",
        "}");
    write("javatests/b/BUILD",
        "java_library(name = 'b',",
        "             srcs = ['BTest.java']," +
        "             deps = ['//java/b', '//javatests/c'])");
    write("javatests/b/BTest.java",
        "package b;",
        "public class BTest {",
        "  public static void bTest() { c.CTest.c(); }",
        "}");
    write("javatests/c/BUILD",
        "java_library(name = 'c',",
        "             srcs = ['CTest.java'])");
    write("javatests/c/CTest.java",
        "package c;",
        "public class CTest {",
        "  @Deprecated public static void c() {}",
        "}");
    write("javatests/d/BUILD",
        "java_library(name = 'd',",
        "             srcs = ['DTest.java'],",
        "             deps = ['//java/d', '//javatests/e'])");
    write("javatests/d/DTest.java",
        "package d;",
        "public class DTest {",
        "  public static void dTest() { d.D.d(); }",
        "}");
    write("javatests/e/BUILD",
        "java_library(name = 'e',",
        "             srcs = ['ETest.java'])");
    write(
        "javatests/e/ETest.java",
        "package e;",
        "import java.lang.Integer;",
        "import java.util.LinkedList;",
        "public class ETest {",
        "  public static void eTest() {",
        "    int i = (int) 0;",
        "  }",
        "}");

    // Always enable cast warnings.
    enableCastWarnings();
  }

  @Test
  public void testExplicitFilter() throws Exception {
    enableDeprecationWarnings();
    addOptions("--output_filter=^//java/a");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/a");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertNoEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testExplicitFilterNoJavacoptOverride() throws Exception {
    addOptions("--output_filter=^//java/d");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/d");

    assertEvent("D.java:6: warning: [cast] redundant cast to int");
    assertNoEvent("E.java:6: warning: [cast] redundant cast to int");
  }

  @Test
  public void testPackagesAOF_A() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/a");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertNoEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testPackagesAOF_B() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/b");

    assertNoEvent(deprecationMessages("b", "B", "b"));
    assertEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testPackagesAOF_C() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/c");

    assertNoEvent(deprecationMessages("b", "B", "b"));
    assertNoEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testPackagesAOF_D() throws Exception {
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/d");

    assertEvent("D.java:6: warning: [cast] redundant cast to int");
    assertNoEvent("E.java:6: warning: [cast] redundant cast to int");
  }

  @Test
  public void testPackagesAOF_AB() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/a", "//java/b");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testPackagesAOF_AC() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/a", "//java/c");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertNoEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testPackagesAOF_BC() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/b", "//java/c");

    assertNoEvent(deprecationMessages("b", "B", "b"));
    assertEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testPackagesAOF_ABC() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/a", "//java/b", "//java/c");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testPackagesAOF_javaTestsA() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//javatests/a");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertNoEvent(deprecationMessages("c", "C", "c"));
    assertNoEvent(deprecationMessages("c", "CTest", "c"));
  }

  @Test
  public void testPackagesAOF_javaTestsAB() throws Exception {
    enableDeprecationWarnings();
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//javatests/a", "//java/b");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertEvent(deprecationMessages("c", "C", "c"));
    assertEvent(deprecationMessages("c", "CTest", "c"));
  }

  @Test
  public void testPackagesAOF_javaTestsD() throws Exception {
    addOptions("--auto_output_filter=packages");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//javatests/d");

    assertEvent("D.java:6: warning: [cast] redundant cast to int");
    assertNoEvent("E.java:6: warning: [cast] redundant cast to int");
    assertNoEvent("ETest.java:6: warning: [cast] redundant cast to int");
  }

  @Test
  public void testEmptyFilter() throws Exception {
    enableDeprecationWarnings();
    addOptions("--output_filter=");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/a");

    assertEvent(deprecationMessages("b", "B", "b"));
    assertEvent(deprecationMessages("c", "C", "c"));
  }

  @Test
  public void testNoMatchFilter() throws Exception {
    enableDeprecationWarnings();
    addOptions("--output_filter=DONT_MATCH");
    CommandEnvironment env = runtimeWrapper.newCommand();
    env.getReporter().addHandler(stderr);
    buildTarget("//java/a");

    assertNoEvent(deprecationMessages("b", "B", "b"));
    assertNoEvent(deprecationMessages("c", "C", "c"));
  }

  private void assertEvent(String... choices) {
    for (Event event : stderr) {
      for (String msg : choices) {
        if (event.getMessage().contains(msg)) {
          return;
        }
      }
    }

    fail(String.format("Expected one of [%s] in output: %s",
        Joiner.on(',').join(choices),
        Joiner.on('\n').join(stderr)
        ));
  }

  private void assertNoEvent(String... choices) {
    for (Event event : stderr) {
      for (String msg : choices) {
        if (event.getMessage().contains(msg)) {
          fail("Event '" + msg + "' was found in output: \n" + Joiner.on('\n').join(stderr));
        }
      }
    }
  }

  private String[] deprecationMessages(String pkg, String clazz, String method) {
    return new String[] {
        String.format("%s() in %s.%s has been deprecated", method, pkg, clazz), // javac6
        String.format("%s() in %s has been deprecated", method, clazz) // javac7
    };
  }
}
