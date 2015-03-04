// Copyright 2006-2015 Google Inc. All Rights Reserved.
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
package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.testutil.JunitTestUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.util.FsApparatus;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.Arrays;

@RunWith(JUnit4.class)
public class BuildFileASTTest {

  private FsApparatus scratch = FsApparatus.newInMemory();

  private EventCollectionApparatus events = new EventCollectionApparatus(EventKind.ALL_EVENTS);

  private class ScratchPathPackageLocator implements CachingPackageLocator {
    @Override
    public Path getBuildFileForPackage(String packageName) {
      return scratch.path(packageName).getRelative("BUILD");
    }
  }

  private CachingPackageLocator locator = new ScratchPathPackageLocator();

  /**
   * Parses the contents of the specified string (using DUMMY_PATH as the fake
   * filename) and returns the AST. Resets the error handler beforehand.
   */
  private BuildFileAST parseBuildFile(String... lines) throws IOException {
    Path file = scratch.file("/a/build/file/BUILD", lines);
    return BuildFileAST.parseBuildFile(file, events.reporter(), locator, false);
  }

  @Test
  public void testParseBuildFileOK() throws Exception {
    Path buildFile = scratch.file("/BUILD",
        "# a file in the build language",
        "",
        "x = [1,2,'foo',4] + [1,2, \"%s%d\" % ('foo', 1)]");

    Environment env = new Environment();
    Reporter reporter = new Reporter();
    BuildFileAST buildfile = BuildFileAST.parseBuildFile(buildFile, reporter, null, false);

    assertTrue(buildfile.exec(env, reporter));

    // Test final environment is correctly modified:
    //
    // input1.BUILD contains:
    // x = [1,2,'foo',4] + [1,2, "%s%d" % ('foo', 1)]
    assertEquals(Arrays.<Object>asList(1, 2, "foo", 4, 1, 2, "foo1"),
                 env.lookup("x"));
  }

  @Test
  public void testEvalException() throws Exception {
    Path buildFile = scratch.file("/input1.BUILD",
        "x = 1",
        "y = [2,3]",
        "",
        "z = x + y");

    Environment env = new Environment();
    Reporter reporter = new Reporter();
    EventCollector collector = new EventCollector(EventKind.ALL_EVENTS);
    reporter.addHandler(collector);
    BuildFileAST buildfile = BuildFileAST.parseBuildFile(buildFile, reporter, null, false);

    assertFalse(buildfile.exec(env, reporter));
    Event e = JunitTestUtils.assertContainsEvent(collector,
        "unsupported operand type(s) for +: 'int' and 'List'");
    assertEquals(4, e.getLocation().getStartLineAndColumn().getLine());
  }

  @Test
  public void testParsesFineWithNewlines() throws Exception {
    BuildFileAST buildFileAST = parseBuildFile("foo()\n"
                                               + "bar()\n"
                                               + "something = baz()\n"
                                               + "bar()");
    assertThat(buildFileAST.getStatements()).hasSize(4);
  }

  @Test
  public void testFailsIfNewlinesAreMissing() throws Exception {
    events.setFailFast(false);

    BuildFileAST buildFileAST =
      parseBuildFile("foo() bar() something = baz() bar()");

    Event event = events.collector().iterator().next();
    assertEquals("syntax error at \'bar\'", event.getMessage());
    assertEquals("/a/build/file/BUILD",
                 event.getLocation().getPath().toString());
    assertEquals(1, event.getLocation().getStartLineAndColumn().getLine());
    assertTrue(buildFileAST.containsErrors());
  }

  @Test
  public void testImplicitStringConcatenationFails() throws Exception {
    events.setFailFast(false);
    BuildFileAST buildFileAST = parseBuildFile("a = 'foo' 'bar'");
    Event event = events.collector().iterator().next();
    assertEquals("Implicit string concatenation is forbidden, use the + operator",
        event.getMessage());
    assertEquals("/a/build/file/BUILD",
                 event.getLocation().getPath().toString());
    assertEquals(1, event.getLocation().getStartLineAndColumn().getLine());
    assertEquals(10, event.getLocation().getStartLineAndColumn().getColumn());
    assertTrue(buildFileAST.containsErrors());
  }

  @Test
  public void testImplicitStringConcatenationAcrossLinesIsIllegal() throws Exception {
    events.setFailFast(false);
    BuildFileAST buildFileAST = parseBuildFile("a = 'foo'\n  'bar'");

    Event event = events.collector().iterator().next();
    assertEquals("indentation error", event.getMessage());
    assertEquals("/a/build/file/BUILD",
                 event.getLocation().getPath().toString());
    assertEquals(2, event.getLocation().getStartLineAndColumn().getLine());
    assertEquals(2, event.getLocation().getStartLineAndColumn().getColumn());
    assertTrue(buildFileAST.containsErrors());
  }

  /**
   * If the specified EventCollector does contain an event which has
   * 'expectedEvent' as a substring, the matching event is
   * returned. Otherwise this will return null.
   */
  public static Event findEvent(EventCollector eventCollector,
                                String expectedEvent) {
    for (Event event : eventCollector) {
      if (event.getMessage().contains(expectedEvent)) {
        return event;
      }
    }
    return null;
  }

  @Test
  public void testWithSyntaxErrorsDoesNotPrintDollarError() throws Exception {
    events.setFailFast(false);
    BuildFileAST buildFile = parseBuildFile(
        "abi = cxx_abi + '-glibc-' + glibc_version + '-' + "
        + "generic_cpu + '-' + sysname",
        "libs = [abi + opt_level + '/lib/libcc.a']",
        "shlibs = [abi + opt_level + '/lib/libcc.so']",
        "+* shlibs", // syntax error at '+'
        "cc_library(name = 'cc',",
        "           srcs = libs,",
        "           includes = [ abi + opt_level + '/include' ])");
    assertTrue(buildFile.containsErrors());
    Event event = events.collector().iterator().next();
    assertEquals("syntax error at '+'", event.getMessage());
    Environment env = new Environment();
    assertFalse(buildFile.exec(env, events.reporter()));
    assertNull(findEvent(events.collector(), "$error$"));
    // This message should not be printed anymore.
    Event event2 = findEvent(events.collector(), "contains syntax error(s)");
    assertNull(event2);
  }

  @Test
  public void testInclude() throws Exception {
    scratch.file("/foo/bar/BUILD",
        "c = 4\n"
        + "d = 5\n");
    Path buildFile = scratch.file("/BUILD",
        "a = 2\n"
        + "include(\"//foo/bar:BUILD\")\n"
        + "b = 4\n");

    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(buildFile, events.reporter(),
                                                            locator, false);

    assertFalse(buildFileAST.containsErrors());
    assertThat(buildFileAST.getStatements()).hasSize(5);
  }

  @Test
  public void testInclude2() throws Exception {
    scratch.file("/foo/bar/defs",
        "a = 1\n");
    Path buildFile = scratch.file("/BUILD",
        "include(\"//foo/bar:defs\")\n"
        + "b = a + 1\n");

    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(buildFile, events.reporter(),
                                                            locator, false);

    assertFalse(buildFileAST.containsErrors());
    assertThat(buildFileAST.getStatements()).hasSize(3);

    Environment env = new Environment();
    Reporter reporter = new Reporter();
    assertFalse(buildFileAST.exec(env, reporter));
    assertEquals(2, env.lookup("b"));
  }

  @Test
  public void testMultipleIncludes() throws Exception {
    String fileA =
        "include(\"//foo:fileB\")\n"
        + "include(\"//foo:fileC\")\n";
    scratch.file("/foo/fileB",
        "b = 3\n"
        + "include(\"//foo:fileD\")\n");
    scratch.file("/foo/fileC",
        "include(\"//foo:fileD\")\n"
        + "c = b + 2\n");
    scratch.file("/foo/fileD",
        "b = b + 1\n"); // this code is included twice

    BuildFileAST buildFileAST = parseBuildFile(fileA);
    assertFalse(buildFileAST.containsErrors());
    assertThat(buildFileAST.getStatements()).hasSize(8);

    Environment env = new Environment();
    Reporter reporter = new Reporter();
    assertFalse(buildFileAST.exec(env, reporter));
    assertEquals(5, env.lookup("b"));
    assertEquals(7, env.lookup("c"));
  }

  @Test
  public void testFailInclude() throws Exception {
    events.setFailFast(false);
    BuildFileAST buildFileAST = parseBuildFile("include(\"//nonexistent\")");
    assertThat(buildFileAST.getStatements()).hasSize(1);
    events.assertContainsEvent("Include of '//nonexistent' failed");
  }


  private class EmptyPackageLocator implements CachingPackageLocator {
    @Override
    public Path getBuildFileForPackage(String packageName) {
      return null;
    }
  }

  private CachingPackageLocator emptyLocator = new EmptyPackageLocator();

  @Test
  public void testFailInclude2() throws Exception {
    events.setFailFast(false);
    Path buildFile = scratch.file("/foo/bar/BUILD",
        "include(\"//nonexistent:foo\")\n");
    BuildFileAST buildFileAST = BuildFileAST.parseBuildFile(buildFile, events.reporter(),
                                                            emptyLocator, false);
    assertThat(buildFileAST.getStatements()).hasSize(1);
    events.assertContainsEvent("Package 'nonexistent' not found");
  }

  @Test
  public void testInvalidInclude() throws Exception {
    events.setFailFast(false);
    BuildFileAST buildFileAST = parseBuildFile("include(2)");
    assertThat(buildFileAST.getStatements()).isEmpty();
    events.assertContainsEvent("syntax error at '2'");
  }

  @Test
  public void testRecursiveInclude() throws Exception {
    events.setFailFast(false);
    Path buildFile = scratch.file("/foo/bar/BUILD",
        "include(\"//foo/bar:BUILD\")\n");

    BuildFileAST.parseBuildFile(buildFile, events.reporter(), locator, false);
    events.assertContainsEvent("Recursive inclusion");
  }

  @Test
  public void testParseErrorInclude() throws Exception {
    events.setFailFast(false);

    scratch.file("/foo/bar/file",
        "a = 2 + % 3\n"); // parse error

    parseBuildFile("include(\"//foo/bar:file\")");

    // Check the location is properly reported
    Event event = events.collector().iterator().next();
    assertEquals("/foo/bar/file:1:9", event.getLocation().print());
    assertEquals("syntax error at '%'", event.getMessage());
  }

  @Test
  public void testNonExistentIncludeReported() throws Exception {
    events.setFailFast(false);
    BuildFileAST buildFileAST = parseBuildFile("include('//foo:bar')");
    assertThat(buildFileAST.getStatements()).hasSize(1);
  }
}
