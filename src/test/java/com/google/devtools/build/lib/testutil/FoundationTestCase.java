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

import com.google.common.io.Files;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import junit.framework.TestCase;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Set;

/**
 * This is a specialization of {@link TestCase} that's useful for implementing tests of the
 * "foundation" library.
 */
public abstract class FoundationTestCase extends TestCase {

  protected Path rootDirectory;

  protected Path outputBase;

  protected Path actionOutputBase;

  // May be overridden by subclasses:
  protected Reporter reporter;
  protected EventCollector eventCollector;

  protected Scratch scratch;


  // Individual tests can opt-out of this handler if they expect an error, by
  // calling reporter.removeHandler(failFastHandler).
  protected static final EventHandler failFastHandler = new EventHandler() {
      @Override
      public void handle(Event event) {
        if (EventKind.ERRORS.contains(event.getKind())) {
          fail(event.toString());
        }
      }
    };

  protected static final EventHandler printHandler = new EventHandler() {
      @Override
      public void handle(Event event) {
        System.out.println(event);
      }
    };

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    scratch = new Scratch(createFileSystem());
    outputBase = scratch.dir("/usr/local/google/_blaze_jrluser/FAKEMD5/");
    rootDirectory = scratch.dir("/" + TestConstants.TEST_WORKSPACE_DIRECTORY);
    scratchFile(rootDirectory.getRelative("WORKSPACE").getPathString(),
        "bind(",
        "  name = 'objc_proto_lib',",
        "  actual = '//objcproto:ProtocolBuffers_lib',",
        ")",
        "bind(",
        "  name = 'objc_proto_cpp_lib',",
        "  actual = '//objcproto:ProtocolBuffersCPP_lib',",
        ")");
    copySkylarkFilesIfExist();
    actionOutputBase = scratch.dir("/usr/local/google/_blaze_jrluser/FAKEMD5/action_out/");
    eventCollector = new EventCollector(EventKind.ERRORS_AND_WARNINGS);
    reporter = new Reporter(eventCollector);
    reporter.addHandler(failFastHandler);
  }

  /*
   * Creates the file system; override to inject FS behavior.
   */
  protected FileSystem createFileSystem() {
     return new InMemoryFileSystem(BlazeClock.instance());
  }

  private void copySkylarkFilesIfExist() throws IOException {
    scratchFile(rootDirectory.getRelative("devtools/blaze/rules/BUILD").getPathString());
    scratchFile(rootDirectory.getRelative("rules/BUILD").getPathString());
    copySkylarkFilesIfExist("devtools/blaze/rules/staging", "devtools/blaze/rules/staging");
    copySkylarkFilesIfExist("third_party/bazel/tools/build_rules", "rules");
  }

  private void copySkylarkFilesIfExist(String from, String to) throws IOException {
    File rulesDir = new File(from);
    if (rulesDir.exists() && rulesDir.isDirectory()) {
      for (String fileName : rulesDir.list()) {
        File file = new File(from + "/" + fileName);
        if (file.isFile() && fileName.endsWith(".bzl")) {
          String context = loadFile(file);
          Path path = rootDirectory.getRelative(to + "/" + fileName);
          if (path.exists()) {
            overwriteScratchFile(path.getPathString(), context);
          } else {
            scratchFile(path.getPathString(), context);
          }
        }
      }
    }
  }

  @Override
  protected void tearDown() throws Exception {
    Thread.interrupted(); // Clear any interrupt pending against this thread,
                          // so that we don't cause later tests to fail.

    super.tearDown();
  }

  /**
   * A scratch filesystem that is completely in-memory. Since this file system
   * is "cached" in a private (but *not* static) field in the test class,
   * each testFoo method in junit sees a fresh filesystem.
   */
  protected FileSystem scratchFS() {
    return scratch.getFileSystem();
  }

  /**
   * Create a scratch file in the scratch filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  protected Path scratchFile(String pathName, String... lines)
      throws IOException {
    return scratch.file(pathName, lines);
  }

  /**
   * Like {@code scratchFile}, but the file is first deleted if it already
   * exists.
   */
  protected Path overwriteScratchFile(String pathName, String... lines) throws IOException {
    return scratch.overwriteFile(pathName, lines);
  }

  /**
   * Create a scratch file in the given filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  protected Path scratchFile(FileSystem fs, String pathName, String... lines)
      throws IOException {
    return scratch.file(fs, pathName, lines);
  }

  /**
   * Create a scratch file in the given filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  protected Path scratchFile(FileSystem fs, String pathName, byte[] content)
      throws IOException {
    return scratch.file(fs, pathName, content);
  }

  // Mix-in assertions:

  protected void assertNoEvents() {
    JunitTestUtils.assertNoEvents(eventCollector);
  }

  protected Event assertContainsEvent(String expectedMessage) {
    return JunitTestUtils.assertContainsEvent(eventCollector,
                                              expectedMessage);
  }

  protected Event assertContainsEvent(String expectedMessage, Set<EventKind> kinds) {
    return JunitTestUtils.assertContainsEvent(eventCollector,
                                              expectedMessage,
                                              kinds);
  }

  protected void assertContainsEventWithFrequency(String expectedMessage,
      int expectedFrequency) {
    JunitTestUtils.assertContainsEventWithFrequency(eventCollector, expectedMessage,
        expectedFrequency);
  }

  protected void assertDoesNotContainEvent(String expectedMessage) {
    JunitTestUtils.assertDoesNotContainEvent(eventCollector,
                                             expectedMessage);
  }

  protected Event assertContainsEventWithWordsInQuotes(String... words) {
    return JunitTestUtils.assertContainsEventWithWordsInQuotes(
        eventCollector, words);
  }

  protected void assertContainsEventsInOrder(String... expectedMessages) {
    JunitTestUtils.assertContainsEventsInOrder(eventCollector, expectedMessages);
  }

  @SuppressWarnings({"unchecked", "varargs"})
  protected static <T> void assertContainsSublist(List<T> arguments,
                                                  T... expectedSublist) {
    JunitTestUtils.assertContainsSublist(arguments, expectedSublist);
  }

  @SuppressWarnings({"unchecked", "varargs"})
  protected static <T> void assertDoesNotContainSublist(List<T> arguments,
                                                        T... expectedSublist) {
    JunitTestUtils.assertDoesNotContainSublist(arguments, expectedSublist);
  }

  protected static <T> void assertContainsSubset(Iterable<T> arguments,
                                                 Iterable<T> expectedSubset) {
    JunitTestUtils.assertContainsSubset(arguments, expectedSubset);
  }

  protected String loadFile(File file) throws IOException {
    return Files.toString(file, Charset.defaultCharset());
  }
}
