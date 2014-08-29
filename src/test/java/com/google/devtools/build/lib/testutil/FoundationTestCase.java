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
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Set;

/**
 * This is a specialization of {@link ChattyAssertsTestCase} that's useful for
 * implementing tests of the "foundation" library.
 */
public abstract class FoundationTestCase extends ChattyAssertsTestCase {

  protected Path rootDirectory;

  protected Path outputBase;

  protected Path actionOutputBase;

  // May be overridden by subclasses:
  protected Reporter reporter;
  protected EventCollector eventCollector;

  private FileSystem fileSystem = null;

  // Individual tests can opt-out of this handler if they expect an error, by
  // calling reporter.removeHandler(failFastHandler).
  protected static final EventHandler failFastHandler = new EventHandler() {
      @Override
      public Set<EventKind> getEventMask() {
        return EventKind.ERRORS;
      }
      @Override
      public void handle(Event event) {
        fail(event.toString());
      }
      @Override
      public boolean showOutput(String tag) {
        return true;
      }
    };

  protected static final EventHandler printHandler = new EventHandler() {
      @Override
      public Set<EventKind> getEventMask() {
        return EventKind.ALL_EVENTS;
      }
      @Override
      public void handle(Event event) {
        System.out.println(event);
      }
      @Override
      public boolean showOutput(String tag) {
        return true;
      }
    };

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    scratchDir("/home/jrluser/src-foo/google3");
    outputBase = scratchDir("/usr/local/google/_blaze_jrluser/FAKEMD5/");
    rootDirectory = scratchDir("/google3");
    actionOutputBase = scratchDir("/usr/local/google/_blaze_jrluser/FAKEMD5/action_out/");
    eventCollector = new EventCollector(EventKind.ERRORS_AND_WARNINGS);
    reporter = new Reporter(eventCollector);
    reporter.addHandler(failFastHandler);
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
    if (fileSystem == null) {
      fileSystem = createFileSystem();
    }
    return fileSystem;
  }

  /**
   * Creates a new in-memory filesystem.
   */
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(BlazeClock.instance());
  }

  /**
   * Create a scratch file in the scratch filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  protected Path scratchFile(String pathName, String... lines)
      throws IOException {
    Path newFile = scratchFile(scratchFS(), pathName, lines);
    newFile.setLastModifiedTime(-1L);
    return newFile;
  }

  /**
   * Like {@code scratchFile}, but the file is first deleted if it already
   * exists.
   */
  protected Path overwriteScratchFile(String pathName, String... lines) throws IOException {
    Path oldFile = scratchFS().getPath(pathName);
    long newMTime = oldFile.exists() ? oldFile.getLastModifiedTime() + 1 : -1;
    oldFile.delete();
    Path newFile = scratchFile(scratchFS(), pathName, lines);
    newFile.setLastModifiedTime(newMTime);
    return newFile;
  }

  /**
   * Deletes the specified scratch file, using the same specification as {@link Path#delete}.
   */
  protected boolean deleteScratchFile(String pathName) throws IOException {
    Path file = scratchFS().getPath(pathName);
    return file.delete();
  }

  /**
   * Create a scratch file in the given filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  protected Path scratchFile(FileSystem fs, String pathName, String... lines)
      throws IOException {
    Path file = newScratchFile(fs, pathName);
    FileSystemUtils.writeContentAsLatin1(file, linesAsString(lines));
    return file;
  }

  /**
   * Create a scratch file in the given filesystem, with the given pathName,
   * consisting of a set of lines. The method returns a Path instance for the
   * scratch file.
   */
  protected Path scratchFile(FileSystem fs, String pathName, byte[] content)
      throws IOException {
    Path file = newScratchFile(fs, pathName);
    FileSystemUtils.writeContent(file, content);
    return file;
  }

  private Path newScratchFile(FileSystem fs, String pathName) throws IOException {
    Path file = fs.getPath(pathName);
    Path parentDir = file.getParentDirectory();
    if (!parentDir.exists()) {
      FileSystemUtils.createDirectoryAndParents(parentDir);
    }
    if (file.exists()) {
      throw new IOException("Could not create scratch file (file exists) "
          + pathName);
    }
    return file;
  }

  /**
   * Create a directory in the scratch filesystem, with the given path name.
   */
  protected Path scratchDir(String pathName) throws IOException {
    Path dir = scratchFS().getPath(pathName);
    if (!dir.exists()) {
      FileSystemUtils.createDirectoryAndParents(dir);
    }
    if (!dir.isDirectory()) {
      throw new IOException("Exists, but is not a directory: " + pathName);
    }
    return dir;
  }

  /**
   * Converts the lines into a String with linebreaks. Useful for creating
   * in-memory input for a file, for example.
   */
  private static String linesAsString(String... lines) {
    StringBuilder builder = new StringBuilder();
    for (String line : lines) {
      builder.append(line);
      builder.append('\n');
    }
    return builder.toString();
  }

  /**
   * If "expectedSuffix" is not a suffix of "actual", fails with an informative
   * assertion.
   */
  protected void assertEndsWith(String expectedSuffix, String actual) {
    if (!actual.endsWith(expectedSuffix)) {
      fail("\"" + actual + "\" does not end with "
           + "\"" + expectedSuffix + "\"");
    }
  }

  /**
   * If "expectedPrefix" is not a prefix of "actual", fails with an informative
   * assertion.
   */
  protected void assertStartsWith(String expectedPrefix, String actual) {
    if (!actual.startsWith(expectedPrefix)) {
      fail("\"" + actual + "\" does not start with "
           + "\"" + expectedPrefix + "\"");
    }
  }

  // Mix-in assertions:

  protected void assertNoEvents() {
    JunitTestUtils.assertNoEvents(eventCollector);
  }

  protected Event assertContainsEvent(String expectedMessage) {
    return JunitTestUtils.assertContainsEvent(eventCollector,
                                              expectedMessage);
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
