// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link AnalysisTestCase} with custom filesystem that can throw on stat if desired. */
@RunWith(JUnit4.class)
public class AnalysisWithIOExceptionsTest extends AnalysisTestCase {
  private Function<PathFragment, String> crashMessage = (path) -> null;

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(DigestHashFunction.SHA256) {
      @Override
      public FileStatus statIfFound(PathFragment path, boolean followSymlinks) throws IOException {
        String crash = crashMessage.apply(path);
        if (crash != null) {
          throw new IOException(crash);
        }
        return super.statIfFound(path, followSymlinks);
      }
    };
  }

  @Test
  public void testGlobIOException() throws Exception {
    scratch.file(
        "b/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'b', deps= ['//a:a'])");
    scratch.file(
        "a/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'a', srcs = glob(['a.sh']))");
    crashMessage = path -> path.toString().contains("a.sh") ? "bork" : null;
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//b:b"));
  }

  @Test
  public void testIncrementalGlobIOException() throws Exception {
    scratch.file(
        "b/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'b', deps= ['//a:a'])");
    scratch.file(
        "a/BUILD",
        """
        load('//test_defs:foo_library.bzl', 'foo_library')
        foo_library(name = 'a', srcs = glob(['a.sh']))
        foo_library(name = 'expensive', srcs = ['expensive.sh'])
        """);
    Path aShFile = scratch.file("a/a.sh");
    update("//b:b");
    skyframeExecutor.invalidateFilesUnderPathForTesting(
        reporter,
        ModifiedFileSet.builder().modify(aShFile.relativeTo(rootDirectory)).build(),
        Root.fromPath(rootDirectory));
    crashMessage = path -> path.toString().contains("a.sh") ? "bork" : null;
    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//b:b"));
  }

  @Test
  public void testWorkspaceError() throws IOException {
    scratch.file("a/BUILD");
    crashMessage = path -> path.toString().contains("MODULE.bazel") ? "bork" : null;
    reporter.removeHandler(failFastHandler);
    assertThrows(
        TargetParsingException.class,
        () -> update(new FlagBuilder().with(Flag.KEEP_GOING), "//a:a"));
  }

  @Test
  public void testGlobExceptionWithCrossingLabel() throws Exception {
    reporter.removeHandler(failFastHandler);
    Path buildPath =
        scratch.file(
            "foo/BUILD",
            """
            load('//test_defs:foo_library.bzl', 'foo_library')
            foo_library(name = 'foo', srcs = glob(['subdir/*.sh']))
            foo_library(name = 'crosses/directory', srcs = ['foo.sh'])
            """);
    scratch.file(
        "top/BUILD",
        "load('//test_defs:foo_library.bzl', 'foo_library')",
        "foo_library(name = 'top', deps = ['//foo:foo'], srcs = ['top.sh'])");
    Path errorPath = buildPath.getParentDirectory().getChild("subdir");
    crashMessage = path -> errorPath.asFragment().equals(path) ? "custom crash: bork" : null;
    assertThrows(ViewCreationFailedException.class, () -> update("//top:top"));
  }
}
