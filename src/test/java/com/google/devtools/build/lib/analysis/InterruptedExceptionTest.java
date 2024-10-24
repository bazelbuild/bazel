// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests verifying appropriate propagation of {@link InterruptedException} during filesystem
 * operations.
 */
@RunWith(JUnit4.class)
public class InterruptedExceptionTest extends AnalysisTestCase {

  private final Thread mainThread = Thread.currentThread();

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(DigestHashFunction.SHA256) {
      @Override
      protected Collection<Dirent> readdir(PathFragment path, boolean followSymlinks)
          throws IOException {
        if (path.toString().contains("causes_interrupt")) {
          mainThread.interrupt();
        }
        return super.readdir(path, followSymlinks);
      }
    };
  }

  @Test
  public void testGlobInterruptedException() throws Exception {
    scratch.file("a/BUILD", "sh_library(name = 'a', srcs = glob(['**/*']))");
    scratch.file("a/b/foo.sh", "testfile");
    scratch.file("a/causes_interrupt/bar.sh", "testfile");
    reporter.removeHandler(failFastHandler);

    assertThrows(InterruptedException.class, () -> update("//a:a"));
  }

  @Test
  public void testStarlarkGlobInterruptedException() throws Exception {
    scratch.file(
        "a/gen.bzl",
        """
        def gen():
            native.filegroup(name = "a", srcs = native.glob(["**/*"]))
        """);
    scratch.file(
        "a/BUILD",
        """
        load("//a:gen.bzl", "gen")

        gen()
        """);

    scratch.file("a/b/foo.sh", "testfile");
    scratch.file("a/causes_interrupt/bar.sh", "testfile");
    reporter.removeHandler(failFastHandler);

    assertThrows(InterruptedException.class, () -> update("//a:a"));
  }
}
