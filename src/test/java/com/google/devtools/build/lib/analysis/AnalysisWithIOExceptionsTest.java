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

import static org.junit.Assert.fail;

import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link AnalysisTestCase} with custom filesystem that can throw on stat if desired. */
@RunWith(JUnit4.class)
public class AnalysisWithIOExceptionsTest extends AnalysisTestCase {
  private static final Function<Path, String> NULL_FUNCTION = (path) -> null;

  private Function<Path, String> crashMessage = NULL_FUNCTION;

  @Override
  protected FileSystem createFileSystem() {
    return new InMemoryFileSystem(BlazeClock.instance()) {
      @Override
      public FileStatus stat(Path path, boolean followSymlinks) throws IOException {
        String crash = crashMessage.apply(path);
        if (crash != null) {
          throw new IOException(crash);
        }
        return super.stat(path, followSymlinks);
      }
    };
  }

  @Test
  public void testGlobIOException() throws Exception {
    scratch.file("b/BUILD", "sh_library(name = 'b', deps= ['//a:a'])");
    scratch.file("a/BUILD", "sh_library(name = 'a', srcs = glob(['a.sh']))");
    crashMessage = path -> path.toString().contains("a.sh") ? "bork" : null;
    reporter.removeHandler(failFastHandler);
    try {
      update("//b:b");
      fail("Expected failure");
    } catch (ViewCreationFailedException expected) {
    }
  }
}
