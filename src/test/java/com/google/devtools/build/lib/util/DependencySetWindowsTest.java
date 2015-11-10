// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Set;

@RunWith(JUnit4.class)
public class DependencySetWindowsTest {

  private Scratch scratch = new Scratch();

  private DependencySet newDependencySet() {
    return new DependencySet(scratch.resolve("/"));
  }

  @Test
  public void dotDParser_windowsPaths() throws Exception {
    Path dotd = scratch.file("/tmp/foo.d",
        "bazel-out/hello-lib/cpp/hello-lib.o: \\",
        " cpp/hello-lib.cc cpp/hello-lib.h c:\\mingw\\include\\stdio.h \\",
        " c:\\mingw\\include\\_mingw.h \\",
        " c:\\mingw\\lib\\gcc\\mingw32\\4.8.1\\include\\stdarg.h");

    Set<PathFragment> expected = Sets.newHashSet(
        new PathFragment("cpp/hello-lib.cc"),
        new PathFragment("cpp/hello-lib.h"),
        new PathFragment("C:/mingw/include/stdio.h"),
        new PathFragment("C:/mingw/include/_mingw.h"),
        new PathFragment("C:/mingw/lib/gcc/mingw32/4.8.1/include/stdarg.h"));

    assertThat(newDependencySet().read(dotd).getDependencies()).containsExactlyElementsIn(expected);
  }

  @Test
  public void dotDParser_windowsPathsWithSpaces() throws Exception {
    Path dotd = scratch.file("/tmp/foo.d",
        "bazel-out/hello-lib/cpp/hello-lib.o: \\",
        "C:\\Program\\ Files\\ (x86)\\LLVM\\stddef.h");
    assertThat(newDependencySet().read(dotd).getDependencies())
        .containsExactlyElementsIn(
            Sets.newHashSet(new PathFragment("C:/Program Files (x86)/LLVM/stddef.h")));
  }

  @Test
  public void dotDParser_mixedWindowsPaths() throws Exception {
    // This is (slightly simplified) actual output from clang. Yes, clang will happily mix
    // forward slashes and backslashes in a single path, not to mention using backslashes as
    // separators next to backslashes as escape characters.
    Path dotd = scratch.file("/tmp/foo.d",
        "bazel-out/hello-lib/cpp/hello-lib.o: \\",
        "cpp/hello-lib.cc cpp/hello-lib.h /mingw/include\\stdio.h \\",
        "/mingw/include\\_mingw.h \\",
        "C:\\Program\\ Files\\ (x86)\\LLVM\\bin\\..\\lib\\clang\\3.5.0\\include\\stddef.h \\",
        "C:\\Program\\ Files\\ (x86)\\LLVM\\bin\\..\\lib\\clang\\3.5.0\\include\\stdarg.h");

    Set<PathFragment> expected = Sets.newHashSet(
        new PathFragment("cpp/hello-lib.cc"),
        new PathFragment("cpp/hello-lib.h"),
        new PathFragment("/mingw/include/stdio.h"),
        new PathFragment("/mingw/include/_mingw.h"),
        new PathFragment("C:/Program Files (x86)/LLVM/lib/clang/3.5.0/include/stddef.h"),
        new PathFragment("C:/Program Files (x86)/LLVM/lib/clang/3.5.0/include/stdarg.h"));

    assertThat(newDependencySet().read(dotd).getDependencies()).containsExactlyElementsIn(expected);
  }
}
