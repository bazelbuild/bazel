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

package com.google.devtools.build.lib.blackbox.tests;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.blackbox.framework.BuilderRunner;
import com.google.devtools.build.lib.blackbox.framework.PathUtils;
import com.google.devtools.build.lib.blackbox.framework.ProcessResult;
import com.google.devtools.build.lib.blackbox.junit.AbstractBlackBoxTest;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Set;
import org.junit.Test;

public class DirtinessBlackBoxTest extends AbstractBlackBoxTest {
  private final static String CC_FILE_TEMPLATE = String.join("\n",
      "#include <stdio.h>",
      "int main() {",
      "  printf(\"%s\\n\");",
      "}");

  @Test
  public void testSymlinkedCcFileChangeInBuild() throws Exception {
    Path extDir = context().getTmpDir().resolve("some_path");
    extDir.toFile().mkdirs();
    Path target = extDir.resolve("external.cc");
    PathUtils.writeFile(target, String.format(CC_FILE_TEMPLATE, "one"));
    assertThat(target.toFile().exists()).isTrue();

    Path aDir = context().getWorkDir().resolve("a");
    aDir.toFile().mkdirs();
    assertThat(aDir.toFile().isDirectory()).isTrue();
    Path foo = aDir.resolve("foo.cc");
    Files.createSymbolicLink(foo, target);
    context().write("a/BUILD", "cc_binary(name = 'foo', srcs = ['foo.cc'])");

    BuilderRunner bazel = context().bazel();
    bazel.build("//a:foo");
    ProcessResult runResultTwo = context().runBuiltBinary(bazel, "a/foo", 1000);
    assertThat(runResultTwo.outString()).isEqualTo("one");

    PathUtils.writeFile(target, String.format(CC_FILE_TEMPLATE, "two"));
    bazel.build("//a:foo");
    runResultTwo = context().runBuiltBinary(bazel, "a/foo", 1000);
    assertThat(runResultTwo.outString()).isEqualTo("two");
  }

  @Test
  public void testSymlinkedBuildFileChangeInQuery() throws Exception {
    Path extDir = context().getTmpDir().resolve("some_other_path");
    extDir.toFile().mkdirs();
    Path targetBuild = extDir.resolve("BUILD");
    PathUtils.writeFile(targetBuild, "cc_library(name = 'foo', srcs = [])");

    Path aDir = context().getWorkDir().resolve("a");
    aDir.toFile().mkdirs();
    assertThat(aDir.toFile().isDirectory()).isTrue();

    Path buildPath = aDir.resolve("BUILD");
    Files.createSymbolicLink(buildPath, targetBuild);

    BuilderRunner bazel = context().bazel();
    ProcessResult queryResult = bazel.query("a:all");
    String[] lines = queryResult.outString().split("\n");
    assertThat(lines.length).isEqualTo(1);
    assertThat(lines[0]).isEqualTo("//a:foo");

    PathUtils.append(targetBuild, "cc_library(name = 'bar', srcs = [])");
    ProcessResult secondQueryResult = bazel.query("a:all");

    Set<String> newLines = Sets.newHashSet(Arrays.asList(secondQueryResult.outString().split("\n")));
    assertThat(newLines.size()).isEqualTo(2);
    assertThat(newLines).contains("//a:foo");
    assertThat(newLines).contains("//a:bar");
  }
}
