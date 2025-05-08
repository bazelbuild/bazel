// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.generatedprojecttest;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.generatedprojecttest.util.BuildFileContentsGenerator;
import com.google.devtools.build.lib.generatedprojecttest.util.TestProjectBuilder;
import com.google.devtools.build.lib.testutil.BuildRuleBuilder;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;
import net.starlark.java.syntax.SyntaxError;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@code BuildFileContentsGenerator}.
 */
@RunWith(JUnit4.class)
public final class BuildFileContentsGeneratorTest {

  /**
   * The generator being tested.
   */
  private final BuildFileContentsGenerator generator = new BuildFileContentsGenerator();

  @Test
  public void testSetDefaultPackageVisibility() throws IllegalStateException {
    generator.setDefaultPackageVisibility("//visibility:private");
    assertThat(generator.getContents())
        .startsWith("package(default_visibility = ['//visibility:private'])");
  }

  @Test
  public void defaultPackageVisibilityIsAddedToStartOfBuildFile() throws IllegalStateException {
    generator.addRule(new BuildRuleBuilder("cc_library", generator.uniqueRuleName()));
    generator.setDefaultPackageVisibility("//visibility:private");
    assertThat(generator.getContents())
        .startsWith("package(default_visibility = ['//visibility:private'])");
  }

  @Test
  public void defaultPackageVisibilityDefaultsToPublic() throws IllegalStateException {
    generator.addRule(new BuildRuleBuilder("cc_library", generator.uniqueRuleName()));
    assertThat(generator.getContents())
        .startsWith("package(default_visibility = ['//visibility:public'])");
  }

  @Test
  public void settingDefaultPackageVisibilityTwiceCausesException() throws IllegalStateException {
    generator.setDefaultPackageVisibility("//visibility:private");
    assertThrows(
        IllegalStateException.class,
        () -> generator.setDefaultPackageVisibility("//visibility:private"));
  }

  @Test
  public void testContentsSyntax() throws IOException {
    // TODO(blaze-team): (2012) write various simple generator examples to test the generated syntax
    TestProjectBuilder builder = new TestProjectBuilder("tmp");
    BuildFileContentsGenerator generator = new BuildFileContentsGenerator();
    builder.createFileInDir("/a", "BUILD", generator);
    Scratch scratch = builder.getScratch();
    Path path = scratch.resolve("/tmp/a/BUILD");

    byte[] bytes = FileSystemUtils.readWithKnownFileSize(path, path.getFileSize());
    ParserInput input = ParserInput.fromLatin1(bytes, path.toString());
    StarlarkFile file = StarlarkFile.parse(input);
    for (SyntaxError error : file.errors()) {
      System.err.println(error);
    }
    assertThat(file.ok()).isTrue();
  }
}
