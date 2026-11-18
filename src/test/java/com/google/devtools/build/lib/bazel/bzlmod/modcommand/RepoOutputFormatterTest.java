// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod.modcommand;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModOptions.OutputFormat;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionValue.RepoOverride;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.StarlarkFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RepoOutputFormatter}. */
@RunWith(JUnit4.class)
public final class RepoOutputFormatterTest {

  @Test
  public void repoOverrideOutput_isValidStarlark() {
    StringWriter output = new StringWriter();
    RepoOutputFormatter formatter =
        new RepoOutputFormatter(
            new PrintWriter(output),
            new ByteArrayOutputStream(),
            OutputFormat.TEXT,
            /* externalDirectory= */ null);

    formatter.print(
        "@@bazel_tools",
        new RepoOverride(PathFragment.create("/embedded_tools\nnot-a-repository")));

    StarlarkFile file =
        StarlarkFile.parse(ParserInput.fromString(output.toString(), "show_repo output"));
    assertThat(file.ok()).isTrue();
    assertThat(output.toString()).contains("# Builtin or overridden repo located at:");
    assertThat(output.toString()).contains("/embedded_tools\\nnot-a-repository");
  }
}
