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

package com.google.devtools.build.java.turbine;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.java.bazel.JavacBootclasspath;
import com.google.turbine.options.TurbineOptions;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class TurbineTest {

  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();

  @Test
  public void jadepCommand() throws Exception {

    Path source = tempFolder.newFile("Hello.java").toPath();
    Files.write(source, "import p.Lib;\nclass Hello extends Lib {}".getBytes(UTF_8));

    Path output = tempFolder.newFile("output.jar").toPath();
    Path outputDeps = tempFolder.newFile("output.jdeps").toPath();
    Path tempdir = tempFolder.newFolder("tempdir").toPath();

    TurbineOptions.Builder optionsBuilder =
        TurbineOptions.builder()
            .setJavacFallback(false)
            .setOutput(output.toString())
            .setTempDir(tempdir.toString())
            .addBootClassPathEntries(
                JavacBootclasspath.asPaths()
                    .stream()
                    .map(Path::toString)
                    .collect(toImmutableList()))
            .addSources(ImmutableList.of(source.toString()))
            .setOutputDeps(outputDeps.toString())
            .addAllJavacOpts(Arrays.asList("-source", "8", "-target", "8"))
            .setTargetLabel("//test");

    StringWriter errOutput = new StringWriter();
    int result = -1;
    try (PrintWriter writer = new PrintWriter(errOutput, true)) {
      result =
          new Turbine(
                  "An exception has occurred in turbine.",
                  "",
                  (type, target) -> String.format("jadep -classnames=%s %s", type, target))
              .compile(optionsBuilder.build(), writer);
    }
    assertThat(errOutput.toString())
        .contains(
            "Hello.java:1: error: symbol not found p.Lib\n"
                + "import p.Lib;\n"
                + "       ^\n"
                + "\n"
                + "\033[35m\033[1m** Command to add missing dependencies:\033[0m\n"
                + "\n"
                + "jadep -classnames=p.Lib //test");
    assertThat(result).isEqualTo(1);
  }
}
