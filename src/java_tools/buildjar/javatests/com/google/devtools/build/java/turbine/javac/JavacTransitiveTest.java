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

package com.google.devtools.build.java.turbine.javac;

import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.toList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.turbine.deps.AbstractTransitiveTest;
import com.google.turbine.options.TurbineOptions;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class JavacTransitiveTest extends AbstractTransitiveTest {

  private static final ImmutableList<Path> BOOTCLASSPATH =
      ImmutableList.of(Paths.get(System.getProperty("java.home")).resolve("lib/rt.jar"));

  @Override
  protected Path runTurbine(ImmutableList<Path> sources, ImmutableList<Path> classpath)
      throws IOException {
    Path out = temporaryFolder.newFolder().toPath().resolve("out.jar");
    boolean ok =
        JavacTurbine.compile(
                TurbineOptions.builder()
                    .addSources(sources.stream().map(Path::toString).collect(toList()))
                    .addClassPathEntries(classpath.stream().map(Path::toString).collect(toList()))
                    .addBootClassPathEntries(Iterables.transform(BOOTCLASSPATH, Path::toString))
                    .setOutput(out.toString())
                    .build())
            .ok();
    assertThat(ok).isTrue();
    return out;
  }
}
