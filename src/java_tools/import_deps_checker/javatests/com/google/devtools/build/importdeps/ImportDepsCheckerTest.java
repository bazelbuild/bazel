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
package com.google.devtools.build.importdeps;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.view.proto.Deps.Dependencies;
import com.google.devtools.build.lib.view.proto.Deps.Dependency;
import com.google.devtools.build.lib.view.proto.Deps.Dependency.Kind;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link ImportDepsChecker} */
@RunWith(JUnit4.class)
public class ImportDepsCheckerTest extends AbstractClassCacheTest {

  @Test
  public void testJdepsProtos() throws IOException {
    testJdepsProto(
        ImmutableList.of(bootclasspath),
        ImmutableList.of(libraryJar),
        ImmutableList.of(clientJar),
        /*expectedCheckResult=*/ false,
        ImmutableList.of(libraryJar));
    testJdepsProto(
        ImmutableList.of(clientJar), // fake non-empty bootclasspath.
        ImmutableList.of(libraryJar),
        ImmutableList.of(clientJar),
        false,
        ImmutableList.of(libraryJar));
    testJdepsProto(
        ImmutableList.of(bootclasspath),
        ImmutableList.of(libraryJar, libraryAnnotationsJar),
        ImmutableList.of(clientJar),
        false,
        ImmutableList.of(libraryJar, libraryAnnotationsJar));
    testJdepsProto(
        ImmutableList.of(bootclasspath),
        ImmutableList.of(libraryAnnotationsJar),
        ImmutableList.of(libraryJar),
        false,
        ImmutableList.of());
    testJdepsProto(
        ImmutableList.of(bootclasspath),
        ImmutableList.of(
            libraryJar, libraryAnnotationsJar, libraryExceptionJar, libraryInterfaceJar),
        ImmutableList.of(clientJar),
        true,
        ImmutableList.of(
            libraryJar, libraryAnnotationsJar, libraryExceptionJar, libraryInterfaceJar));
  }

  private static final String DUMMY_RULE_LABEL = "empty";

  private static void testJdepsProto(
      ImmutableList<Path> bootclasspath,
      ImmutableList<Path> regularClasspath,
      ImmutableList<Path> inputJars,
      boolean expectedCheckResult,
      ImmutableList<Path> expectedJdeps)
      throws IOException {
    try (ImportDepsChecker checker =
        new ImportDepsChecker(
            bootclasspath, regularClasspath, regularClasspath, inputJars, false)) {
      assertThat(checker.check()).isEqualTo(expectedCheckResult);
      Dependencies deps = checker.emitJdepsProto(DUMMY_RULE_LABEL);
      assertThat(deps.getDependencyList())
          .containsExactlyElementsIn(
              expectedJdeps
                  .stream()
                  .map(
                      path ->
                          Dependency.newBuilder()
                              .setKind(Kind.EXPLICIT)
                              .setPath(path.toString())
                              .build())
                  .collect(Collectors.toList()));
      assertPathsAreRelative(deps);
      assertThat(checker.check()).isEqualTo(expectedCheckResult);
      Dependencies deps2 = checker.emitJdepsProto(DUMMY_RULE_LABEL);
      assertThat(deps).isEqualTo(deps2);

      System.err.println(deps2);

      Dependencies depsFromMain =
          getJdepsProtoWithMainEntry(bootclasspath, regularClasspath, inputJars);
      assertThat(deps).isEqualTo(depsFromMain);
    }
  }

  private static Dependencies getJdepsProtoWithMainEntry(
      ImmutableList<Path> bootclasspath,
      ImmutableList<Path> regularClasspath,
      ImmutableList<Path> inputJars)
      throws IOException {
    ArrayList<String> args = new ArrayList<>();
    bootclasspath.forEach(
        s -> {
          args.add("--bootclasspath_entry");
          args.add(s.toString());
        });
    regularClasspath.forEach(
        s -> {
          args.add("--classpath_entry");
          args.add(s.toString());
        });
    inputJars.forEach(
        s -> {
          args.add("--input");
          args.add(s.toString());
        });
    args.add("--jdeps_output");
    Path jdepsFile = Files.createTempFile("temp_importdeps", ".jdeps");
    args.add(jdepsFile.toString());
    args.add("--rule_label=" + DUMMY_RULE_LABEL);

    args.add("--output");
    args.add(Files.createTempFile("temp_output", ".txt").toString());
    args.add("--checking_mode=silence");
    Main.checkDeps(args.toArray(new String[0]));

    try (InputStream inputStream = Files.newInputStream(jdepsFile)) {
      return Dependencies.parseFrom(inputStream);
    }
  }

  private static void assertPathsAreRelative(Dependencies deps) {
    for (Dependency dep : deps.getDependencyList()) {
      assertThat(dep.getPath().startsWith("/")).isFalse();
    }
  }
}
