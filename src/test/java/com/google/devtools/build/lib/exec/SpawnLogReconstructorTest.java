// Copyright 2024 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.TestConstants.PRODUCT_NAME;

import java.util.regex.MatchResult;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SpawnLogReconstructorTest {

  @Test
  public void extractRunfilesPathDefault() {
    assertThat(matchDefault("file.txt")).isEqualTo(new Result(null, "file.txt"));
    assertThat(matchDefault("pkg/file.txt")).isEqualTo(new Result(null, "pkg/file.txt"));
    assertThat(matchDefault("pkg/external/file.txt"))
        .isEqualTo(new Result(null, "pkg/external/file.txt"));
    assertThat(matchDefault("external/some_repo/pkg/file.txt"))
        .isEqualTo(new Result("some_repo", "pkg/file.txt"));
    assertThat(matchDefault("external/some-repo+/pkg/file.txt"))
        .isEqualTo(new Result("some-repo+", "pkg/file.txt"));
    assertThat(matchDefault(PRODUCT_NAME + "-out/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result(null, "pkg/file.txt"));
    assertThat(matchDefault(PRODUCT_NAME + "-out/k8-fastbuild/bin/pkg/external/file.txt"))
        .isEqualTo(new Result(null, "pkg/external/file.txt"));
    assertThat(matchDefault(PRODUCT_NAME + "-out/k8-fastbuild/bin/external/some_repo/pkg/file.txt"))
        .isEqualTo(new Result("some_repo", "pkg/file.txt"));
    assertThat(
            matchDefault(PRODUCT_NAME + "-out/k8-fastbuild/bin/external/some-repo+/pkg/file.txt"))
        .isEqualTo(new Result("some-repo+", "pkg/file.txt"));
  }

  @Test
  public void extractRunfilesPathSibling() {
    assertThat(matchSibling("file.txt")).isEqualTo(new Result(null, "file.txt"));
    assertThat(matchSibling("pkg/file.txt")).isEqualTo(new Result(null, "pkg/file.txt"));
    assertThat(matchSibling("pkg/external/file.txt"))
        .isEqualTo(new Result(null, "pkg/external/file.txt"));
    assertThat(matchSibling("external/pkg/file.txt"))
        .isEqualTo(new Result(null, "external/pkg/file.txt"));
    assertThat(matchSibling("../some_repo/pkg/file.txt"))
        .isEqualTo(new Result("some_repo", "pkg/file.txt"));
    assertThat(matchSibling("../some-repo+/pkg/file.txt"))
        .isEqualTo(new Result("some-repo+", "pkg/file.txt"));
    assertThat(matchSibling(PRODUCT_NAME + "-out/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result(null, "pkg/file.txt"));
    assertThat(matchSibling(PRODUCT_NAME + "-out/k8-fastbuild/bin/pkg/external/file.txt"))
        .isEqualTo(new Result(null, "pkg/external/file.txt"));
    assertThat(matchSibling(PRODUCT_NAME + "-out/k8-fastbuild/bin/external/pkg/file.txt"))
        .isEqualTo(new Result(null, "external/pkg/file.txt"));
    assertThat(matchSibling(PRODUCT_NAME + "-out/some_repo/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result("some_repo", "pkg/file.txt"));
    assertThat(matchSibling(PRODUCT_NAME + "-out/some-repo+/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result("some-repo+", "pkg/file.txt"));
    assertThat(
            matchSibling(
                PRODUCT_NAME + "-out/k8-fastbuild/coverage-metadata/bin/other/pkg/gen.txt"))
        .isEqualTo(new Result(null, "bin/other/pkg/gen.txt"));
    assertThat(
            matchSibling(
                PRODUCT_NAME
                    + "-out/some_repo/k8-fastbuild/coverage-metadata/bin/other/pkg/gen.txt"))
        .isEqualTo(new Result("some_repo", "bin/other/pkg/gen.txt"));
  }

  private record Result(String repo, String path) {}

  private static Result matchDefault(String path) {
    MatchResult result =
        SpawnLogReconstructor.extractRunfilesPath(path, /* siblingRepositoryLayout= */ false);
    return new Result(result.group("repo"), result.group("path"));
  }

  private static Result matchSibling(String path) {
    MatchResult result =
        SpawnLogReconstructor.extractRunfilesPath(path, /* siblingRepositoryLayout= */ true);
    return new Result(result.group("repo"), result.group("path"));
  }
}
