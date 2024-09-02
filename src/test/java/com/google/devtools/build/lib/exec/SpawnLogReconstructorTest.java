package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;

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
    assertThat(matchDefault("bazel-out/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result(null, "pkg/file.txt"));
    assertThat(matchDefault("bazel-out/k8-fastbuild/bin/pkg/external/file.txt"))
        .isEqualTo(new Result(null, "pkg/external/file.txt"));
    assertThat(matchDefault("bazel-out/k8-fastbuild/bin/external/some_repo/pkg/file.txt"))
        .isEqualTo(new Result("some_repo", "pkg/file.txt"));
    assertThat(matchDefault("bazel-out/k8-fastbuild/bin/external/some-repo+/pkg/file.txt"))
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
    assertThat(matchSibling("bazel-out/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result(null, "pkg/file.txt"));
    assertThat(matchSibling("bazel-out/k8-fastbuild/bin/pkg/external/file.txt"))
        .isEqualTo(new Result(null, "pkg/external/file.txt"));
    assertThat(matchSibling("bazel-out/k8-fastbuild/bin/external/pkg/file.txt"))
        .isEqualTo(new Result(null, "external/pkg/file.txt"));
    assertThat(matchSibling("bazel-out/some_repo/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result("some_repo", "pkg/file.txt"));
    assertThat(matchSibling("bazel-out/some-repo+/k8-fastbuild/bin/pkg/file.txt"))
        .isEqualTo(new Result("some-repo+", "pkg/file.txt"));
    assertThat(matchSibling("bazel-out/k8-fastbuild/coverage-metadata/bin/other/pkg/gen.txt"))
        .isEqualTo(new Result(null, "bin/other/pkg/gen.txt"));
    assertThat(
            matchSibling(
                "bazel-out/some_repo/k8-fastbuild/coverage-metadata/bin/other/pkg/gen.txt"))
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
