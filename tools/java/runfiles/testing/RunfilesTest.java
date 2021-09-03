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

package com.google.devtools.build.runfiles;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Runfiles}. */
@RunWith(JUnit4.class)
public final class RunfilesTest {

  private static boolean isWindows() {
    return File.separatorChar == '\\';
  }

  private void assertRlocationArg(Runfiles runfiles, String path, @Nullable String error) {
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> runfiles.rlocation(path));
    if (error != null) {
        assertThat(e).hasMessageThat().contains(error);
    }
  }

  @Test
  public void testRlocationArgumentValidation() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Runfiles r = Runfiles.create(ImmutableMap.of("RUNFILES_DIR", dir.toString()));
    assertRlocationArg(r, null, null);
    assertRlocationArg(r, "", null);
    assertRlocationArg(r, "../foo", "is not normalized");
    assertRlocationArg(r, "foo/..", "is not normalized");
    assertRlocationArg(r, "foo/../bar", "is not normalized");
    assertRlocationArg(r, "./foo", "is not normalized");
    assertRlocationArg(r, "foo/.", "is not normalized");
    assertRlocationArg(r, "foo/./bar", "is not normalized");
    assertRlocationArg(r, "//foobar", "is not normalized");
    assertRlocationArg(r, "foo//", "is not normalized");
    assertRlocationArg(r, "foo//bar", "is not normalized");
    assertRlocationArg(r, "\\foo", "path is absolute without a drive letter");
  }

  @Test
  public void testCreatesManifestBasedRunfiles() throws Exception {
    try (MockFile mf = new MockFile(ImmutableList.of("a/b c/d"))) {
      Runfiles r =
          Runfiles.create(
              ImmutableMap.of(
                  "RUNFILES_MANIFEST_ONLY", "1",
                  "RUNFILES_MANIFEST_FILE", mf.path.toString(),
                  "RUNFILES_DIR", "ignored when RUNFILES_MANIFEST_ONLY=1",
                  "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                  "TEST_SRCDIR", "should always be ignored"));
      assertThat(r.rlocation("a/b")).isEqualTo("c/d");
      assertThat(r.rlocation("foo")).isNull();

      if (isWindows()) {
        assertThat(r.rlocation("c:/foo")).isEqualTo("c:/foo");
        assertThat(r.rlocation("c:\\foo")).isEqualTo("c:\\foo");
      } else {
        assertThat(r.rlocation("/foo")).isEqualTo("/foo");
      }
    }
  }

  @Test
  public void testCreatesDirectoryBasedRunfiles() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Runfiles r =
        Runfiles.create(
            ImmutableMap.of(
                "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                "RUNFILES_DIR", dir.toString(),
                "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                "TEST_SRCDIR", "should always be ignored"));
    assertThat(r.rlocation("a/b")).endsWith("/a/b");
    assertThat(r.rlocation("foo")).endsWith("/foo");

    r =
        Runfiles.create(
            ImmutableMap.of(
                "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                "RUNFILES_DIR", "",
                "JAVA_RUNFILES", dir.toString(),
                "TEST_SRCDIR", "should always be ignored"));
    assertThat(r.rlocation("a/b")).endsWith("/a/b");
    assertThat(r.rlocation("foo")).endsWith("/foo");
  }

  @Test
  public void testIgnoresTestSrcdirWhenJavaRunfilesIsUndefinedAndJustFails() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Runfiles.create(
        ImmutableMap.of(
            "RUNFILES_DIR", dir.toString(),
            "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
            "TEST_SRCDIR", "should always be ignored"));

    Runfiles.create(
        ImmutableMap.of(
            "JAVA_RUNFILES", dir.toString(),
            "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
            "TEST_SRCDIR", "should always be ignored"));

    IOException e =
        assertThrows(
            IOException.class,
            () ->
                Runfiles.create(
                    ImmutableMap.of(
                        "RUNFILES_DIR", "",
                        "JAVA_RUNFILES", "",
                        "RUNFILES_MANIFEST_FILE",
                            "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                        "TEST_SRCDIR", "should always be ignored")));
    assertThat(e).hasMessageThat().contains("$RUNFILES_DIR and $JAVA_RUNFILES");
  }

  @Test
  public void testFailsToCreateManifestBasedBecauseManifestDoesNotExist() {
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                Runfiles.create(
                    ImmutableMap.of(
                        "RUNFILES_MANIFEST_ONLY", "1",
                        "RUNFILES_MANIFEST_FILE", "non-existing path")));
    assertThat(e).hasMessageThat().contains("non-existing path");
  }

  @Test
  public void testManifestBasedEnvVars() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Path mf = dir.resolve("MANIFEST");
    Files.write(mf, Collections.emptyList(), StandardCharsets.UTF_8);
    Map<String, String> envvars =
        Runfiles.create(
                ImmutableMap.of(
                    "RUNFILES_MANIFEST_ONLY", "1",
                    "RUNFILES_MANIFEST_FILE", mf.toString(),
                    "RUNFILES_DIR", "ignored when RUNFILES_MANIFEST_ONLY=1",
                    "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                    "TEST_SRCDIR", "should always be ignored"))
            .getEnvVars();
    assertThat(envvars.keySet())
        .containsExactly(
            "RUNFILES_MANIFEST_ONLY", "RUNFILES_MANIFEST_FILE", "RUNFILES_DIR", "JAVA_RUNFILES");
    assertThat(envvars.get("RUNFILES_MANIFEST_ONLY")).isEqualTo("1");
    assertThat(envvars.get("RUNFILES_MANIFEST_FILE")).isEqualTo(mf.toString());
    assertThat(envvars.get("RUNFILES_DIR")).isEqualTo(dir.toString());
    assertThat(envvars.get("JAVA_RUNFILES")).isEqualTo(dir.toString());

    Path rfDir = dir.resolve("foo.runfiles");
    Files.createDirectories(rfDir);
    mf = dir.resolve("foo.runfiles_manifest");
    Files.write(mf, Collections.emptyList(), StandardCharsets.UTF_8);
    envvars =
        Runfiles.create(
                ImmutableMap.of(
                    "RUNFILES_MANIFEST_ONLY", "1",
                    "RUNFILES_MANIFEST_FILE", mf.toString(),
                    "RUNFILES_DIR", "ignored when RUNFILES_MANIFEST_ONLY=1",
                    "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                    "TEST_SRCDIR", "should always be ignored"))
            .getEnvVars();
    assertThat(envvars.get("RUNFILES_MANIFEST_ONLY")).isEqualTo("1");
    assertThat(envvars.get("RUNFILES_MANIFEST_FILE")).isEqualTo(mf.toString());
    assertThat(envvars.get("RUNFILES_DIR")).isEqualTo(rfDir.toString());
    assertThat(envvars.get("JAVA_RUNFILES")).isEqualTo(rfDir.toString());
  }

  @Test
  public void testDirectoryBasedEnvVars() throws Exception {
    Path dir =
        Files.createTempDirectory(
            FileSystems.getDefault().getPath(System.getenv("TEST_TMPDIR")), null);

    Map<String, String> envvars =
        Runfiles.create(
                ImmutableMap.of(
                    "RUNFILES_MANIFEST_FILE",
                    "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                    "RUNFILES_DIR",
                    dir.toString(),
                    "JAVA_RUNFILES",
                    "ignored when RUNFILES_DIR has a value",
                    "TEST_SRCDIR",
                    "should always be ignored"))
            .getEnvVars();
    assertThat(envvars.keySet()).containsExactly("RUNFILES_DIR", "JAVA_RUNFILES");
    assertThat(envvars.get("RUNFILES_DIR")).isEqualTo(dir.toString());
    assertThat(envvars.get("JAVA_RUNFILES")).isEqualTo(dir.toString());
  }

  @Test
  public void testDirectoryBasedRlocation() {
    // The DirectoryBased implementation simply joins the runfiles directory and the runfile's path
    // on a "/". DirectoryBased does not perform any normalization, nor does it check that the path
    // exists.
    File dir = new File(System.getenv("TEST_TMPDIR"), "mock/runfiles");
    assertThat(dir.mkdirs()).isTrue();
    Runfiles r = Runfiles.createDirectoryBasedForTesting(dir.toString());
    // Escaping for "\": once for string and once for regex.
    assertThat(r.rlocation("arg")).matches(".*[/\\\\]mock[/\\\\]runfiles[/\\\\]arg");
  }

  @Test
  public void testManifestBasedRlocation() throws Exception {
    try (MockFile mf =
        new MockFile(
            ImmutableList.of(
                "Foo/runfile1 C:/Actual Path\\runfile1",
                "Foo/Bar/runfile2 D:\\the path\\run file 2.txt"))) {
      Runfiles r = Runfiles.createManifestBasedForTesting(mf.path.toString());
      assertThat(r.rlocation("Foo/runfile1")).isEqualTo("C:/Actual Path\\runfile1");
      assertThat(r.rlocation("Foo/Bar/runfile2")).isEqualTo("D:\\the path\\run file 2.txt");
      assertThat(r.rlocation("unknown")).isNull();
    }
  }

  @Test
  public void testDirectoryBasedCtorArgumentValidation() {
    assertThrows(
        IllegalArgumentException.class, () -> Runfiles.createDirectoryBasedForTesting(null));

    assertThrows(IllegalArgumentException.class, () -> Runfiles.createDirectoryBasedForTesting(""));

    assertThrows(
        IllegalArgumentException.class,
        () -> Runfiles.createDirectoryBasedForTesting("non-existent directory is bad"));

    Runfiles.createDirectoryBasedForTesting(System.getenv("TEST_TMPDIR"));
  }

  @Test
  public void testManifestBasedCtorArgumentValidation() throws Exception {
    assertThrows(
        IllegalArgumentException.class, () -> Runfiles.createManifestBasedForTesting(null));

    assertThrows(IllegalArgumentException.class, () -> Runfiles.createManifestBasedForTesting(""));

    try (MockFile mf = new MockFile(ImmutableList.of("a b"))) {
      Runfiles.createManifestBasedForTesting(mf.path.toString());
    }
  }
}
