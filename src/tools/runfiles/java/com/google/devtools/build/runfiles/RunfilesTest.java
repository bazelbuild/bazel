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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.io.File;
import java.io.IOException;
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

  private void assertRlocationArg(Runfiles runfiles, String path, @Nullable String error)
      throws Exception {
    try {
      runfiles.rlocation(path);
      fail();
    } catch (IllegalArgumentException e) {
      if (error != null) {
        assertThat(e).hasMessageThat().contains(error);
      }
    }
  }

  @Test
  public void testRlocationArgumentValidation() throws Exception {
    Runfiles r = Runfiles.create(ImmutableMap.of("RUNFILES_DIR", "whatever"));
    assertRlocationArg(r, null, null);
    assertRlocationArg(r, "", null);
    assertRlocationArg(r, "foo/..", "contains uplevel");
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
    Runfiles r =
        Runfiles.create(
            ImmutableMap.of(
                "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                "RUNFILES_DIR", "runfiles/dir",
                "JAVA_RUNFILES", "ignored when RUNFILES_DIR has a value",
                "TEST_SRCDIR", "should always be ignored"));
    assertThat(r.rlocation("a/b")).isEqualTo("runfiles/dir/a/b");
    assertThat(r.rlocation("foo")).isEqualTo("runfiles/dir/foo");

    r =
        Runfiles.create(
            ImmutableMap.of(
                "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
                "RUNFILES_DIR", "",
                "JAVA_RUNFILES", "runfiles/dir",
                "TEST_SRCDIR", "should always be ignored"));
    assertThat(r.rlocation("a/b")).isEqualTo("runfiles/dir/a/b");
    assertThat(r.rlocation("foo")).isEqualTo("runfiles/dir/foo");
  }

  @Test
  public void testIgnoresTestSrcdirWhenJavaRunfilesIsUndefinedAndJustFails() throws Exception {
    Runfiles.create(
        ImmutableMap.of(
            "RUNFILES_DIR", "non-empty",
            "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
            "TEST_SRCDIR", "should always be ignored"));

    Runfiles.create(
        ImmutableMap.of(
            "JAVA_RUNFILES", "non-empty",
            "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
            "TEST_SRCDIR", "should always be ignored"));

    try {
      // The method must ignore TEST_SRCDIR, for the scenario when Bazel runs a test which itself
      // runs Bazel to build and run java_binary. The java_binary should not pick up the test's
      // TEST_SRCDIR.
      Runfiles.create(
          ImmutableMap.of(
              "RUNFILES_DIR", "",
              "JAVA_RUNFILES", "",
              "RUNFILES_MANIFEST_FILE", "ignored when RUNFILES_MANIFEST_ONLY is not set to 1",
              "TEST_SRCDIR", "should always be ignored"));
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().contains("$RUNFILES_DIR and $JAVA_RUNFILES");
    }
  }

  @Test
  public void testFailsToCreateManifestBasedBecauseManifestDoesNotExist() throws Exception {
    try {
      Runfiles.create(
          ImmutableMap.of(
              "RUNFILES_MANIFEST_ONLY", "1",
              "RUNFILES_MANIFEST_FILE", "non-existing path"));
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().contains("non-existing path");
    }
  }
}
