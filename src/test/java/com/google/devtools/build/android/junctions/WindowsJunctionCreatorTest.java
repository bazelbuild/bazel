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
package com.google.devtools.build.android.junctions;

import static com.google.common.truth.Truth.assertThat;

import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for WindowsJunctionCreator. */
@RunWith(JUnit4.class)
public class WindowsJunctionCreatorTest {
  private Path tmproot = null;

  @Before
  public void acquireTmpRoot() {
    String tmpEnv = System.getenv("TEST_TMPDIR");
    assertThat(tmpEnv).isNotNull();
    tmproot = FileSystems.getDefault().getPath(tmpEnv);
    // Cast Path to Object to disambiguate which assertThat-overload to use.
    assertThat((Object) tmproot).isNotNull();
  }

  @Test
  public void testNullInput() throws Exception {
    try (JunctionCreator jc = new WindowsJunctionCreator(tmproot.resolve("foo"))) {
      // Cast Path to Object to disambiguate which assertThat-overload to use.
      assertThat((Object) jc.create(null)).isNull();
    }
  }

  @Test
  public void testFileInput() throws Exception {
    Path dir = tmproot.resolve("foo"); // [tmproot]/foo/
    Path file = dir.resolve("bar"); // [tmproot]/foo/bar
    Path juncroot = tmproot.resolve("junc"); // [tmproot]/junc/
    Path junc = juncroot.resolve("0"); // [tmproot]/junc/0 -> [tmproot]/foo/bar
    Path fileViaJunc = junc.resolve("bar"); // [tmproot]/junc/0/bar
    try {
      Files.createDirectories(dir);
      Files.createDirectories(juncroot);

      // Create a scratch file and assert its existence.
      Files.write(file, "hello".getBytes());
      assertThat(file.toFile().exists()).isTrue();

      // Assert that the junction doesn't exist yet.
      assertThat(junc.toFile().exists()).isFalse();
      assertThat(fileViaJunc.toFile().exists()).isFalse();

      try (JunctionCreator jc = new WindowsJunctionCreator(juncroot)) {
        // Assert creation of a junction for a file (more precisely, for its parent directory).
        // Cast Path to Object to disambiguate which assertThat-overload to use.
        assertThat((Object) jc.create(file)).isEqualTo(fileViaJunc);
        // Assert that the junction now exists.
        assertThat(junc.toFile().exists()).isTrue();
        // Assert that `file` should still exist.
        assertThat(file.toFile().exists()).isTrue();
        // Assert that the junction indeed points to `dir`, by asserting the existence of `file`
        // through the junction.
        assertThat(fileViaJunc.toFile().exists()).isTrue();
      }
      // Assert that WindowsJunctionCreator.close cleaned up the junction and root directory, but
      // not the files in the directory that the junction pointed to.
      assertThat(junc.toFile().exists()).isFalse();
      assertThat(juncroot.toFile().exists()).isFalse();
      // Assert that the `file` and `directory` still exist. Deleting the junction should not have
      // affected them.
      assertThat(file.toFile().exists()).isTrue();
      assertThat(dir.toFile().exists()).isTrue();
    } finally {
      file.toFile().delete();
      dir.toFile().delete();
    }
  }

  @Test
  public void testJunctionCaching() throws Exception {
    Path dir1 = tmproot.resolve("foo"); // [tmproot]/foo/
    Path dir2 = dir1.resolve("foo"); // [tmproot]/foo/foo/
    Path file1 = dir1.resolve("bar"); // [tmproot]/foo/bar
    Path file2 = dir1.resolve("baz"); // [tmproot]/foo/baz
    Path file3 = dir2.resolve("bar"); // [tmproot]/foo/foo/bar
    Path juncroot = tmproot.resolve("junc"); // [tmproot]/junc/
    Path junc0 = juncroot.resolve("0"); // [tmproot]/junc/0 -> [tmproot]/foo/
    Path junc1 = juncroot.resolve("1"); // [tmproot]/junc/1 -> [tmproot]/foo/foo/
    Path file1junc = junc0.resolve("bar"); // [tmproot]/junc/0/bar
    Path file2junc = junc0.resolve("baz"); // [tmproot]/junc/0/baz
    Path file3junc = junc1.resolve("bar"); // [tmproot]/junc/1/bar

    try {
      Files.createDirectories(dir2);
      Files.createDirectories(juncroot);

      // Create scratch files and assert their existence.
      Files.write(file1, "i am file1".getBytes());
      Files.write(file2, "i am file2".getBytes());
      Files.write(file3, "i am file3".getBytes());
      assertThat(file1.toFile().exists()).isTrue();
      assertThat(file2.toFile().exists()).isTrue();
      assertThat(file3.toFile().exists()).isTrue();

      // Assert that the junctions don't exist yet.
      assertThat(junc0.toFile().exists()).isFalse();
      assertThat(junc1.toFile().exists()).isFalse();
      assertThat(file1junc.toFile().exists()).isFalse();
      assertThat(file2junc.toFile().exists()).isFalse();
      assertThat(file3junc.toFile().exists()).isFalse();

      try (JunctionCreator jc = new WindowsJunctionCreator(juncroot)) {
        Path dir1juncActual = jc.create(dir1);
        Path file1juncActual = jc.create(file1);
        Path dir2juncActual = jc.create(dir2);
        Path file3juncActual = jc.create(file3);
        Path file2juncActual = jc.create(file2);

        // Assert that the junctions now exists.
        assertThat(dir1juncActual.toFile().exists()).isTrue();
        assertThat(dir2juncActual.toFile().exists()).isTrue();
        assertThat(file1juncActual.toFile().exists()).isTrue();
        assertThat(file2juncActual.toFile().exists()).isTrue();
        assertThat(file3juncActual.toFile().exists()).isTrue();

        // Assert that the junctions were chached.
        // Cast Path to Object to disambiguate which assertThat-overload to use.
        assertThat((Object) dir1juncActual).isEqualTo(junc0);
        assertThat((Object) dir2juncActual).isEqualTo(junc1);
        assertThat((Object) file1juncActual).isEqualTo(file1junc);
        assertThat((Object) file2juncActual).isEqualTo(file2junc);
        assertThat((Object) file3juncActual).isEqualTo(file3junc);
        assertThat((Object) file1juncActual.getParent()).isEqualTo(junc0);
        assertThat((Object) file2juncActual.getParent()).isEqualTo(junc0);
        assertThat((Object) file3juncActual.getParent()).isEqualTo(junc1);

        // Assert that the directory junctions indeed point where they should, by asserting the
        // existence of `file1`, `file2`, and `file3` through them.
        assertThat(dir1juncActual.resolve("bar").toFile().exists()).isTrue();
        assertThat(dir1juncActual.resolve("baz").toFile().exists()).isTrue();
        assertThat(dir2juncActual.resolve("bar").toFile().exists()).isTrue();

        // Assert that the file junctions indeed point where they should, by asserting the contents
        // we can read from them.
        assertThat(Files.readAllBytes(file1junc)).isEqualTo("i am file1".getBytes());
        assertThat(Files.readAllBytes(file2junc)).isEqualTo("i am file2".getBytes());
        assertThat(Files.readAllBytes(file3junc)).isEqualTo("i am file3".getBytes());
      }

      // Assert that WindowsJunctionCreator.close cleaned up the junction and root directory, but
      // not the files in the directory that the junction pointed to.
      assertThat(junc0.toFile().exists()).isFalse();
      assertThat(junc1.toFile().exists()).isFalse();
      assertThat(file1junc.toFile().exists()).isFalse();
      assertThat(file2junc.toFile().exists()).isFalse();
      assertThat(file3junc.toFile().exists()).isFalse();
      assertThat(juncroot.toFile().exists()).isFalse();

      // Assert that the original files (and consequently the directories) still exist. Deleting the
      // junction should not have affected them.
      assertThat(file1.toFile().exists()).isTrue();
      assertThat(file2.toFile().exists()).isTrue();
      assertThat(file3.toFile().exists()).isTrue();
    } finally {
      file3.toFile().delete();
      file2.toFile().delete();
      file1.toFile().delete();
      dir2.toFile().delete();
      dir1.toFile().delete();
    }
  }
}
