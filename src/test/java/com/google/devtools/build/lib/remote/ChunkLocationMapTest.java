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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.primitives.Bytes;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import java.util.List;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ChunkLocationMap}. */
@RunWith(JUnit4.class)
public final class ChunkLocationMapTest {
  private static final DigestUtil DIGEST_UTIL =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  private static final byte[] FIRST = "first chunk".getBytes(UTF_8);
  private static final byte[] SECOND = "second chunk".getBytes(UTF_8);
  private static final Digest FIRST_DIGEST = DIGEST_UTIL.compute(FIRST);
  private static final Digest SECOND_DIGEST = DIGEST_UTIL.compute(SECOND);

  private final ChunkLocationMap map = new ChunkLocationMap();

  private Path root;

  @Before
  public void setUp() throws Exception {
    root = TestUtils.createUniqueTmpDir(null);
  }

  @After
  public void tearDown() throws Exception {
    root.deleteTree();
  }

  @Test
  public void read_unknownChunk_returnsNull() {
    assertThat(read(FIRST_DIGEST)).isNull();
  }

  @Test
  public void read_returnsChunksAtRecordedOffsets() throws Exception {
    Path file = writeFile("out", Bytes.concat(FIRST, SECOND));

    map.addFile(file, List.of(FIRST_DIGEST, SECOND_DIGEST));

    assertThat(read(FIRST_DIGEST)).isEqualTo(FIRST);
    assertThat(read(SECOND_DIGEST)).isEqualTo(SECOND);
  }

  @Test
  public void read_rewrittenFile_returnsNullAndForgetsLocation() throws Exception {
    Path file = writeFile("out", FIRST);
    map.addFile(file, List.of(FIRST_DIGEST));

    writeFile("out", SECOND);
    assertThat(read(FIRST_DIGEST)).isNull();

    // The location is gone for good, even once the file happens to hold the chunk again.
    writeFile("out", FIRST);
    assertThat(read(FIRST_DIGEST)).isNull();
  }

  @Test
  public void read_deletedFile_returnsNullAndForgetsLocation() throws Exception {
    Path file = writeFile("out", FIRST);
    map.addFile(file, List.of(FIRST_DIGEST));

    file.delete();
    assertThat(read(FIRST_DIGEST)).isNull();

    writeFile("out", FIRST);
    assertThat(read(FIRST_DIGEST)).isNull();
  }

  @Test
  public void read_locationIsDestination_returnsNullButKeepsLocation() throws Exception {
    Path file = writeFile("out", FIRST);
    map.addFile(file, List.of(FIRST_DIGEST));

    // A download writing to this very file may not read from it, but another one still may.
    assertThat(map.read(FIRST_DIGEST, file, DIGEST_UTIL)).isNull();
    assertThat(read(FIRST_DIGEST)).isEqualTo(FIRST);
  }

  @Test
  public void addFile_mostRecentFileWins() throws Exception {
    Path stale = writeFile("stale", FIRST);
    Path fresh = writeFile("fresh", Bytes.concat(SECOND, FIRST));

    map.addFile(stale, List.of(FIRST_DIGEST));
    map.addFile(fresh, List.of(SECOND_DIGEST, FIRST_DIGEST));
    stale.delete();

    assertThat(read(FIRST_DIGEST)).isEqualTo(FIRST);
  }

  @Test
  public void clear_forgetsLocations() throws Exception {
    Path file = writeFile("out", FIRST);
    map.addFile(file, List.of(FIRST_DIGEST));

    map.clear();

    assertThat(read(FIRST_DIGEST)).isNull();
  }

  private byte[] read(Digest digest) {
    return map.read(digest, /* destination= */ null, DIGEST_UTIL);
  }

  private Path writeFile(String name, byte[] contents) throws Exception {
    Path path = root.getRelative("execroot/" + name);
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContent(path, contents);
    return path;
  }
}
