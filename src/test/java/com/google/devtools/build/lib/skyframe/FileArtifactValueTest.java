// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.FileArtifactValue.createShareable;
import static org.junit.Assert.fail;

import com.google.common.io.BaseEncoding;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FileArtifactValueTest {
  private final ManualClock clock = new ManualClock();
  private final FileSystem fs = new InMemoryFileSystem(clock);

  private Path scratchFile(String name, long mtime, String content) throws IOException {
    Path path = fs.getPath(name);
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(path, content);
    path.setLastModifiedTime(mtime);
    return path;
  }

  private Path scratchDir(String name, long mtime) throws IOException {
    Path path = fs.getPath(name);
    path.createDirectoryAndParents();
    path.setLastModifiedTime(mtime);
    return path;
  }

  private static byte[] toBytes(String hex) {
    return BaseEncoding.base16().upperCase().decode(hex);
  }

  @Test
  public void testEqualsAndHashCode() throws Exception {
    // Each "equality group" is checked for equality within itself (including hashCode equality)
    // and inequality with members of other equality groups.
    new EqualsTester()
        .addEqualityGroup(
            FileArtifactValue.createNormalFile(
                toBytes("00112233445566778899AABBCCDDEEFF"),
                /*proxy=*/ null,
                1L,
                /*isShareable=*/ true),
            FileArtifactValue.createNormalFile(
                toBytes("00112233445566778899AABBCCDDEEFF"),
                /*proxy=*/ null,
                1L,
                /*isShareable=*/ true))
        .addEqualityGroup(
            FileArtifactValue.createNormalFile(
                toBytes("00112233445566778899AABBCCDDEEFF"),
                /*proxy=*/ null,
                2L,
                /*isShareable=*/ true))
        .addEqualityGroup(FileArtifactValue.createDirectory(1))
        .addEqualityGroup(
            FileArtifactValue.createNormalFile(
                toBytes("FFFFFF00000000000000000000000000"),
                /*proxy=*/ null,
                1L,
                /*isShareable=*/ true))
        .addEqualityGroup(
            FileArtifactValue.createNormalFile(
                toBytes("FFFFFF00000000000000000000000000"),
                /*proxy=*/ null,
                1L,
                /*isShareable=*/ false))
        .addEqualityGroup(
            FileArtifactValue.createDirectory(2), FileArtifactValue.createDirectory(2))
        .addEqualityGroup(FileArtifactValue.OMITTED_FILE_MARKER)
        .addEqualityGroup(FileArtifactValue.MISSING_FILE_MARKER)
        .addEqualityGroup(FileArtifactValue.DEFAULT_MIDDLEMAN)
        .addEqualityGroup("a string")
        .testEquals();
  }

  @Test
  public void testEquality() throws Exception {
    Path path1 = scratchFile("/dir/artifact1", 0L, "content");
    Path path2 = scratchFile("/dir/artifact2", 0L, "content");
    Path digestPath = scratchFile("/dir/diffDigest", 0L, "1234567");
    Path mtimePath = scratchFile("/dir/diffMtime", 1L, "content");

    Path empty1 = scratchFile("/dir/empty1", 0L, "");
    Path empty2 = scratchFile("/dir/empty2", 1L, "");
    Path empty3 = scratchFile("/dir/empty3", 1L, "");

    Path dir1 = scratchDir("/dir1", 0L);
    Path dir2 = scratchDir("/dir2", 1L);
    Path dir3 = scratchDir("/dir3", 1L);

    new EqualsTester()
        // We check for ctime and inode equality for paths.
        .addEqualityGroup(createShareable(path1))
        .addEqualityGroup(createShareable(path2))
        .addEqualityGroup(createShareable(mtimePath))
        .addEqualityGroup(createShareable(digestPath))
        .addEqualityGroup(createShareable(empty1))
        .addEqualityGroup(createShareable(empty2))
        .addEqualityGroup(createShareable(empty3))
        // We check for mtime equality for directories.
        .addEqualityGroup(createShareable(dir1))
        .addEqualityGroup(createShareable(dir2), createShareable(dir3))
        .testEquals();
  }

  @Test
  public void testCtimeInEquality() throws Exception {
    Path path = scratchFile("/dir/artifact1", 0L, "content");
    FileArtifactValue before = createShareable(path);
    clock.advanceMillis(1);
    path.chmod(0777);
    FileArtifactValue after = createShareable(path);
    assertThat(before).isNotEqualTo(after);
  }

  @Test
  public void testNoMtimeIfNonemptyFile() throws Exception {
    Path path = scratchFile("/root/non-empty", 1L, "abc");
    FileArtifactValue value = createShareable(path);
    assertThat(value.getDigest()).isEqualTo(path.getDigest());
    assertThat(value.getSize()).isEqualTo(3L);
    try {
      value.getModifiedTime();
      fail("mtime for non-empty file should not be stored.");
    } catch (UnsupportedOperationException e) {
      // Expected.
    }
  }

  @Test
  public void testDirectory() throws Exception {
    Path path = scratchDir("/dir", /*mtime=*/ 1L);
    FileArtifactValue value = createShareable(path);
    assertThat(value.getDigest()).isNull();
    assertThat(value.getModifiedTime()).isEqualTo(1L);
  }

  // Empty files are the same as normal files -- mtime is not stored.
  @Test
  public void testEmptyFile() throws Exception {
    Path path = scratchFile("/root/empty", 1L, "");
    path.setLastModifiedTime(1L);
    FileArtifactValue value = createShareable(path);
    assertThat(value.getDigest()).isEqualTo(path.getDigest());
    assertThat(value.getSize()).isEqualTo(0L);
    try {
      value.getModifiedTime();
      fail("mtime for non-empty file should not be stored.");
    } catch (UnsupportedOperationException e) {
      // Expected.
    }
  }

  @Test
  public void testIOException() throws Exception {
    final IOException exception = new IOException("beep");
    FileSystem fs =
        new InMemoryFileSystem() {
          @Override
          public byte[] getDigest(Path path) throws IOException {
            throw exception;
          }

          @Override
          protected byte[] getFastDigest(Path path) throws IOException {
            throw exception;
          }
        };
    Path path = fs.getPath("/some/path");
    path.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(path, "content");
    try {
      createShareable(path);
      fail();
    } catch (IOException e) {
      assertThat(e).isSameAs(exception);
    }
  }

  @Test
  public void testUptodateCheck() throws Exception {
    Path path = scratchFile("/dir/artifact1", 0L, "content");
    FileArtifactValue value = createShareable(path);
    clock.advanceMillis(1);
    assertThat(value.wasModifiedSinceDigest(path)).isFalse();
    clock.advanceMillis(1);
    assertThat(value.wasModifiedSinceDigest(path)).isFalse();
    clock.advanceMillis(1);
    path.setLastModifiedTime(123); // Changing mtime implicitly updates ctime.
    assertThat(value.wasModifiedSinceDigest(path)).isTrue();
    clock.advanceMillis(1);
    assertThat(value.wasModifiedSinceDigest(path)).isTrue();
  }

  @Test
  public void testUptodateCheckDeleteFile() throws Exception {
    Path path = scratchFile("/dir/artifact1", 0L, "content");
    FileArtifactValue value = createShareable(path);
    assertThat(value.wasModifiedSinceDigest(path)).isFalse();
    path.delete();
    assertThat(value.wasModifiedSinceDigest(path)).isTrue();
  }

  @Test
  public void testUptodateCheckDirectory() throws Exception {
    // For now, we don't attempt to detect changes to directories.
    Path path = scratchDir("/dir", 0L);
    FileArtifactValue value = createShareable(path);
    assertThat(value.wasModifiedSinceDigest(path)).isFalse();
    path.delete();
    clock.advanceMillis(1);
    assertThat(value.wasModifiedSinceDigest(path)).isFalse();
  }

  @Test
  public void testUptodateChangeFileToDirectory() throws Exception {
    // For now, we don't attempt to detect changes to directories.
    Path path = scratchFile("/dir/file", 0L, "");
    FileArtifactValue value = createShareable(path);
    assertThat(value.wasModifiedSinceDigest(path)).isFalse();
    // If we only check ctime, then we need to change the clock here, or we get a ctime match on the
    // stat.
    path.delete();
    path.createDirectoryAndParents();
    clock.advanceMillis(1);
    assertThat(value.wasModifiedSinceDigest(path)).isTrue();
  }

  @Test
  public void testIsMarkerValue_marker() {
    assertThat(FileArtifactValue.DEFAULT_MIDDLEMAN.isMarkerValue()).isTrue();
    assertThat(FileArtifactValue.MISSING_FILE_MARKER.isMarkerValue()).isTrue();
    assertThat(FileArtifactValue.OMITTED_FILE_MARKER.isMarkerValue()).isTrue();
  }

  @Test
  public void testIsMarkerValue_notMarker() throws Exception {
    FileArtifactValue value = createShareable(scratchFile("/dir/artifact1", 0L, "content"));
    assertThat(value.isMarkerValue()).isFalse();
  }
}
