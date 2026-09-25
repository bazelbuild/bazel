// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.repository.RepoRecordedInput.WithValue.parse;
import static com.google.devtools.build.lib.rules.repository.RepoRecordedInput.WithValue.splitIntoBatches;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValueWithContentsProxy;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValueWithDigest;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Test class for {@link RepoRecordedInput}. */
@RunWith(JUnit4.class)
public class RepoRecordedInputTest extends BuildViewTestCase {
  private static void assertMarkerFileEscaping(String testCase) {
    String escaped = RepoRecordedInput.escape(testCase);
    assertThat(RepoRecordedInput.unescape(escaped)).isEqualTo(testCase);
  }

  @Test
  public void testMarkerFileEscaping() {
    assertMarkerFileEscaping(null);
    assertMarkerFileEscaping("\\0");
    assertMarkerFileEscaping("a\\0");
    assertMarkerFileEscaping("a b");
    assertMarkerFileEscaping("a b c");
    assertMarkerFileEscaping("a \\b");
    assertMarkerFileEscaping("a \\nb");
    assertMarkerFileEscaping("a \\\\nb");
    assertMarkerFileEscaping("a \\\nb");
    assertMarkerFileEscaping("a \nb");
  }

  @Test
  public void testFileValueToMarkerValue() throws Exception {
    RootedPath path =
        RootedPath.toRootedPath(Root.fromPath(rootDirectory), scratch.file("foo", "bar"));

    // Digest should be returned if the FileValue has it.
    FileValue fv = new RegularFileStateValueWithDigest(3, new byte[] {1, 2, 3, 4});
    assertThat(RepoRecordedInput.File.fileValueToMarkerValue(path, fv)).isEqualTo("01020304");

    // Digest should also be returned if the FileStateValue doesn't have it.
    FileStatus status = Mockito.mock(FileStatus.class);
    when(status.getLastChangeTime()).thenReturn(100L);
    when(status.getNodeId()).thenReturn(200L);
    fv = new RegularFileStateValueWithContentsProxy(3, FileContentsProxy.create(status));
    String expectedDigest = BaseEncoding.base16().lowerCase().encode(path.asPath().getDigest());
    assertThat(RepoRecordedInput.File.fileValueToMarkerValue(path, fv)).isEqualTo(expectedDigest);
  }

  @Test
  public void testFileValueToMarkerValue_usesDigestCache() throws Exception {
    var getDigestCounter = new AtomicInteger();
    var statCounter = new AtomicInteger();
    var tracingFileSystem =
        new InMemoryFileSystem(DigestHashFunction.SHA256) {
          @Override
          public byte[] getDigest(PathFragment path) throws IOException {
            getDigestCounter.incrementAndGet();
            return super.getDigest(path);
          }

          @Override
          public FileStatus stat(PathFragment path, boolean followSymlinks) throws IOException {
            statCounter.incrementAndGet();
            return super.stat(path, followSymlinks);
          }
        };
    var file = tracingFileSystem.getPath("/file.txt");
    FileSystemUtils.writeContentAsLatin1(file, "some contents");
    var path = RootedPath.toRootedPath(Root.absoluteRoot(tracingFileSystem), file);
    var fv = FileStateValue.create(path, SyscallCache.NO_CACHE, /* tsgm= */ null);
    assertThat(fv.getDigest()).isNull();

    DigestUtils.configureCache(/* maximumSize= */ 100);
    try {
      String expectedDigest = BaseEncoding.base16().lowerCase().encode(file.getDigest());
      getDigestCounter.set(0);
      statCounter.set(0);

      assertThat(RepoRecordedInput.File.fileValueToMarkerValue(path, fv))
          .isEqualTo(expectedDigest);
      assertThat(RepoRecordedInput.File.fileValueToMarkerValue(path, fv))
          .isEqualTo(expectedDigest);
      assertThat(getDigestCounter.get()).isEqualTo(1);
      // The cache key is derived from the FileValue's contents proxy.
      assertThat(statCounter.get()).isEqualTo(0);
    } finally {
      DigestUtils.configureCache(/* maximumSize= */ 0);
    }
  }

  @Test
  public void testSplitIntoBatches() {
    assertThat(splitIntoBatches(ImmutableList.of())).isEmpty();
    assertThat(
            splitIntoBatches(
                ImmutableList.of(
                    parse("FILE:@@//foo:bar abc").orElseThrow(),
                    parse("FILE:@@//:baz cba").orElseThrow(),
                    parse("FILE:@@foo//:baz bac").orElseThrow(),
                    parse("ENV:KEY value").orElseThrow())))
        .containsExactly(
            ImmutableList.of(
                parse("FILE:@@//foo:bar abc").orElseThrow(),
                parse("FILE:@@//:baz cba").orElseThrow()),
            ImmutableList.of(
                parse("FILE:@@foo//:baz bac").orElseThrow(), parse("ENV:KEY value").orElseThrow()));
  }
}
