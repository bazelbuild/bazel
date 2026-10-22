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
package com.google.devtools.build.lib.sandbox.backend;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.protobuf.ByteString;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileUriBlobs}. */
@RunWith(JUnit4.class)
public final class FileUriBlobsTest {
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  @Test
  public void encode_percentEncodesAndRoundTrips() {
    String path = "/a b/ü%/x~y-z_.txt";
    String internal = new String(path.getBytes(java.nio.charset.StandardCharsets.UTF_8),
        java.nio.charset.StandardCharsets.ISO_8859_1);

    ByteString uri = FileUriBlobs.encode(internal);

    assertThat(uri.toStringUtf8()).isEqualTo("file:///a%20b/%C3%BC%25/x~y-z_.txt");
    assertThat(FileUriBlobs.decode(uri)).isEqualTo(internal);
  }

  @Test
  public void encode_relativePath_fails() {
    assertThrows(IllegalArgumentException.class, () -> FileUriBlobs.encode("a/b"));
  }

  @Test
  public void decode_rejectsHostAndBadEncoding() {
    assertThrows(
        IllegalArgumentException.class,
        () -> FileUriBlobs.decode(ByteString.copyFromUtf8("file://host/a")));
    assertThrows(
        IllegalArgumentException.class,
        () -> FileUriBlobs.decode(ByteString.copyFromUtf8("file:///a%2")));
    assertThrows(
        IllegalArgumentException.class,
        () -> FileUriBlobs.decode(ByteString.copyFromUtf8("file:///a%zz")));
    assertThrows(
        IllegalArgumentException.class, () -> FileUriBlobs.decode(ByteString.copyFromUtf8("/a")));
  }

  @Test
  public void isFileUri_uriForDifferentContents() {
    Digest digest = digestUtil.computeAsUtf8("contents");
    ByteString uri = FileUriBlobs.encode("/some/file");

    assertThat(FileUriBlobs.isFileUri(uri, digest, digestUtil)).isTrue();
  }

  @Test
  public void isFileUri_notAUri() {
    ByteString data = ByteString.copyFromUtf8("contents");

    assertThat(FileUriBlobs.isFileUri(data, digestUtil.compute(data), digestUtil)).isFalse();
  }

  @Test
  public void isFileUri_realBlobThatStartsWithScheme() {
    ByteString data = ByteString.copyFromUtf8("file:///etc/passwd");

    // The blob that the digest describes is the data itself.
    assertThat(FileUriBlobs.isFileUri(data, digestUtil.compute(data), digestUtil)).isFalse();
  }

  @Test
  public void isFileUri_sameSizeDifferentHash() {
    ByteString data = ByteString.copyFromUtf8("file:///aaaa");
    ByteString other = ByteString.copyFromUtf8("file:///bbbb");

    assertThat(FileUriBlobs.isFileUri(data, digestUtil.compute(other), digestUtil)).isTrue();
  }
}
