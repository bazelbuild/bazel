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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;
import static org.mockito.Mockito.times;

import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;

/** Tests for {@link ByteStreamBuildEventArtifactUploader}. */
@RunWith(JUnit4.class)
public class ByteStreamBuildEventArtifactUploaderTest {
  private final FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);

  @Mock private ByteStreamUploader uploader;

  @Before
  public void before() {
    MockitoAnnotations.initMocks(this);
    Mockito.when(uploader.uploadBlobAsync(Mockito.any(Chunker.class)))
        .thenReturn(Futures.immediateFuture(null));
  }

  @SuppressWarnings("unchecked")
  @Test
  public void pathConversionWithServerAndInstanceName() throws Exception {
    ByteStreamBuildEventArtifactUploader artifactUploader =
        new ByteStreamBuildEventArtifactUploader(uploader, "serverName", "instanceName");

    Path file1 = fs.getPath("/foo");
    FileSystemUtils.writeContentAsLatin1(file1, "foo");
    Path file2 = fs.getPath("/bar");
    FileSystemUtils.writeContentAsLatin1(file2, "bar");

    Map<Path, LocalFile> files = new HashMap<>();
    files.put(file1, new LocalFile(file1, LocalFileType.OUTPUT));
    files.put(file2, new LocalFile(file2, LocalFileType.OUTPUT));
    PathConverter converter = artifactUploader.upload(files).get();

    Mockito.verify(uploader, times(2)).uploadBlobAsync(Mockito.any(Chunker.class));

    assertThat(converter.apply(file1))
        .isEqualTo(
            "bytestream://serverName/instanceName/blobs/"
                + "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae/3");
    assertThat(converter.apply(file2))
        .isEqualTo(
            "bytestream://serverName/instanceName/blobs/"
                + "fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9/3");

    try {
      converter.apply(fs.getPath("/foobar"));
      fail("exception expected");
    } catch (IllegalStateException e) {
      // Intentionally left empty.
    }
  }

  @Test
  public void pathConversionWithServerAndNoInstanceName() throws Exception {
    ByteStreamBuildEventArtifactUploader artifactUploader =
        new ByteStreamBuildEventArtifactUploader(uploader, "serverName", null);

    Path file1 = fs.getPath("/foo");
    FileSystemUtils.writeContentAsLatin1(file1, "foo");

    PathConverter converter =
        artifactUploader
            .upload(Collections.singletonMap(file1, new LocalFile(file1, LocalFileType.OUTPUT)))
            .get();

    assertThat(converter.apply(file1))
        .isEqualTo(
            "bytestream://serverName/blobs/"
                + "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae/3");
  }
}
