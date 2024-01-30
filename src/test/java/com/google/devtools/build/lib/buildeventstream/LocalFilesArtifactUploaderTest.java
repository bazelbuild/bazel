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
package com.google.devtools.build.lib.buildeventstream;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.OutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LocalFilesArtifactUploader} */
@RunWith(JUnit4.class)
public class LocalFilesArtifactUploaderTest {
  private final FileSystem fileSystem = new InMemoryFileSystem(DigestHashFunction.SHA256);
  private final LocalFilesArtifactUploader artifactUploader = new LocalFilesArtifactUploader();

  @Test
  public void testFile() throws Exception {
    Path file = fileSystem.getPath("/test");
    // No need to create the file since LocalFileType.OUTPUT_FILE skips the filesystem check.
    ListenableFuture<PathConverter> future =
        artifactUploader.upload(
            ImmutableMap.of(
                file,
                new LocalFile(
                    file,
                    LocalFileType.OUTPUT_FILE,
                    /* artifact= */ null,
                    /* artifactMetadata= */ null)));
    PathConverter pathConverter = future.get();
    assertThat(pathConverter.apply(file)).isEqualTo("file:///test");
  }

  @Test
  public void testDirectory_notUploaded() throws Exception {
    Path file = fileSystem.getPath("/test");
    // No need to create the file since LocalFileType.OUTPUT_DIRECTORY skips the filesystem check.
    ListenableFuture<PathConverter> future =
        artifactUploader.upload(
            ImmutableMap.of(
                file,
                new LocalFile(
                    file,
                    LocalFileType.OUTPUT_DIRECTORY,
                    /* artifact= */ null,
                    /* artifactMetadata= */ null)));
    PathConverter pathConverter = future.get();
    assertThat(pathConverter.apply(file)).isNull();
  }

  @Test
  public void testSymlink_notUploaded() throws Exception {
    Path file = fileSystem.getPath("/test");
    // No need to create the file since LocalFileType.OUTPUT_FILE skips the filesystem check.
    ListenableFuture<PathConverter> future =
        artifactUploader.upload(
            ImmutableMap.of(
                file,
                new LocalFile(
                    file,
                    LocalFileType.OUTPUT_SYMLINK,
                    /* artifact= */ null,
                    /* artifactMetadata= */ null)));
    PathConverter pathConverter = future.get();
    assertThat(pathConverter.apply(file)).isNull();
  }

  @Test
  public void testUnknown_uploadedIfFile() throws Exception {
    Path file = fileSystem.getPath("/test");
    // Must create file since LocalFileType.OUTPUT triggers a filesystem check.
    try (OutputStream out = file.getOutputStream()) {
      out.write(new byte[0]);
    }
    ListenableFuture<PathConverter> future =
        artifactUploader.upload(
            ImmutableMap.of(
                file,
                new LocalFile(
                    file,
                    LocalFileType.OUTPUT,
                    /* artifact= */ null,
                    /* artifactMetadata= */ null)));
    PathConverter pathConverter = future.get();
    assertThat(pathConverter.apply(file)).isEqualTo("file:///test");
  }

  @Test
  public void testUnknown_notUploadedIfDirectory() throws Exception {
    Path dir = fileSystem.getPath("/test");
    // Must create directory since LocalFileType.OUTPUT triggers a filesystem check.
    dir.createDirectoryAndParents();
    ListenableFuture<PathConverter> future =
        artifactUploader.upload(
            ImmutableMap.of(
                dir,
                new LocalFile(
                    dir, LocalFileType.OUTPUT, /*artifact=*/ null, /*artifactMetadata=*/ null)));
    PathConverter pathConverter = future.get();
    assertThat(pathConverter.apply(dir)).isNull();
  }
}
