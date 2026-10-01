// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.common;

import static com.google.common.truth.Truth.assertThat;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BulkTransferExceptionTest {

  @Test
  public void shouldProvideGenericMessageIfNoAddedException() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    assertThat(bulkTransferException.getMessage()).isEqualTo("Unknown error during bulk transfer");
  }

  @Test
  public void shouldPreserveMessageAsIsFromSingleException() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException("Failure Type A"));
    assertThat(bulkTransferException.getMessage()).isEqualTo("Failure Type A");
  }

  @Test
  public void shouldSortAndRemoveDuplicatesWhenAggregatingMessages() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException("Failure Type B"));
    bulkTransferException.add(new IOException("Failure Type A"));
    bulkTransferException.add(new IOException("Failure Type B"));
    assertThat(bulkTransferException.getMessage())
        .isEqualTo(
            "Multiple errors during bulk transfer:\n" + "Failure Type A\n" + "Failure Type B");
  }

  @Test
  public void shouldProvideGenericMessageIfOnlyNullMessages() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException());
    assertThat(bulkTransferException.getMessage()).isEqualTo("Unknown error during bulk transfer");
  }

  @Test
  public void shouldIgnoreNullMessagesWhenGettingMessage() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new IOException("Failure Type A"));
    bulkTransferException.add(new IOException());
    assertThat(bulkTransferException.getMessage()).isEqualTo("Failure Type A");
  }

  @Test
  public void getLostArtifacts_fileBelowSourceDirectoryInExternalRepo_directoryIsLost() {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var sourceDir =
        new Artifact.SourceArtifact(
            ArtifactRoot.asExternalSourceRoot(
                Root.fromPath(fs.getPath("/output_base/external/repo"))),
            PathFragment.create("external/repo/dir"),
            /* owner= */ () -> null);
    var digest = Digest.newBuilder().setHash("abc").setSizeBytes(3).build();
    var e =
        new BulkTransferException(
            new CacheNotFoundException(digest, PathFragment.create("external/repo/dir/sub/file")));

    var lostArtifacts =
        e.getLostArtifacts(execPath -> execPath.equals(sourceDir.getExecPath()) ? sourceDir : null);

    assertThat(lostArtifacts.byDigest()).containsExactly(DigestUtil.toString(digest), sourceDir);
  }

  @Test
  public void getLostArtifacts_fileBelowSourceDirectoryInMainRepo_nothingIsLost() {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var sourceDir =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/workspace"))), "dir");
    var digest = Digest.newBuilder().setHash("abc").setSizeBytes(3).build();
    var e =
        new BulkTransferException(
            new CacheNotFoundException(digest, PathFragment.create("dir/sub/file")));

    var lostArtifacts =
        e.getLostArtifacts(execPath -> execPath.equals(sourceDir.getExecPath()) ? sourceDir : null);

    assertThat(lostArtifacts.isEmpty()).isTrue();
  }

  @Test
  public void getLostArtifacts_fileBelowDerivedArtifact_nothingIsLost() {
    var fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    var derived =
        ActionsTestUtil.createArtifact(
            ArtifactRoot.asDerivedRoot(fs.getPath("/exec"), RootType.OUTPUT, "out"), "dir");
    var digest = Digest.newBuilder().setHash("abc").setSizeBytes(3).build();
    var e =
        new BulkTransferException(
            new CacheNotFoundException(digest, derived.getExecPath().getRelative("file")));

    var lostArtifacts =
        e.getLostArtifacts(execPath -> execPath.equals(derived.getExecPath()) ? derived : null);

    assertThat(lostArtifacts.isEmpty()).isTrue();
  }
}
