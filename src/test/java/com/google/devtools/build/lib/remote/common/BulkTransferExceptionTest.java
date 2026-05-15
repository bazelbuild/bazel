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
import static org.junit.Assert.assertThrows;

import build.bazel.remote.execution.v2.Digest;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.vfs.PathFragment;
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
  public void getLostArtifacts_returnsResolvableInputsAndSkipsUnrewindableCacheMisses() {
    var annotatedActionInput = ActionInputHelper.fromPath("bazel-out/k8-fastbuild/bin/foo.facts");
    var resolvedActionInput = ActionInputHelper.fromPath("bazel-out/k8-fastbuild/bin/bar.facts");

    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(
        new CacheNotFoundException(digest("abc", 1), annotatedActionInput.getExecPath()));
    bulkTransferException.add(
        new CacheNotFoundException(digest("def", 2), resolvedActionInput.getExecPath()));
    bulkTransferException.add(new CacheNotFoundException(digest("ghi", 3), "stdout"));
    bulkTransferException.add(
        new CacheNotFoundException(
            digest("jkl", 4), PathFragment.create("bazel-out/k8-fastbuild/bin/foo.out")));

    assertThat(
            bulkTransferException
                .getLostArtifacts(
                    execPath ->
                        execPath.equals(annotatedActionInput.getExecPath())
                            ? annotatedActionInput
                            : execPath.equals(resolvedActionInput.getExecPath())
                                ? resolvedActionInput
                                : null)
                .byDigest())
        .containsExactly("abc/1", annotatedActionInput, "def/2", resolvedActionInput);
  }

  @Test
  public void getLostArtifacts_skipsExecPathThatDoesNotResolveForCurrentAction() {
    var actionInput = ActionInputHelper.fromPath("bazel-out/k8-fastbuild/bin/foo.facts");

    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(
        new CacheNotFoundException(digest("abc", 1), actionInput.getExecPath()));

    assertThat(bulkTransferException.getLostArtifacts(unused -> null).byDigest()).isEmpty();
  }

  @Test
  public void getLostArtifacts_returnsEmptyIfNoCacheMissResolvesToActionInput() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new CacheNotFoundException(digest("abc", 1), "stdout"));
    bulkTransferException.add(
        new CacheNotFoundException(
            digest("def", 2), PathFragment.create("bazel-out/k8-fastbuild/bin/foo.out")));

    assertThat(bulkTransferException.getLostArtifacts(unused -> null).byDigest()).isEmpty();
  }

  @Test
  public void getLostArtifacts_requiresFilenameForUnannotatedCacheMisses() {
    BulkTransferException bulkTransferException = new BulkTransferException();
    bulkTransferException.add(new CacheNotFoundException(digest("abc", 1)));

    assertThrows(
        IllegalArgumentException.class,
        () -> bulkTransferException.getLostArtifacts(ActionInputHelper::fromPath));
  }

  private static Digest digest(String hash, long sizeBytes) {
    return Digest.newBuilder().setHash(hash).setSizeBytes(sizeBytes).build();
  }
}
