// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.remote.worker;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.remote.util.Utils.getFromFuture;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.GetTreeRequest;
import build.bazel.remote.execution.v2.GetTreeResponse;
import com.google.devtools.build.lib.remote.SimpleBlobStoreActionCache;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import io.grpc.stub.StreamObserver;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link com.google.devtools.build.remote.worker.CasServer} */
@RunWith(JUnit4.class)
public class CasServerTest {
  private Digest uploadDirectory(
      SimpleBlobStoreActionCache cache, DigestUtil digestUtil, Directory directory)
      throws IOException, InterruptedException {
    Digest digest = digestUtil.compute(directory);
    getFromFuture(cache.uploadBlob(digest, directory.toByteString()));
    return digest;
  }

  @Test
  public void testGetTree() throws IOException, InterruptedException {
    FileSystem fileSystem = new InMemoryFileSystem();
    Path cacheDir = fileSystem.getPath("/cache");
    cacheDir.createDirectory();
    DigestUtil digestUtil = new DigestUtil(DigestHashFunction.DEFAULT_HASH_FOR_TESTS);
    OnDiskBlobStoreActionCache cache =
        new OnDiskBlobStoreActionCache(new RemoteOptions(), cacheDir, digestUtil);
    CasServer casServer = new CasServer(cache);

    // Fake directory structure:
    // root
    // - foo
    //   - bar
    //     - baz
    // - quux
    Directory emptyDir = Directory.newBuilder().build();
    Digest emptyDirDigest = uploadDirectory(cache, digestUtil, emptyDir);
    Directory barDir =
        Directory.newBuilder()
            .addDirectories(
                DirectoryNode.newBuilder().setName("baz").setDigest(emptyDirDigest).build())
            .build();
    Digest barDigest = uploadDirectory(cache, digestUtil, barDir);
    Directory fooDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("bar").setDigest(barDigest).build())
            .build();
    Digest fooDigest = uploadDirectory(cache, digestUtil, fooDir);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("foo").setDigest(fooDigest).build())
            .addDirectories(
                DirectoryNode.newBuilder().setName("quux").setDigest(emptyDirDigest).build())
            .build();
    Digest rootDigest = uploadDirectory(cache, digestUtil, rootDir);

    GetTreeRequest request = GetTreeRequest.newBuilder().setRootDigest(rootDigest).build();
    GetTreeResponse.Builder fullResponseBuilder = GetTreeResponse.newBuilder();
    AtomicBoolean completed = new AtomicBoolean(false);
    AtomicBoolean hadError = new AtomicBoolean(false);
    StreamObserver<GetTreeResponse> responseObserver =
        new StreamObserver<GetTreeResponse>() {
          @Override
          public void onNext(GetTreeResponse response) {
            fullResponseBuilder.mergeFrom(response);
          }

          @Override
          public void onError(Throwable t) {
            hadError.set(true);
          }

          @Override
          public void onCompleted() {
            completed.set(true);
          }
        };

    casServer.getTree(request, responseObserver);

    assertThat(completed.get()).isTrue();
    assertThat(hadError.get()).isFalse();
    GetTreeResponse fullResponse = fullResponseBuilder.build();
    assertThat(fullResponse.getNextPageToken()).isEmpty();
    assertThat(fullResponse.getDirectoriesList())
        .containsExactly(rootDir, fooDir, barDir, emptyDir);
  }
}
