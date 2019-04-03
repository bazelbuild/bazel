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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.DigestUtil.ActionKey;
import com.google.devtools.build.lib.remote.util.StaticMetadataProvider;
import com.google.devtools.build.lib.remote.util.StringActionInput;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.Options;
import com.google.protobuf.ByteString;
import io.grpc.Context;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteActionInputFetcher}. */
@RunWith(JUnit4.class)
public class RemoteActionInputFetcherTest {

  private static final DigestHashFunction HASH_FUNCTION = DigestHashFunction.SHA256;

  private Path execRoot;
  private ArtifactRoot artifactRoot;
  private RemoteOptions options;
  private DigestUtil digestUtil;

  @Before
  public void setUp() throws IOException {
    FileSystem fs = new InMemoryFileSystem(new JavaClock(), HASH_FUNCTION);
    execRoot = fs.getPath("/exec");
    execRoot.createDirectoryAndParents();
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, execRoot.getRelative("root"));
    artifactRoot.getRoot().asPath().createDirectoryAndParents();
    options = Options.getDefaults(RemoteOptions.class);
    digestUtil = new DigestUtil(HASH_FUNCTION);
  }

  @Test
  public void testFetching() throws Exception {
    // arrange
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();
    Map<Digest, ByteString> cacheEntries = new HashMap<>();
    Artifact a1 = createRemoteArtifact("file1", "hello world", metadata, cacheEntries);
    Artifact a2 = createRemoteArtifact("file2", "fizz buzz", metadata, cacheEntries);
    MetadataProvider metadataProvider = new StaticMetadataProvider(metadata);
    AbstractRemoteActionCache remoteCache =
        new StaticRemoteActionCache(options, digestUtil, cacheEntries);
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(remoteCache, execRoot, Context.current());

    // act
    actionInputFetcher.prefetchFiles(metadata.keySet(), metadataProvider);

    // assert
    assertThat(FileSystemUtils.readContent(a1.getPath(), StandardCharsets.UTF_8))
        .isEqualTo("hello world");
    assertThat(a1.getPath().isExecutable()).isTrue();
    assertThat(FileSystemUtils.readContent(a2.getPath(), StandardCharsets.UTF_8))
        .isEqualTo("fizz buzz");
    assertThat(a2.getPath().isExecutable()).isTrue();
    assertThat(actionInputFetcher.downloadedFiles()).hasSize(2);
    assertThat(actionInputFetcher.downloadedFiles()).containsAllOf(a1.getPath(), a2.getPath());
    assertThat(actionInputFetcher.downloadsInProgress).isEmpty();
  }

  @Test
  public void testStagingVirtualActionInput() throws Exception {
    // arrange
    MetadataProvider metadataProvider = new StaticMetadataProvider(new HashMap<>());
    AbstractRemoteActionCache remoteCache =
        new StaticRemoteActionCache(options, digestUtil, new HashMap<>());
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(remoteCache, execRoot, Context.current());
    VirtualActionInput a = new StringActionInput("hello world", PathFragment.create("file1"));

    // act
    actionInputFetcher.prefetchFiles(ImmutableList.of(a), metadataProvider);

    // assert
    Path p = execRoot.getRelative(a.getExecPath());
    assertThat(FileSystemUtils.readContent(p, StandardCharsets.UTF_8)).isEqualTo("hello world");
    assertThat(p.isExecutable()).isFalse();
    assertThat(actionInputFetcher.downloadedFiles()).isEmpty();
    assertThat(actionInputFetcher.downloadsInProgress).isEmpty();
  }

  @Test
  public void testFileNotFound() throws Exception {
    // Test that we get an exception if an input file is missing

    // arrange
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();
    Artifact a =
        createRemoteArtifact("file1", "hello world", metadata, /* cacheEntries= */ new HashMap<>());
    MetadataProvider metadataProvider = new StaticMetadataProvider(metadata);
    AbstractRemoteActionCache remoteCache =
        new StaticRemoteActionCache(options, digestUtil, new HashMap<>());
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(remoteCache, execRoot, Context.current());

    // act
    try {
      actionInputFetcher.prefetchFiles(ImmutableList.of(a), metadataProvider);
      fail("expected IOException");
    } catch (IOException e) {
      // Intentionally left empty
    }

    // assert
    assertThat(actionInputFetcher.downloadedFiles()).isEmpty();
    assertThat(actionInputFetcher.downloadsInProgress).isEmpty();
  }

  @Test
  public void testIgnoreNoneRemoteFiles() throws Exception {
    // Test that files that are not remote are not downloaded

    // arrange
    Path p = execRoot.getRelative(artifactRoot.getExecPath()).getRelative("file1");
    FileSystemUtils.writeContent(p, StandardCharsets.UTF_8, "hello world");
    Artifact a = new Artifact(p, artifactRoot);
    FileArtifactValue f = FileArtifactValue.create(a);
    MetadataProvider metadataProvider = new StaticMetadataProvider(ImmutableMap.of(a, f));
    AbstractRemoteActionCache remoteCache =
        new StaticRemoteActionCache(options, digestUtil, new HashMap<>());
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(remoteCache, execRoot, Context.current());

    // act
    actionInputFetcher.prefetchFiles(ImmutableList.of(a), metadataProvider);

    // assert
    assertThat(actionInputFetcher.downloadedFiles()).isEmpty();
    assertThat(actionInputFetcher.downloadsInProgress).isEmpty();
  }

  @Test
  public void testDownloadFile() throws Exception {
    // arrange
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();
    Map<Digest, ByteString> cacheEntries = new HashMap<>();
    Artifact a1 = createRemoteArtifact("file1", "hello world", metadata, cacheEntries);
    AbstractRemoteActionCache remoteCache =
        new StaticRemoteActionCache(options, digestUtil, cacheEntries);
    RemoteActionInputFetcher actionInputFetcher =
        new RemoteActionInputFetcher(remoteCache, execRoot, Context.current());

    // act
    actionInputFetcher.downloadFile(a1.getPath(), metadata.get(a1));

    // assert
    assertThat(FileSystemUtils.readContent(a1.getPath(), StandardCharsets.UTF_8))
        .isEqualTo("hello world");
    assertThat(a1.getPath().isExecutable()).isTrue();
    assertThat(a1.getPath().isReadable()).isTrue();
  }

  private Artifact createRemoteArtifact(
      String pathFragment,
      String contents,
      Map<ActionInput, FileArtifactValue> metadata,
      Map<Digest, ByteString> cacheEntries) {
    Path p = artifactRoot.getRoot().getRelative(pathFragment);
    Artifact a = new Artifact(p, artifactRoot);
    byte[] b = contents.getBytes(StandardCharsets.UTF_8);
    HashCode h = HASH_FUNCTION.getHashFunction().hashBytes(b);
    FileArtifactValue f =
        new RemoteFileArtifactValue(h.asBytes(), b.length, /* locationIndex= */ 1);
    metadata.put(a, f);
    cacheEntries.put(DigestUtil.buildDigest(h.asBytes(), b.length), ByteString.copyFrom(b));
    return a;
  }

  private static class StaticRemoteActionCache extends AbstractRemoteActionCache {

    private final ImmutableMap<Digest, ByteString> cacheEntries;

    public StaticRemoteActionCache(
        RemoteOptions options, DigestUtil digestUtil, Map<Digest, ByteString> cacheEntries) {
      super(options, digestUtil);
      this.cacheEntries = ImmutableMap.copyOf(cacheEntries);
    }

    @Override
    ActionResult getCachedActionResult(ActionKey actionKey) {
      throw new UnsupportedOperationException();
    }

    @Override
    void upload(
        ActionKey actionKey,
        Action action,
        Command command,
        Path execRoot,
        Collection<Path> files,
        FileOutErr outErr) {
      throw new UnsupportedOperationException();
    }

    @Override
    protected ListenableFuture<Void> downloadBlob(Digest digest, OutputStream out) {
      ByteString data = cacheEntries.get(digest);
      if (data == null) {
        return Futures.immediateFailedFuture(new CacheNotFoundException(digest, digestUtil));
      }
      try {
        data.writeTo(out);
      } catch (IOException e) {
        return Futures.immediateFailedFuture(e);
      }
      return Futures.immediateFuture(null);
    }

    @Override
    public void close() {
      // Intentionally left empty.
    }
  }
}
