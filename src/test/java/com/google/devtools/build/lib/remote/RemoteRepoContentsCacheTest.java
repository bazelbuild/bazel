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

package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.OutputDirectory;
import build.bazel.remote.execution.v2.OutputFile;
import build.bazel.remote.execution.v2.Platform;
import build.bazel.remote.execution.v2.Tree;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.remote.common.ActionKey;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.InMemoryCacheClient;
import com.google.devtools.build.lib.skyframe.EnvironmentVariableValue;
import com.google.devtools.build.lib.skyframe.RepoEnvironmentFunction;
import com.google.devtools.build.lib.unsafe.StringUnsafe;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.protobuf.ByteString;
import java.time.Duration;
import java.util.HashMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class RemoteRepoContentsCacheTest {

  private static final String PREDECLARED_INPUT_HASH = "predeclared_hash";

  private FileSystem fs;
  private DigestUtil digestUtil;

  private final HashMap<Digest, byte[]> cas = new HashMap<>();

  private Path repoDirInRemoteFs;
  private final Reporter eventHandler = new Reporter();
  private SkyFunction.Environment env;
  private RepositoryName repoName;
  private InMemoryCacheClient inMemoryCacheClient;
  private RemoteRepoContentsCacheImpl remoteRepoContentsCache;

  @Before
  public void setUp() throws Exception {
    fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    digestUtil = new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);
    env = mock(SkyFunction.Environment.class);
    when(env.getListener()).thenReturn(eventHandler);
    when(env.valuesMissing()).thenReturn(false);

    Path installBase = fs.getPath("/install_base");
    Path outputBase = fs.getPath("/output_base");
    Path userRoot = fs.getPath("/user_root");
    var serverDirectories = new ServerDirectories(installBase, outputBase, userRoot);
    Path workspace = fs.getPath("/workspace");
    var directories = new BlazeDirectories(serverDirectories, workspace, "bazel");

    inMemoryCacheClient = new InMemoryCacheClient(cas);
    var cache = new CombinedCache(inMemoryCacheClient, null, null, digestUtil, null);
    remoteRepoContentsCache =
        new RemoteRepoContentsCacheImpl(
            directories, cache, "test-req", "test-cmd", true, false, false);

    repoName = RepositoryName.createUnvalidated("my_repo");
    var repoDir = fs.getPath("/output_base/external/my_repo");
    var externalDir = fs.getPath("/output_base/external");
    var remoteFs = new RemoteExternalOverlayFileSystem(externalDir.asFragment(), fs);

    var prefetcher = mock(AbstractActionInputPrefetcher.class);
    when(prefetcher.prefetchFilesInterruptibly(any(), any(), any(), any(), any()))
        .thenReturn(Futures.immediateVoidFuture());
    remoteFs.beforeCommand(
        cache, prefetcher, eventHandler, "test-req", "test-cmd", null, Duration.ofHours(1));

    repoDirInRemoteFs = remoteFs.getPath(repoDir.asFragment());
  }

  // addOutputFiles and addOutputDirectories are deprecated in REAPI v2 in favor of addOutputPaths,
  // but are populated in the test action key for backwards compatibility with older RE backends.
  @SuppressWarnings("deprecation")
  private ActionKey getActionKey(String inputHash) {
    var command =
        Command.newBuilder()
            .addArguments("0336b325-9db8-4592-a5eb-79b4970bc4ce")
            .addOutputPaths(".recorded_inputs")
            .addOutputPaths("repo_contents")
            .addOutputFiles(".recorded_inputs")
            .addOutputDirectories("repo_contents")
            .setPlatform(Platform.getDefaultInstance())
            .build();
    var inputRoot = Directory.getDefaultInstance();
    var baseAction =
        Action.newBuilder()
            .setCommandDigest(digestUtil.compute(command))
            .setInputRootDigest(digestUtil.compute(inputRoot))
            .setPlatform(Platform.getDefaultInstance())
            .build();
    var action =
        baseAction.toBuilder()
            .setSalt(ByteString.copyFrom(StringUnsafe.getByteArray(inputHash)))
            .build();
    return new ActionKey(digestUtil.compute(action));
  }

  private void setupCache(String markerContent) throws Exception {
    byte[] markerBytes = markerContent.getBytes(ISO_8859_1);
    Digest markerDigest = digestUtil.compute(markerBytes);
    var unusedBlob =
        inMemoryCacheClient.uploadBlob(null, markerDigest, ByteString.copyFrom(markerBytes), false);

    // Setup ActionResult pointing to the marker file and dummy directory.
    var actionResult =
        ActionResult.newBuilder()
            .addOutputFiles(
                OutputFile.newBuilder().setPath(".recorded_inputs").setDigest(markerDigest).build())
            .addOutputDirectories(
                OutputDirectory.newBuilder()
                    .setPath("repo_contents")
                    .setTreeDigest(digestUtil.compute(Tree.getDefaultInstance()))
                    .build())
            .build();

    var unused =
        inMemoryCacheClient.uploadActionResult(
            null, getActionKey(PREDECLARED_INPUT_HASH), actionResult);
  }

  @Test
  public void lookupCache_unauthorizedEnvVar_rejected() throws Exception {
    setupCache(PREDECLARED_INPUT_HASH + "\nENV:UNAUTHORIZED_VAR secret_value\n");

    var allowedEnviron = ImmutableSet.of("AUTHORIZED_VAR");

    boolean hit =
        remoteRepoContentsCache.lookupCache(
            repoName, repoDirInRemoteFs, PREDECLARED_INPUT_HASH, allowedEnviron, env);

    assertThat(hit).isFalse();
    verify(env, never()).getValue(any());
  }

  @Test
  public void lookupCache_authorizedEnvVar_success() throws Exception {
    setupCache(PREDECLARED_INPUT_HASH + "\nENV:AUTHORIZED_VAR secret_value\n");

    // Mock Skyframe to return the expected value for the env var.
    var envKey = RepoEnvironmentFunction.key("AUTHORIZED_VAR");
    var envValue = new EnvironmentVariableValue("secret_value");
    when(env.getValue(envKey)).thenReturn(envValue);

    // Mock getValuesAndExceptions
    var lookupResult = mock(SkyframeLookupResult.class);
    when(lookupResult.get(envKey)).thenReturn(envValue);
    when(env.getValuesAndExceptions(any())).thenReturn(lookupResult);

    var allowedEnviron = ImmutableSet.of("AUTHORIZED_VAR");

    boolean hit =
        remoteRepoContentsCache.lookupCache(
            repoName, repoDirInRemoteFs, PREDECLARED_INPUT_HASH, allowedEnviron, env);

    assertThat(hit).isTrue();
    verify(env).getValue(envKey);
  }
}
