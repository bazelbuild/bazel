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
import static org.junit.Assume.assumeTrue;

import build.bazel.remote.execution.v2.Action;
import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.Command;
import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.OutputFile;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.authandtls.credentialhelper.CredentialModule;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.remote.RemoteModule;
import com.google.devtools.build.lib.remote.options.RemoteStartupOptions;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.BuildSummaryStatsModule;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.SyscallCache;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A strategy that doesn't handle caching is still behind the disk cache: a hit is served before the
 * strategy, and so the backend never sees the spawn.
 */
@RunWith(JUnit4.class)
public final class SandboxBackendDiskCacheIntegrationTest extends BuildIntegrationTestCase {
  private final DigestUtil digestUtil =
      new DigestUtil(SyscallCache.NO_CACHE, DigestHashFunction.SHA256);

  private Path dir;
  private FakeBackend fake;

  @Override
  protected ImmutableList<Class<? extends OptionsBase>> getStartupOptionClasses() {
    return ImmutableList.<Class<? extends OptionsBase>>builder()
        .addAll(super.getStartupOptionClasses())
        .add(RemoteStartupOptions.class)
        .build();
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new CredentialModule())
        .addBlazeModule(new RemoteModule())
        .addBlazeModule(new BuildSummaryStatsModule())
        .addBlazeModule(new BlockWaitingModule());
  }

  private Path script;

  // Called again when the server is restarted, which forgets the options.
  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    // Backends can't be reached on Windows yet, and these tests use unix sockets and shell scripts.
    assumeTrue(SandboxBackendTransport.isSupported());
    if (fake == null) {
      startBackend();
    }
    addOptions(
        "--spawn_strategy=b",
        "--sandbox_backend=b=" + script,
        "--disk_cache=" + dir.resolve("disk-cache"));
  }

  private void startBackend() throws Exception {
    dir = Files.createTempDirectory(Path.of("/tmp"), "sbk-dc-");
    fake = FakeBackend.onUnixSocket(dir.resolve("s").toString());
    script = dir.resolve("backend.sh");
    // Only makes the socket appear, by linking it to the socket of the server in this JVM.
    Files.writeString(
        script,
        "#!/bin/sh\n"
            + "for a in \"$@\"; do case $a in --socket=*) socket=${a#--socket=};; esac; done\n"
            + "ln -s \""
            + dir.resolve("s")
            + "\" \"$socket\"\n"
            + "exec sleep 600\n");
    script.toFile().setExecutable(true);
    fake.setExecuteHandler(
        (request, responses) -> {
          Action action = Action.parseFrom(fake.stored.get(request.getActionDigest().getHash()));
          Command command = Command.parseFrom(fake.stored.get(action.getCommandDigest().getHash()));
          Path file = dir.resolve("staged");
          Files.writeString(file, "from the backend");
          Digest digest = digestUtil.computeAsUtf8("from the backend");
          fake.serve(digest, FileUriBlobs.encode(file.toString()));
          ActionResult.Builder result = ActionResult.newBuilder();
          for (String output : command.getOutputPathsList()) {
            result.addOutputFiles(OutputFile.newBuilder().setPath(output).setDigest(digest));
          }
          FakeBackend.respond(responses, result.build());
        });
  }

  @After
  public void stopBackend() throws Exception {
    SandboxBackends.stopAll();
    if (fake != null) {
      fake.close();
      try (var files = Files.walk(dir)) {
        files.sorted(Comparator.reverseOrder()).map(Path::toFile).forEach(f -> f.delete());
      }
    }
  }

  private void cleanAndRestartServer() throws Exception {
    getOutputBase().getRelative("action_cache").deleteTreesBelow();
    createRuntimeWrapper();
  }

  @Test
  public void diskCacheHit_backendReceivesNoExecute() throws Exception {
    write("a/BUILD", "genrule(name = 'g', outs = ['out.txt'], cmd = 'exit 1')");

    buildTarget("//a:g");
    assertThat(fake.executes).hasSize(1);
    assertThat(readContentAsLatin1String(getArtifacts("//a:g").get(0))).isEqualTo("from the backend");

    cleanAndRestartServer();
    buildTarget("//a:g");

    events.assertContainsInfo("1 disk cache hit");
    assertThat(readContentAsLatin1String(getArtifacts("//a:g").get(0))).isEqualTo("from the backend");
    // Neither Execute nor any other call after the backend was asked to run the action once.
    assertThat(fake.executes).hasSize(1);
  }
}
