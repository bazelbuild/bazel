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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.common.options.OptionsParser;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RemoteGrpcServiceConfig}. */
@RunWith(JUnit4.class)
public final class RemoteGrpcServiceConfigTest {

  private static RemoteOptions parseRemoteOptions(String... args) throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(RemoteOptions.class).build();
    parser.parse(args);
    return parser.getOptions(RemoteOptions.class);
  }

  @Test
  public void create_usesRemoteTimeoutForRemoteServices() {
    assertThat(RemoteGrpcServiceConfig.create(Duration.ofSeconds(123)))
        .containsExactly(
            "methodConfig",
            ImmutableList.of(
                ImmutableMap.of(
                    "name",
                    ImmutableList.of(
                        ImmutableMap.of("service", "build.bazel.remote.execution.v2.ActionCache"),
                        ImmutableMap.of("service", "build.bazel.remote.execution.v2.Capabilities"),
                        ImmutableMap.of(
                            "service", "build.bazel.remote.execution.v2.ContentAddressableStorage"),
                        ImmutableMap.of("service", "google.bytestream.ByteStream"),
                        ImmutableMap.of("service", "build.bazel.remote.asset.v1.Fetch")),
                    "timeout",
                    "123s")));
  }

  @Test
  public void create_usesUserSuppliedJsonFile() throws Exception {
    Scratch scratch = new Scratch(new InMemoryFileSystem(DigestHashFunction.SHA256));
    Path workspace = scratch.dir("/workspace with spaces");
    Path workingDirectory = scratch.dir("/workspace with spaces/subdir");
    String serviceConfig =
        """
        {
          "methodConfig": [
            {
              "name": [
                {
                  "service": "build.bazel.remote.execution.v2.ActionCache",
                  "method": "GetActionResult"
                },
                {"service": "google.bytestream.ByteStream"}
              ],
              "timeout": "3.500s"
            }
          ]
        }
        """;
    scratch.file("/workspace with spaces/subdir/service_config.json", serviceConfig);
    scratch.file(System.getProperty("user.home") + "/service_config.json", serviceConfig);
    scratch.file("/workspace with spaces/subdir/~/service_config.json", "{}");

    for (String configPath :
        ImmutableList.of(
            "service_config.json",
            "/workspace with spaces/subdir/service_config.json",
            "%workspace%/subdir/service_config.json",
            "%workspace%" + File.separator + "subdir" + File.separator + "service_config.json",
            "~/service_config.json")) {
      assertThat(
              RemoteGrpcServiceConfig.create(
                  parseRemoteOptions("--remote_grpc_service_config=" + configPath),
                  workingDirectory,
                  workspace))
          .containsExactly(
              "methodConfig",
              ImmutableList.of(
                  ImmutableMap.of(
                      "name",
                      ImmutableList.of(
                          ImmutableMap.of(
                              "service",
                              "build.bazel.remote.execution.v2.ActionCache",
                              "method",
                              "GetActionResult"),
                          ImmutableMap.of("service", "google.bytestream.ByteStream")),
                      "timeout",
                      "3.500s")));
    }
  }

  @Test
  public void create_expandsWorkspaceBeforeNormalizingParentReferences() throws Exception {
    Scratch scratch = new Scratch(new InMemoryFileSystem(DigestHashFunction.SHA256));
    Path workspace = scratch.dir("/parent/workspace");
    Path workingDirectory = scratch.dir("/parent/workspace/subdir");
    scratch.file("/parent/service_config.json", "{\"methodConfig\": []}");
    scratch.file("/parent/workspace/subdir/service_config.json", "{}");

    assertThat(
            RemoteGrpcServiceConfig.create(
                parseRemoteOptions(
                    "--remote_grpc_service_config=%workspace%/../service_config.json"),
                workingDirectory,
                workspace))
        .containsExactly("methodConfig", ImmutableList.of());
  }

  @Test
  public void create_workspaceRelativePathRequiresWorkspace() throws Exception {
    Scratch scratch = new Scratch(new InMemoryFileSystem(DigestHashFunction.SHA256));
    Path workingDirectory = scratch.dir("/working-directory");

    IOException e =
        Assert.assertThrows(
            IOException.class,
            () ->
                RemoteGrpcServiceConfig.create(
                    parseRemoteOptions("--remote_grpc_service_config=%workspace%/config.json"),
                    workingDirectory,
                    null));

    assertThat(e).hasMessageThat().contains("%workspace% requires a workspace");
  }

  @Test
  public void create_rejectsMissingWorkspaceRelativeFile() throws Exception {
    Scratch scratch = new Scratch(new InMemoryFileSystem(DigestHashFunction.SHA256));
    Path workspace = scratch.dir("/workspace");
    Path workingDirectory = scratch.dir("/workspace/subdir");

    IOException e =
        Assert.assertThrows(
            IOException.class,
            () ->
                RemoteGrpcServiceConfig.create(
                    parseRemoteOptions("--remote_grpc_service_config=%workspace%/missing.json"),
                    workingDirectory,
                    workspace));

    assertThat(e).hasMessageThat().contains("/workspace/missing.json");
  }

  @Test
  public void create_rejectsUnsupportedJsonFields() throws Exception {
    Scratch scratch = new Scratch(new InMemoryFileSystem(DigestHashFunction.SHA256));
    Path workspace = scratch.dir("/workspace");
    scratch.file(
        "/workspace/service_config.json",
        """
        {
          "methodConfig": [
            {
              "name": [{"service": "google.bytestream.ByteStream"}],
              "timeout": "1s",
              "retryPolicy": {}
            }
          ]
        }
        """);

    IOException e =
        Assert.assertThrows(
            IOException.class,
            () ->
                RemoteGrpcServiceConfig.create(
                    parseRemoteOptions("--remote_grpc_service_config=service_config.json"),
                    workspace,
                    workspace));

    assertThat(e)
        .hasMessageThat()
        .contains("methodConfig[0] contains unsupported field 'retryPolicy'");
  }
}
