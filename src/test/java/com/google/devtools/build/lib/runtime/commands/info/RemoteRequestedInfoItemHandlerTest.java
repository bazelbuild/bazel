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
package com.google.devtools.build.lib.runtime.commands.info;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.runtime.commands.PathToReplaceUtils.bytes;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.info.InfoItemHandler.InfoItemOutputType;
import com.google.devtools.build.lib.server.CommandProtos.InfoItem;
import com.google.devtools.build.lib.server.CommandProtos.InfoResponse;
import com.google.devtools.build.lib.server.CommandProtos.PathToReplace;
import com.google.protobuf.ByteString;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class RemoteRequestedInfoItemHandlerTest extends BuildIntegrationTestCase {
  @Test
  public void testRemoteRequestedInfoItemHandlerCreation() throws Exception {
    InfoItemHandler infoItemHandler =
        InfoItemHandler.create(runtimeWrapper.newCommand(), InfoItemOutputType.RESPONSE_PROTO);
    assertThat(infoItemHandler).isInstanceOf(RemoteRequestedInfoItemHandler.class);
  }

  @Test
  public void testRemoteRequestedInfoItemHandler_noExtensionIfNothingAdded() throws Exception {
    CommandEnvironment env = runtimeWrapper.newCommand();
    try (RemoteRequestedInfoItemHandler remoteRequestedInfoItemHandler =
        new RemoteRequestedInfoItemHandler(env)) {
      // No-op so nothing is added to remoteRequestedInfoItemHandler.
    }
    assertThat(env.getResponseExtensions()).hasSize(1);
    assertThat(env.getResponseExtensions().get(0).is(InfoResponse.class)).isTrue();
    InfoResponse infoResponse = env.getResponseExtensions().get(0).unpack(InfoResponse.class);

    assertThat(infoResponse.getPathToReplaceList())
        .containsAtLeast(
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.OUTPUT_BASE)
                .setValue(bytes(env.getOutputBase().getPathString()))
                .build(),
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.BUILD_WORKING_DIRECTORY)
                .setValue(bytes(env.getWorkspace().getPathString()))
                .build(),
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.BUILD_WORKSPACE_DIRECTORY)
                .setValue(bytes(env.getWorkspace().getPathString()))
                .build());
    assertThat(infoResponse.getInfoItemList()).isEmpty();
  }

  @Test
  public void testRemoteRequestedInfoItemHandler_addTwoItems() throws Exception {
    CommandEnvironment env = runtimeWrapper.newCommand();
    try (RemoteRequestedInfoItemHandler remoteRequestedInfoItemHandler =
        new RemoteRequestedInfoItemHandler(env)) {
      remoteRequestedInfoItemHandler.addInfoItem(
          "foo", "value-foo\n".getBytes(UTF_8), /* printKeys= */ true);
      remoteRequestedInfoItemHandler.addInfoItem(
          "bar", "value-bar\n".getBytes(UTF_8), /* printKeys= */ true);
    }
    assertThat(env.getResponseExtensions()).hasSize(1);
    assertThat(env.getResponseExtensions().get(0).is(InfoResponse.class)).isTrue();
    InfoResponse infoResponse = env.getResponseExtensions().get(0).unpack(InfoResponse.class);

    assertThat(infoResponse.getPathToReplaceList())
        .containsAtLeast(
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.OUTPUT_BASE)
                .setValue(bytes(env.getOutputBase().getPathString()))
                .build(),
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.BUILD_WORKING_DIRECTORY)
                .setValue(bytes(env.getWorkspace().getPathString()))
                .build(),
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.BUILD_WORKSPACE_DIRECTORY)
                .setValue(bytes(env.getWorkspace().getPathString()))
                .build());
    assertThat(infoResponse.getInfoItemList())
        .containsExactly(
            InfoItem.newBuilder()
                .setKey("foo")
                .setValue(ByteString.copyFromUtf8("value-foo\n"))
                .build(),
            InfoItem.newBuilder()
                .setKey("bar")
                .setValue(ByteString.copyFromUtf8("value-bar\n"))
                .build());
  }
}
