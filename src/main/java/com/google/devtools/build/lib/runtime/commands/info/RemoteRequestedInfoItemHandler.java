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
// limitations under the License.result.add("--remote_info_request");
package com.google.devtools.build.lib.runtime.commands.info;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.PathToReplaceUtils;
import com.google.devtools.build.lib.server.CommandProtos.InfoItem;
import com.google.devtools.build.lib.server.CommandProtos.InfoResponse;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import java.io.IOException;

class RemoteRequestedInfoItemHandler implements InfoItemHandler {
  private final CommandEnvironment env;
  private final ImmutableList.Builder<InfoItem> infoItemsBuilder;

  RemoteRequestedInfoItemHandler(CommandEnvironment env) {
    this.env = env;
    this.infoItemsBuilder = ImmutableList.builder();
  }

  @Override
  public void addInfoItem(String key, byte[] value, boolean printKeys) {
    infoItemsBuilder.add(
        InfoItem.newBuilder().setKey(key).setValue(ByteString.copyFrom(value)).build());
  }

  @Override
  public void close() throws IOException {
    ImmutableList<InfoItem> infoItems = infoItemsBuilder.build();
    InfoResponse infoResponse =
        InfoResponse.newBuilder()
            .addAllPathToReplace(PathToReplaceUtils.getPathsToReplace(env))
            .addAllInfoItem(infoItems)
            .build();
    env.addResponseExtensions(ImmutableList.of(Any.pack(infoResponse)));
  }
}
