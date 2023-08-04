// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.EnvironmentVariable;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.server.CommandProtos;
import com.google.devtools.build.lib.server.CommandProtos.ExecRequest;
import com.google.protobuf.ByteString;
import java.util.Collection;

/**
 * Event triggered after building of the run command has completed and the {@link ExecRequest} has
 * been constructed.
 */
public class ExecRequestEvent implements BuildEvent {

  private final ExecRequest execRequest;
  private final boolean includeResidue;

  public ExecRequestEvent(ExecRequest execRequest, boolean includeResidue) {
    this.execRequest = execRequest;
    this.includeResidue = includeResidue;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context) {
    BuildEventStreamProtos.ExecRequestConstructed.Builder builder =
        BuildEventStreamProtos.ExecRequestConstructed.newBuilder();
    builder.setWorkingDirectory(execRequest.getWorkingDirectory());
    if (includeResidue) {
      for (ByteString argv : execRequest.getArgvList()) {
        builder.addArgv(argv);
      }
    }
    for (CommandProtos.EnvironmentVariable environmentVariable :
        execRequest.getEnvironmentVariableList()) {
      builder.addEnvironmentVariable(
          EnvironmentVariable.newBuilder()
              .setName(environmentVariable.getName())
              .setValue(environmentVariable.getValue()));
    }
    for (ByteString envVarToClear : execRequest.getEnvironmentVariableToClearList()) {
      builder.addEnvironmentVariableToClear(envVarToClear);
    }
    return GenericBuildEvent.protoChaining(this).setExecRequest(builder.build()).build();
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.execRequestId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }
}
