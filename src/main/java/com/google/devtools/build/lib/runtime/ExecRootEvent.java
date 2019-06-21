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

package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;

/** An event that receives the execRoot of this blaze invocation. */
public class ExecRootEvent implements BuildEvent {

  private final Path execRoot;

  public ExecRootEvent(Path execRoot) {
    this.execRoot = execRoot;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context) {
    BuildEventStreamProtos.WorkspaceConfig workspaceConfigEvent =
        BuildEventStreamProtos.WorkspaceConfig.newBuilder()
            .setLocalExecRoot(execRoot.getPathString())
            .build();
    return BuildEventStreamProtos.BuildEvent.newBuilder()
        .setId(getEventId().asStreamProto())
        .setWorkspaceInfo(workspaceConfigEvent)
        .build();
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.workspaceConfigId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }
}
