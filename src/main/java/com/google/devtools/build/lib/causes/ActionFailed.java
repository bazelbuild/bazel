// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.causes;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ActionCompletedId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Class describing a {@link Cause} that is associated with an action. It is uniquely determined by
 * the path to the primary output. For reference, a Label is attached as well if available.
 */
public class ActionFailed implements Cause {
  private final PathFragment execPath;
  private final Label label;
  private final String configurationChecksum;

  public ActionFailed(PathFragment execPath, Label label, @Nullable String configurationChecksum) {
    this.execPath = execPath;
    this.label = label;
    this.configurationChecksum = configurationChecksum;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("execPath", execPath)
        .add("label", label)
        .add("configurationChecksum", configurationChecksum)
        .toString();
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public BuildEventStreamProtos.BuildEventId getIdProto() {
    ActionCompletedId.Builder actionId =
        ActionCompletedId.newBuilder().setPrimaryOutput(execPath.toString());
    if (label != null) {
      actionId.setLabel(label.toString());
    }
    if (configurationChecksum != null) {
      actionId.setConfiguration(ConfigurationId.newBuilder().setId(configurationChecksum));
    }
    return BuildEventId.newBuilder().setActionCompleted(actionId).build();
  }
}
