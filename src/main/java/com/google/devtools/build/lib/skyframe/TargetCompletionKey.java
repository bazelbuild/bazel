// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.view.config.BuildConfiguration;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeType;

/**
 * Like LabelAndConfiguration, TargetCompletionKey is used as an ArtifactOwner. The only
 * difference is that it's used specifically for target completion Artifacts.
 */
public final class TargetCompletionKey extends ActionLookupNode.ActionLookupKey {
  private final LabelAndConfiguration lac;

  public TargetCompletionKey(Label label, BuildConfiguration configuration) {
    this.lac = new LabelAndConfiguration(label, configuration);
  }

  @Override
  public Label getLabel() {
    return lac.getLabel();
  }

  @Override
  NodeKey getNodeKey() {
    return new NodeKey(NodeTypes.TARGET_COMPLETION, lac);
  }

  @Override
  NodeType getType() {
    throw new UnsupportedOperationException();
  }
}
