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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

/**
 * A post-processed ConfiguredTarget which is known to be transitively error-free from action
 * conflict issues.
 */
class PostConfiguredTargetNode implements Node {

  private final ConfiguredTarget ct;

  public PostConfiguredTargetNode(ConfiguredTarget ct) {
    this.ct = Preconditions.checkNotNull(ct);
  }

  public static ImmutableList<NodeKey> keys(Iterable<LabelAndConfiguration> lacs) {
    ImmutableList.Builder<NodeKey> keys = ImmutableList.builder();
    for (LabelAndConfiguration lac : lacs) {
      keys.add(key(lac));
    }
    return keys.build();
  }

  public static NodeKey key(LabelAndConfiguration lac) {
    return new NodeKey(NodeTypes.POST_CONFIGURED_TARGET, lac);
  }

  public ConfiguredTarget getCt() {
    return ct;
  }
}
