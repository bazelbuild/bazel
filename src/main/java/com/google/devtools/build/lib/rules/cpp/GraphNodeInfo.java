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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.List;

/** Provider used to propagate information for {@link GraphNodeAspect}. */
@Immutable
public final class GraphNodeInfo implements TransitiveInfoProvider {
  private final Label label;
  private final ImmutableList<String> linkedStaticallyBy;
  private final ImmutableList<GraphNodeInfo> children;

  public GraphNodeInfo(Label label, List<Label> linkedStaticallyBy, List<GraphNodeInfo> children) {
    this.label = label;
    this.linkedStaticallyBy =
        linkedStaticallyBy == null
            ? null
            : linkedStaticallyBy.stream()
                .map(Label::toString)
                .collect(ImmutableList.toImmutableList());
    this.children = children == null ? null : ImmutableList.copyOf(children);
  }

  public Label getLabel() {
    return label;
  }

  public List<String> getLinkedStaticallyBy() {
    return linkedStaticallyBy;
  }

  public List<GraphNodeInfo> getChildren() {
    return children;
  }
}
