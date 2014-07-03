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

import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;

/**
 * Node represents visited target in the Skyframe graph after error checking.
 */
@Immutable
@ThreadSafe
public final class TargetMarkerNode implements Node {

  static final TargetMarkerNode TARGET_MARKER_INSTANCE = new TargetMarkerNode();

  private TargetMarkerNode() {
  }

  @ThreadSafe
  public static NodeKey key(Label label) {
    return new NodeKey(NodeTypes.TARGET_MARKER, label);
  }
}
