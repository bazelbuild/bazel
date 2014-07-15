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

import com.google.common.base.Predicate;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeType;

/**
 * Node types in Skyframe.
 */
public final class NodeTypes {
  public static final NodeType BUILD_VARIABLE = new NodeType("BUILD_VARIABLE", false);
  public static final NodeType FILE_STATE = new NodeType("FILE_STATE", false);
  public static final NodeType DIRECTORY_LISTING_STATE =
      new NodeType("DIRECTORY_LISTING_STATE", false);
  public static final NodeType FILE_SYMLINK_CYCLE_UNIQUENESS_NODE =
      NodeType.computed("FILE_SYMLINK_CYCLE_UNIQUENESS_NODE");
  public static final NodeType FILE = NodeType.computed("FILE");
  public static final NodeType DIRECTORY_LISTING = NodeType.computed("DIRECTORY_LISTING");
  public static final NodeType PACKAGE_LOOKUP = NodeType.computed("PACKAGE_LOOKUP");
  public static final NodeType CONTAINING_PACKAGE_LOOKUP =
      NodeType.computed("CONTAINING_PACKAGE_LOOKUP");
  public static final NodeType GLOB = NodeType.computed("GLOB");
  public static final NodeType PACKAGE = NodeType.computed("PACKAGE");
  public static final NodeType TARGET_MARKER = NodeType.computed("TARGET_MARKER");
  public static final NodeType TARGET_PATTERN = NodeType.computed("TARGET_PATTERN");
  public static final NodeType RECURSIVE_PKG = NodeType.computed("RECURSIVE_PKG");
  public static final NodeType TRANSITIVE_TARGET = NodeType.computed("TRANSITIVE_TARGET");
  public static final NodeType CONFIGURED_TARGET = NodeType.computed("CONFIGURED_TARGET");
  public static final NodeType POST_CONFIGURED_TARGET = NodeType.computed("POST_CONFIGURED_TARGET");
  public static final NodeType TARGET_COMPLETION = NodeType.computed("TARGET_COMPLETION");
  public static final NodeType CONFIGURATION_COLLECTION =
      NodeType.computed("CONFIGURATION_COLLECTION");
  public static final NodeType ARTIFACT = NodeType.computed("ARTIFACT");
  public static final NodeType ACTION_EXECUTION = NodeType.computed("ACTION_EXECUTION");
  public static final NodeType ACTION_LOOKUP = NodeType.computed("ACTION_LOOKUP");
  public static final NodeType BUILD_INFO_COLLECTION = NodeType.computed("BUILD_INFO_COLLECTION");
  public static final NodeType BUILD_INFO = NodeType.computed("BUILD_INFO");

  public static Predicate<NodeKey> hasNodeType(final NodeType nodeType) {
    return new Predicate<NodeKey>() {
      @Override
      public boolean apply(NodeKey key) {
        return key.getNodeType() == nodeType;
      }
    };
  }
}
