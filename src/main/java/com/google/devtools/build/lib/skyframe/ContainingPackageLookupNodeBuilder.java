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

import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeKey;

import javax.annotation.Nullable;

/**
 * NodeBuilder for {@link ContainingPackageLookupNode}s.
 */
public class ContainingPackageLookupNodeBuilder implements NodeBuilder {

  @Override
  public Node build(NodeKey nodeKey, Environment env) {
    PathFragment dir = (PathFragment) nodeKey.getNodeName();

    PackageLookupNode pkgLookupNode = (PackageLookupNode) env.getDep(PackageLookupNode.key(dir));
    if (pkgLookupNode == null) {
      return null;
    }

    if (pkgLookupNode.packageExists()) {
      return new ContainingPackageLookupNode(dir);
    }

    PathFragment parentDir = dir.getParentDirectory();
    if (parentDir == null) {
      return new ContainingPackageLookupNode(null);
    }
    ContainingPackageLookupNode parentContainingPkgLookupNode =
        (ContainingPackageLookupNode) env.getDep(ContainingPackageLookupNode.key(parentDir));
    if (parentContainingPkgLookupNode == null) {
      return null;
    }
    return new ContainingPackageLookupNode(
        parentContainingPkgLookupNode.getContainingPackageNameOrNull());
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }
}
