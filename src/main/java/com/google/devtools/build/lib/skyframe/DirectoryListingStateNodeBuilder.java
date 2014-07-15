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

import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A {@link NodeBuilder} for {@link DirectoryListingStateNode}s.
 *
 * <p>Merely calls DirectoryListingStateNode#create, but also adds a dep on the build id for
 * directories outside the package roots.
 */
public class DirectoryListingStateNodeBuilder implements NodeBuilder {

  private final AtomicReference<PathPackageLocator> pkgLocator;

  public DirectoryListingStateNodeBuilder(AtomicReference<PathPackageLocator> pkgLocator) {
    this.pkgLocator = pkgLocator;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env)
      throws DirectoryListingStateNodeBuilderException {
    RootedPath dirRootedPath = (RootedPath) nodeKey.getNodeName();
    // See the note about external files in FileStateNodeBuilder.
    if (!pkgLocator.get().getPathEntries().contains(dirRootedPath.getRoot())) {
      UUID buildId = BuildVariableNode.BUILD_ID.get(env);
      if (buildId == null) {
        return null;
      }
    }
    try {
      return DirectoryListingStateNode.create(dirRootedPath);
    } catch (IOException e) {
      throw new DirectoryListingStateNodeBuilderException(nodeKey, e);
    }
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link DirectoryListingStateNodeBuilder#build}.
   */
  private static final class DirectoryListingStateNodeBuilderException
      extends NodeBuilderException {
    public DirectoryListingStateNodeBuilderException(NodeKey key, IOException e) {
      super(key, e);
    }
  }
}
