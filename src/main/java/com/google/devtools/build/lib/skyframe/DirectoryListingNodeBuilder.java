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

import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;

import javax.annotation.Nullable;

/**
 * A {@link NodeBuilder} for {@link DirectoryListingNode}s.
 */
final class DirectoryListingNodeBuilder implements NodeBuilder {

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws DirectoryListingNodeBuilderException {
    RootedPath dirRootedPath = (RootedPath) nodeKey.getNodeName();

    FileNode dirFileNode = (FileNode) env.getDep(FileNode.key(dirRootedPath));
    if (dirFileNode == null) {
      return null;
    }

    RootedPath realDirRootedPath = dirFileNode.realRootedPath();
    if (!dirFileNode.isDirectory()) {
      // Recall that the directory is assumed to exist (see DirectoryListingNode#key).
      throw new DirectoryListingNodeBuilderException(nodeKey,
          new InconsistentFilesystemException(dirRootedPath.asPath()
              + " is no longer an existing directory. Did you delete it during the build?"));
    }

    DirectoryListingStateNode directoryListingStateNode;
    try {
      directoryListingStateNode = (DirectoryListingStateNode) env.getDepOrThrow(
          DirectoryListingStateNode.key(realDirRootedPath), IOException.class);
    } catch (IOException e) {
      throw new DirectoryListingNodeBuilderException(nodeKey, e);
    }
    if (directoryListingStateNode == null) {
      return null;
    }

    return DirectoryListingNode.node(dirRootedPath, dirFileNode, directoryListingStateNode);
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link DirectoryListingNodeBuilder#build}.
   */
  private static final class DirectoryListingNodeBuilderException extends NodeBuilderException {
    public DirectoryListingNodeBuilderException(NodeKey key, IOException e) {
      super(key, e);
    }

    public DirectoryListingNodeBuilderException(NodeKey key, InconsistentFilesystemException e) {
      super(key, e);
    }
  }
}
