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

import javax.annotation.Nullable;

/**
 * A {@link NodeBuilder} for {@link DirectoryListingNode}s.
 */
final class DirectoryListingNodeBuilder implements NodeBuilder {

  private final AtomicReference<PathPackageLocator> pkgLocator;

  public DirectoryListingNodeBuilder(AtomicReference<PathPackageLocator> pkgLocator) {
    this.pkgLocator = pkgLocator;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws DirectoryListingNodeBuilderException {
    RootedPath dirRootedPath = (RootedPath) nodeKey.getNodeName();

    FileNode dirFile = (FileNode) env.getDep(FileNode.key(dirRootedPath));
    if (dirFile == null) {
      return null;
    }
    if (!dirFile.isDirectory()) {
      // Recall that the directory is assumed to exist (see DirectoryListingNode#key).
      throw new DirectoryListingNodeBuilderException(nodeKey,
          new InconsistentFilesystemException(dirRootedPath.asPath()
              + " is no longer an existing directory. Did you delete it during the build?"));
    }

    if (!dirFile.realRootedPath().equals(dirFile.rootedPath())) {
      // If the realpath of the directory does not match the actual path, then look up the real
      // directory that it corresponds to and fetch the directory entries from that. This saves
      // a little bit of I/O, and more importantly it keeps listings of "virtual" directories
      // correct when changes are reported in the real directory (which what file delta interfaces
      // usually provide).
      //
      // Note that FileNode#isDirectory follows symlinks.
      DirectoryListingNode resolvedDir = (DirectoryListingNode)
          env.getDep(DirectoryListingNode.key(dirFile.realRootedPath()));
      if (resolvedDir == null) {
        return null;
      }
      return new DirectoryListingNode(dirRootedPath, resolvedDir);
    }

    DirectoryListingNode node;
    try {
      node = DirectoryListingNode.nodeForRootedPath(dirRootedPath);
    } catch (IOException e) {
      throw new DirectoryListingNodeBuilderException(nodeKey, e);
    }

    // See the note about external files in FileNodeBuilder.
    if (!pkgLocator.get().getPathEntries().contains(dirRootedPath.getRoot())) {
      UUID buildId = BuildVariableNode.BUILD_ID.get(env);
      if (buildId == null) {
        return null;
      }
    }

    return node;
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
