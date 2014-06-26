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
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A {@link NodeBuilder} for {@link FileStateNode}s.
 *
 * <p>Merely calls FileStateNode#create, but also adds a dep on the build id for files outside
 * the package roots.
 */
public class FileStateNodeBuilder implements NodeBuilder {

  private final TimestampGranularityMonitor tsgm;
  private final AtomicReference<PathPackageLocator> pkgLocator;

  public FileStateNodeBuilder(TimestampGranularityMonitor tsgm,
      AtomicReference<PathPackageLocator> pkgLocator) {
    this.tsgm = tsgm;
    this.pkgLocator = pkgLocator;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws FileStateNodeBuilderException {
    RootedPath rootedPath = (RootedPath) nodeKey.getNodeName();
    if (!pkgLocator.get().getPathEntries().contains(rootedPath.getRoot())) {
      // For files outside the package roots, add a dependency on the build_id. This is sufficient
      // for correctness; all other files will be handled by diff awareness of their respective
      // package path, but these files need to be addressed separately.
      //
      // Using the build_id here seems to introduce a performance concern because the upward
      // transitive closure of these external files will get eagerly invalidated on each
      // incremental build (e.g. if every file had a transitive dependency on the filesystem root,
      // then we'd have a big performance problem). But this a non-issue by design:
      // - We don't add a dependency on the parent directory at the package root boundary, so the
      // only transitive dependencies from files inside the package roots to external files are
      // through symlinks. So the upwards transitive closure of external files is small.
      // - The only way external source files get into the skyframe graph in the first place is
      // through symlinks outside the package roots, which we neither want to encourage nor
      // optimize for since it is not common. So the set of external files is small.
      UUID buildId = BuildVariableNode.BUILD_ID.get(env);
      if (buildId == null) {
        return null;
      }
    }
    try {
      return FileStateNode.create(rootedPath, tsgm);
    } catch (IOException e) {
      throw new FileStateNodeBuilderException(nodeKey, e);
    } catch (InconsistentFilesystemException e) {
      throw new FileStateNodeBuilderException(nodeKey, e);
    }
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link FileStateNodeBuilder#build}.
   */
  private static final class FileStateNodeBuilderException extends NodeBuilderException {
    public FileStateNodeBuilderException(NodeKey key, IOException e) {
      super(key, e);
    }

    public FileStateNodeBuilderException(NodeKey key, InconsistentFilesystemException e) {
      super(key, e);
    }
  }
}
