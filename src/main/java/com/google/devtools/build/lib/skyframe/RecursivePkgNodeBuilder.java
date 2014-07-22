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

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeKey;

import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * RecursivePkgNodeBuilder builds up the set of packages underneath a given directory
 * transitively.
 *
 * <p>Example: foo/BUILD, foo/sub/x, foo/subpkg/BUILD would yield transitive packages "foo" and
 * "foo/subpkg".
 */
public class RecursivePkgNodeBuilder implements NodeBuilder {

  private static final Order ORDER = Order.STABLE_ORDER;

  @Override
  public Node build(NodeKey nodeKey, Environment env) {
    RootedPath rootedPath = (RootedPath) nodeKey.getNodeName();
    Path root = rootedPath.getRoot();
    PathFragment rootRelativePath = rootedPath.getRelativePath();

    NodeKey fileKey = FileNode.key(rootedPath);
    FileNode fileNode = (FileNode) env.getDep(fileKey);
    if (fileNode == null) {
      return null;
    }

    if (!fileNode.isDirectory()) {
      return new RecursivePkgNode(NestedSetBuilder.<String>emptySet(ORDER));
    }

    if (fileNode.isSymlink()) {
      // We do not follow directory symlinks when we look recursively for packages. It also
      // prevents symlink loops.
      return new RecursivePkgNode(NestedSetBuilder.<String>emptySet(ORDER));
    }

    PackageLookupNode pkgLookupNode =
        (PackageLookupNode) env.getDep(PackageLookupNode.key(rootRelativePath));
    if (pkgLookupNode == null) {
      return null;
    }

    NestedSetBuilder<String> packages = new NestedSetBuilder<>(ORDER);

    if (pkgLookupNode.packageExists()) {
      if (pkgLookupNode.getRoot().equals(root)) {
        try {
          PackageNode pkgNode = (PackageNode)
              env.getDepOrThrow(PackageNode.key(rootRelativePath),
                  NoSuchPackageException.class);
          if (pkgNode == null) {
            return null;
          }
          packages.add(pkgNode.getPackage().getName());
        } catch (NoSuchPackageException e) {
          // The package had errors, but don't fail-fast as there might subpackages below the
          // current directory.
          env.getListener().error(null,
              "package contains errors: " + rootRelativePath.getPathString());
          if (e.getPackage() != null) {
            packages.add(e.getPackage().getName());
          }
        }
      }
      // The package lookup succeeded, but was under a different root. We still, however, need to
      // recursively consider subdirectories. For example:
      //
      //  Pretend --package_path=rootA/workspace:rootB/workspace and these are the only files:
      //    rootA/workspace/foo/
      //    rootA/workspace/foo/bar/BUILD
      //    rootB/workspace/foo/BUILD
      //  If we're doing a recursive package lookup under 'rootA/workspace' starting at 'foo', note
      //  that even though the package 'foo' is under 'rootB/workspace', there is still a package
      //  'foo/bar' under 'rootA/workspace'.
    }

    DirectoryListingNode dirNode = (DirectoryListingNode)
        env.getDep(DirectoryListingNode.key(rootedPath));
    if (dirNode == null) {
      return null;
    }

    List<NodeKey> childDeps = Lists.newArrayList();
    for (Dirent dirent : dirNode.getDirents()) {
      String basename = dirent.getName();
      if (rootRelativePath.equals(PathFragment.EMPTY_FRAGMENT) &&
          PathPackageLocator.DEFAULT_TOP_LEVEL_EXCLUDES.contains(basename)) {
        continue;
      }
      NodeKey req = RecursivePkgNode.key(RootedPath.toRootedPath(root,
          rootRelativePath.getRelative(basename)));
      childDeps.add(req);
    }
    Map<NodeKey, Node> childNodeMap = env.getDeps(childDeps);
    if (env.depsMissing()) {
      return null;
    }
    // Aggregate the transitive subpackages.
    for (Node childNode : childNodeMap.values()) {
      if (childNode != null) {
        packages.addTransitive(((RecursivePkgNode) childNode).getPackages());
      }
    }
    return new RecursivePkgNode(packages.build());
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }
}
