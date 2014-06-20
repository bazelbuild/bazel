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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * NodeBuilder for {@link PackageLookupNode}s.
 */
class PackageLookupNodeBuilder implements NodeBuilder {

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final AtomicReference<ImmutableSet<String>> deletedPackages;

  PackageLookupNodeBuilder(AtomicReference<PathPackageLocator> pkgLocator,
      AtomicReference<ImmutableSet<String>> deletedPackages) {
    this.pkgLocator = pkgLocator;
    this.deletedPackages = deletedPackages;
  }

  @Override
  public Node build(NodeKey nodeKey, Environment env) throws PackageLookupNodeBuilderException {
    PathFragment pkg = (PathFragment) nodeKey.getNodeName();

    // This represents a package lookup at the package root.
    if (pkg.equals(PathFragment.EMPTY_FRAGMENT)) {
      return PackageLookupNode.invalidPackageName("The empty package name is invalid");
    }

    String pkgName = pkg.getPathString();
    String packageNameErrorMsg = LabelValidator.validatePackageName(pkgName);
    if (packageNameErrorMsg != null) {
      return PackageLookupNode.invalidPackageName("Invalid package name '" + pkgName + "': "
          + packageNameErrorMsg);
    }

    if (deletedPackages.get().contains(pkg.getPathString())) {
      return PackageLookupNode.deletedPackage();
    }

    // TODO(bazel-team): The following is O(n^2) on the number of elements on the package path due
    // to having restart the NodeBuilder after every new dependency. However, if we try to batch
    // the missing node keys, more dependencies than necessary will be declared. This wart can be
    // fixed once we have nicer continuation support [skyframe-loading]
    for (Path packagePathEntry : pkgLocator.get().getPathEntries()) {
      RootedPath buildFileRootedPath = RootedPath.toRootedPath(packagePathEntry,
          pkg.getChild("BUILD"));
      NodeKey fileNodeKey = FileNode.key(buildFileRootedPath);
      FileNode fileNode = null;
      try {
        fileNode = (FileNode) env.getDepOrThrow(fileNodeKey, Exception.class);
      } catch (IOException e) {
        // TODO(bazel-team): throw an IOException here and let PackageNodeBuilder wrap that into a
        // BuildFileNotFoundException.
        throw new PackageLookupNodeBuilderException(nodeKey, new BuildFileNotFoundException(pkgName,
            "IO errors while looking for BUILD file reading " + buildFileRootedPath.asPath()
            + ": " + e.getMessage(), e));
      } catch (InconsistentFilesystemException e) {
        throw new PackageLookupNodeBuilderException(nodeKey, e);
      } catch (Exception e) {
        throw new IllegalStateException("Not IOException of InconsistentFilesystemException", e);
      }
      if (fileNode == null) {
        return null;
      }
      if (fileNode.isFile()) {
        // Result does not depend on package path entries beyond the first match.
        return PackageLookupNode.success(packagePathEntry);
      }
    }
    return PackageLookupNode.noBuildFile();
  }

  @Nullable
  @Override
  public String extractTag(NodeKey nodeKey) {
    return null;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link PackageLookupNodeBuilder#build}.
   */
  private static final class PackageLookupNodeBuilderException extends NodeBuilderException {
    public PackageLookupNodeBuilderException(NodeKey key, BuildFileNotFoundException e) {
      super(key, e);
    }

    public PackageLookupNodeBuilderException(NodeKey key, InconsistentFilesystemException e) {
      super(key, e, /*isTransient=*/true);
    }
  }
}
