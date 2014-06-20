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

import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeBuilder;
import com.google.devtools.build.skyframe.NodeBuilderException;
import com.google.devtools.build.skyframe.NodeKey;

/**
 * A NodeBuilder for {@link TargetMarkerNode}s.
 */
public final class TargetMarkerNodeBuilder implements NodeBuilder {

  public TargetMarkerNodeBuilder() {
  }

  @Override
  public Node build(NodeKey key, Environment env) throws TargetMarkerNodeBuilderException {
    Label label = (Label) key.getNodeName();
    PathFragment pkgForLabel = label.getPackageFragment();

    if (label.getName().contains("/")) {
      // This target is in a subdirectory, therefore it could potentially be invalidated by
      // a new BUILD file appearing in the hierarchy.
      PathFragment containingDirectory = label.toPathFragment().getParentDirectory();
      ContainingPackageLookupNode containingPackageLookupNode = null;
      try {
        containingPackageLookupNode = (ContainingPackageLookupNode) env.getDepOrThrow(
            ContainingPackageLookupNode.key(containingDirectory),
            BuildFileNotFoundException.class);
      } catch (BuildFileNotFoundException e) {
        // Thrown when there are IO errors looking for BUILD files.
        throw new TargetMarkerNodeBuilderException(key, e);
      }
      if (containingPackageLookupNode == null) {
        return null;
      }
      PathFragment containingPkg = containingPackageLookupNode.getContainingPackageNameOrNull();
      if (containingPkg == null) {
        // This means the label's package doesn't exist. E.g. there is no package 'a/b' and we are
        // trying to build the target for label 'a/b:foo'.
        throw new TargetMarkerNodeBuilderException(key,
            new BuildFileNotFoundException(pkgForLabel.getPathString(),
                "BUILD file not found on package path for '" + pkgForLabel.getPathString() + "'"));
      }
      if (!containingPkg.equals(pkgForLabel)) {
        throw new TargetMarkerNodeBuilderException(key, new NoSuchTargetException(label,
            String.format("Label '%s' crosses boundary of subpackage '%s'", label,
                containingPkg)));
      }
    }

    NodeKey pkgNodeKey = PackageNode.key(pkgForLabel);
    NoSuchPackageException nspe = null;
    Package pkg;
    try {
      PackageNode node = (PackageNode) env.getDepOrThrow(pkgNodeKey, NoSuchPackageException.class);
      if (node == null) {
        return null;
      }
      pkg = node.getPackage();
    } catch (NoSuchPackageException e) {
      // For consistency with pre-Skyframe Blaze, we can return a valid Target from a Package
      // containing errors.
      pkg = e.getPackage();
      if (pkg == null) {
        // Re-throw this exception with our key because root causes should be targets, not packages.
        throw new TargetMarkerNodeBuilderException(key, e);
      }
      nspe = e;
    }

    Target target;
    try {
      target = pkg.getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      throw new TargetMarkerNodeBuilderException(key, e);
    }

    if (nspe != null) {
      // There is a target, but its package is in error. We rethrow so that the root cause is the
      // target, not the package. Note that targets are only in error when their package is
      // "in error" (because a package is in error if there was an error evaluating the package, or
      // if one of its targets was in error).
      throw new TargetMarkerNodeBuilderException(key, new NoSuchTargetException(target, nspe));
    }
    return TargetMarkerNode.TARGET_MARKER_INSTANCE;
  }

  @Override
  public String extractTag(NodeKey nodeKey) {
    return Label.print((Label) nodeKey.getNodeName());
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by
   * {@link TransitiveTargetNodeBuilder#build}.
   */
  private static final class TargetMarkerNodeBuilderException extends NodeBuilderException {
    public TargetMarkerNodeBuilderException(NodeKey key, NoSuchTargetException e) {
      super(key, e);
    }

    public TargetMarkerNodeBuilderException(NodeKey key, NoSuchPackageException e) {
      super(key, e);
    }
  }
}
