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

package com.google.devtools.build.lib.view.fileset;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.util.Fingerprint;

import java.io.IOException;
import java.util.Collection;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * A {@link SymlinkTraversal} which is the composition of other
 * traversals.
 *
 * This class is intentionally package-private.
 */
final class CompoundSymlinkTraversal implements SymlinkTraversal {

  private final Collection<SymlinkTraversal> traversals;

  /**
   * Construct the compound traversal from the given collection.
   * @param traversals a non-null collection of traversals.
   */
  public CompoundSymlinkTraversal(Collection<SymlinkTraversal> traversals) {
    // Use a list to maintain order.
    this.traversals = ImmutableList.copyOf(traversals);
  }

  @Override
  public void addSymlinks(ErrorEventListener listener, FilesetLinks links,
      ThreadPoolExecutor filesetPool) throws IOException, InterruptedException {
    // TODO(bazel-team): Consider paralellizing this for loop.
    // Each traversal could be a recursive file traversal over
    // a source tree. Beware that order does matter here.
    // If there is a symlink conflict between two traversals, the
    // first must win. One implementation might be to do the
    // traversal in parallel, but use a temporary FilesetLinks
    // instance for each traversal.
    for (SymlinkTraversal traversal : traversals) {
      traversal.addSymlinks(listener, links, filesetPool);
    }
  }

  @Override
  public void fingerprint(Fingerprint fp) {
    for (SymlinkTraversal traversal : traversals) {
      traversal.fingerprint(fp);
    }
  }

  @Override
  public boolean executeUnconditionally(PackageUpToDateChecker upToDateChecker) {
    // Note: isVolatile must return true if executeUnconditionally can ever return true
    // for this instance.
    for (SymlinkTraversal traversal : traversals) {
      if (traversal.executeUnconditionally(upToDateChecker)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean isVolatile() {
    for (SymlinkTraversal traversal : traversals) {
      if (traversal.isVolatile()) {
        return true;
      }
    }
    return false;
  }
}
