// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.rewinding;

import com.google.common.base.Throwables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.skyframe.SkyFunction.Reset;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;

/**
 * Rewinding of external repository fetches to recover files that the remote repo contents cache has
 * lost, outside of action execution.
 *
 * <p>Reading a file of a cached repo can fail at any point at which Bazel fetches a repo, loads a
 * package or evaluates a module extension, none of which is an action that {@link
 * ActionRewindStrategy} could rewind. Since the repo rule that produced the file can simply be run
 * again, the failing node instead rewinds the repo's fetch and is then re-evaluated itself.
 */
public final class RepoRewinding {

  /**
   * Returns a {@link Reset} that rewinds the fetch of the repo whose lost file caused the given
   * failure, or null if the failure was not caused by a file that rewinding can recover.
   *
   * <p>The returned {@link Reset} must be returned from {@link
   * com.google.devtools.build.skyframe.SkyFunction#compute} of {@code failedKey}, which is
   * re-evaluated once the repo has been fetched again.
   */
  @Nullable
  public static Reset resetForLostRepoFile(SkyKey failedKey, Throwable failure) {
    return resetForLostRepoFile(failedKey, /* packageId= */ null, failure);
  }

  /**
   * Like {@link #resetForLostRepoFile(SkyKey, Throwable)}, but for a node that read the lost file
   * after looking up the given package, whose lookup is rewound in between.
   */
  @Nullable
  public static Reset resetForLostRepoFile(
      SkyKey failedKey, @Nullable PackageIdentifier packageId, Throwable failure) {
    LostRemoteRepoFileException lostFile = findLostRepoFile(failure);
    if (lostFile == null) {
      return null;
    }
    SkyKey repoDirKey = RepositoryDirectoryValue.key(lostFile.getRepo());
    var rewindGraph = Reset.newRewindGraphFor(failedKey);

    // The label the file was read through identifies the package lookup that resolved it; a node
    // that computes a package resolved its own.
    Label label = lostFile.getLabel();
    PackageIdentifier lookupPackage = label != null ? label.getPackageIdentifier() : packageId;
    if (lookupPackage == null) {
      // The file was read without a lookup, so the failing node depends on the repo fetch directly.
      rewindGraph.putEdge(failedKey, repoDirKey);
    } else if (lookupPackage.getRepository().equals(lostFile.getRepo())) {
      // Resolving the file only requested the package lookup, which caches the repo root and would
      // otherwise be handed out again without ever consulting the rewound repo fetch.
      SkyKey packageLookupKey = PackageLookupValue.key(lookupPackage);
      rewindGraph.putEdge(failedKey, packageLookupKey);
      rewindGraph.putEdge(packageLookupKey, repoDirKey);
    } else {
      // The file was read on behalf of a package in another repo, e.g. a BUILD file loading a .bzl
      // file from the lost repo. The nodes in between are not known here, so rewinding the fetch
      // would dirty a node that is never rebuilt.
      return null;
    }
    return Reset.of(rewindGraph);
  }

  /**
   * Returns whether the given failure was caused by a file that rewinding a repo fetch can recover.
   *
   * <p>Such a failure must not be reported to the user: the build recovers from it.
   */
  public static boolean isLostRepoFile(Throwable failure) {
    return findLostRepoFile(failure) != null;
  }

  /** Returns the lost repo file that caused the given failure, or null if there is none. */
  @Nullable
  public static LostRemoteRepoFileException findLostRepoFile(Throwable failure) {
    for (Throwable cause : Throwables.getCausalChain(failure)) {
      if (cause instanceof LostRemoteRepoFileException lostFile) {
        return lostFile;
      }
    }
    return null;
  }

  private RepoRewinding() {}
}
