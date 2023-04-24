// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.SymlinkForest;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetReadyForSymlinkPlanting;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * An implementation of PackageRoots that allows incremental updating of the packageRootsMap.
 *
 * <p>This class is also in charge of planting the necessary symlinks.
 */
public class IncrementalPackageRoots implements PackageRoots {

  // We only keep track of PackageIdentifier from external repos here as a memory optimization:
  // packages belong to the main repository all share the same root, which is singleSourceRoot.
  private final Map<PackageIdentifier, Root> threadSafeExternalRepoPackageRootsMap;

  @GuardedBy("stateLock")
  @Nullable
  private Set<NestedSet.Node> handledPackageNestedSets = Sets.newConcurrentHashSet();

  @Nullable private Set<Path> plantedExternalRepoLinks = Sets.newConcurrentHashSet();

  private final Object stateLock = new Object();
  private final Path execroot;
  private final Root singleSourceRoot;
  private final String prefix;
  private final boolean useSiblingRepositoryLayout;

  private final boolean allowExternalRepositories;
  @Nullable private EventBus eventBus;

  private IncrementalPackageRoots(
      Path execroot,
      Root singleSourceRoot,
      EventBus eventBus,
      String prefix,
      boolean useSiblingRepositoryLayout,
      boolean allowExternalRepositories) {
    this.threadSafeExternalRepoPackageRootsMap = Maps.newConcurrentMap();
    this.execroot = execroot;
    this.singleSourceRoot = singleSourceRoot;
    this.prefix = prefix;
    this.eventBus = eventBus;
    this.useSiblingRepositoryLayout = useSiblingRepositoryLayout;
    this.allowExternalRepositories = allowExternalRepositories;
  }

  public static IncrementalPackageRoots createAndRegisterToEventBus(
      Path execroot,
      Root singleSourceRoot,
      EventBus eventBus,
      String prefix,
      boolean useSiblingRepositoryLayout,
      boolean allowExternalRepositories) {
    IncrementalPackageRoots incrementalPackageRoots =
        new IncrementalPackageRoots(
            execroot,
            singleSourceRoot,
            eventBus,
            prefix,
            useSiblingRepositoryLayout,
            allowExternalRepositories);
    eventBus.register(incrementalPackageRoots);
    return incrementalPackageRoots;
  }

  /** Eagerly plant the symlinks to the directories under the single source root. */
  public void eagerlyPlantSymlinksToSingleSourceRoot() throws AbruptExitException {
    try {
      SymlinkForest.eagerlyPlantSymlinkForestSinglePackagePath(
          execroot, singleSourceRoot.asPath(), prefix, useSiblingRepositoryLayout);
    } catch (IOException e) {
      throwAbruptExitException(e);
    }
  }

  /** There is currently no use case for this method, and it should not be called. */
  @Override
  public Optional<ImmutableMap<PackageIdentifier, Root>> getPackageRootsMap() {
    throw new UnsupportedOperationException(
        "IncrementalPackageRoots does not provide the package roots map directly.");
  }

  @Override
  public PackageRootLookup getPackageRootLookup() {
    return packageId ->
        packageId.getRepository().isMain()
            ? singleSourceRoot
            : threadSafeExternalRepoPackageRootsMap.get(packageId);
  }

  @AllowConcurrentEvents
  @Subscribe
  public void topLevelTargetReadyForSymlinkPlanting(TopLevelTargetReadyForSymlinkPlanting event)
      throws AbruptExitException {
    if (allowExternalRepositories) {
      registerAndPlantSymlinksForExternalPackages(event.transitivePackagesForSymlinkPlanting());
    }
  }

  @Subscribe
  public void analysisFinished(AnalysisPhaseCompleteEvent unused) {
    dropIntermediateStatesAndUnregisterFromEventBus();
  }

  private void registerAndPlantSymlinksForExternalPackages(NestedSet<Package> packages)
      throws AbruptExitException {
    Set<Path> plantedExternalRepoLinksLocalRef;
    synchronized (stateLock) {
      if (handledPackageNestedSets == null || !handledPackageNestedSets.add(packages.toNode())) {
        return;
      }
      plantedExternalRepoLinksLocalRef = plantedExternalRepoLinks;
      if (plantedExternalRepoLinksLocalRef == null) {
        return;
      }
    }

    // To reach this point, this has to be the first and only time we plant the symlinks for this
    // NestedSet<Package>. That means it's not possible to reach this after analysis has ended.
    for (Package pkg : packages.getLeaves()) {
      PackageIdentifier pkgId = pkg.getPackageIdentifier();
      if (isExternalRepository(pkgId) && pkg.getSourceRoot().isPresent()) {
        threadSafeExternalRepoPackageRootsMap.put(
            pkg.getPackageIdentifier(), pkg.getSourceRoot().get());
        try {
          SymlinkForest.plantSingleSymlinkForExternalRepo(
              pkgId.getRepository(),
              pkg.getSourceRoot().get().asPath(),
              execroot,
              useSiblingRepositoryLayout,
              plantedExternalRepoLinksLocalRef);
        } catch (IOException e) {
          throwAbruptExitException(e);
        }
      }
    }
    for (NestedSet<Package> transitive : packages.getNonLeaves()) {
      registerAndPlantSymlinksForExternalPackages(transitive);
    }
  }

  private static void throwAbruptExitException(IOException e) throws AbruptExitException {
    throw new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage("Failed to prepare the symlink forest: " + e)
                .setSymlinkForest(
                    FailureDetails.SymlinkForest.newBuilder()
                        .setCode(FailureDetails.SymlinkForest.Code.CREATION_FAILED))
                .build()),
        e);
  }

  private static boolean isExternalRepository(PackageIdentifier pkgId) {
    return !pkgId.getRepository().isMain();
  }

  /**
   * Drops the intermediate states and stop receiving new events.
   *
   * <p>This essentially makes this instance read-only. Should be called when and only when all
   * analysis work is done in the build to free up some memory.
   */
  private void dropIntermediateStatesAndUnregisterFromEventBus() {
    // This instance is retained after a build via ArtifactFactory, so it's important that we remove
    // the reference to the eventBus here for it to be GC'ed.
    Preconditions.checkNotNull(eventBus).unregister(this);
    eventBus = null;

    synchronized (stateLock) {
      handledPackageNestedSets = null;
      plantedExternalRepoLinks = null;
    }
  }
}
