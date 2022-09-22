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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.PackageRoots;
import com.google.devtools.build.lib.buildtool.SymlinkForest;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.AspectAnalyzedEvent;
import com.google.devtools.build.lib.skyframe.TopLevelStatusEvents.TopLevelTargetAnalyzedEvent;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import java.io.IOException;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * An implementation of PackageRoots that allows incremental updating of the packageRootsMap.
 *
 * <p>This class is also in charge of planting the necessary symlinks.
 */
public class IncrementalPackageRoots implements PackageRoots {

  // We only keep track of PackageIdentifier from external repos here as a memory optimization:
  // packages belong to the main repository all share the same root, which is singleSourceRoot.
  private final Map<PackageIdentifier, Root> threadSafeExternalRepoPackageRootsMap;
  // Top level events originate from within Skyframe, so duplications are expected.
  private final Set<Object> handledTopLevelEvents = Sets.newConcurrentHashSet();
  private final Set<Path> plantedExternalRepoLinks = Sets.newConcurrentHashSet();
  private final Path execroot;
  private final Root singleSourceRoot;
  private final EventBus eventBus;
  private final String prefix;
  private final boolean useSiblingRepositoryLayout;

  private IncrementalPackageRoots(
      Path execroot,
      Root singleSourceRoot,
      EventBus eventBus,
      String prefix,
      boolean useSiblingRepositoryLayout) {
    this.threadSafeExternalRepoPackageRootsMap = Maps.newConcurrentMap();
    this.execroot = execroot;
    this.singleSourceRoot = singleSourceRoot;
    this.prefix = prefix;
    this.eventBus = eventBus;
    this.useSiblingRepositoryLayout = useSiblingRepositoryLayout;
  }

  public static IncrementalPackageRoots createAndRegisterToEventBus(
      Path execroot,
      Root singleSourceRoot,
      EventBus eventBus,
      String prefix,
      boolean useSiblingRepositoryLayout) {
    IncrementalPackageRoots incrementalPackageRoots =
        new IncrementalPackageRoots(
            execroot, singleSourceRoot, eventBus, prefix, useSiblingRepositoryLayout);
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
  public void addTargetPackage(TopLevelTargetAnalyzedEvent event) throws AbruptExitException {
    if (handledTopLevelEvents.add(event)) {
      registerAndPlantSymlinksForExternalPackages(
          event.transitivePackagesForSymlinkPlanting().toList());
    }
  }

  @AllowConcurrentEvents
  @Subscribe
  public void addAspectPackage(AspectAnalyzedEvent event) throws AbruptExitException {
    if (handledTopLevelEvents.add(event)) {
      registerAndPlantSymlinksForExternalPackages(
          event.transitivePackagesForSymlinkPlanting().toList());
    }
  }

  private void registerAndPlantSymlinksForExternalPackages(Iterable<Package> packages)
      throws AbruptExitException {
    for (Package pkg : packages) {
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
              plantedExternalRepoLinks);
        } catch (IOException e) {
          throwAbruptExitException(e);
        }
      }
    }
  }

  private static void throwAbruptExitException(IOException e) throws AbruptExitException {
    throw new AbruptExitException(
        DetailedExitCode.of(
            FailureDetail.newBuilder()
                .setMessage("Failed to prepare the symlink forest")
                .setSymlinkForest(
                    FailureDetails.SymlinkForest.newBuilder()
                        .setCode(FailureDetails.SymlinkForest.Code.CREATION_FAILED))
                .build()),
        e);
  }

  private static boolean isExternalRepository(PackageIdentifier pkgId) {
    return !pkgId.getRepository().isMain();
  }

  public void shutdown() {
    eventBus.unregister(this);
  }
}
