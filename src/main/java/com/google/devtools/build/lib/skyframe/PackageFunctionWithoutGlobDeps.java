// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.ThreadStateReceiver;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.NonSkyframeGlobber;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Computes the {@link PackageValue} without performing Skyframe globbing.
 *
 * <p>{@link PackageFunctionWithoutGlobDeps} subclass is created when the globbing strategy is
 * {@link com.google.devtools.build.lib.skyframe.PackageFunction.GlobbingStrategy#NON_SKYFRAME},
 * which is used for non-incremental evaluations with no GLOB nodes queried and stored in Skyframe.
 */
final class PackageFunctionWithoutGlobDeps extends PackageFunction {

  PackageFunctionWithoutGlobDeps(
      PackageFactory packageFactory,
      CachingPackageLocator pkgLocator,
      AtomicBoolean showLoadingProgress,
      AtomicInteger numPackagesSuccessfullyLoaded,
      @Nullable BzlLoadFunction bzlLoadFunctionForInlining,
      @Nullable PackageProgressReceiver packageProgress,
      ActionOnIOExceptionReadingBuildFile actionOnIOExceptionReadingBuildFile,
      boolean shouldUseRepoDotBazel,
      Function<SkyKey, ThreadStateReceiver> threadStateReceiverFactoryForMetrics,
      AtomicReference<Semaphore> cpuBoundSemaphore) {
    super(
        packageFactory,
        pkgLocator,
        showLoadingProgress,
        numPackagesSuccessfullyLoaded,
        bzlLoadFunctionForInlining,
        packageProgress,
        actionOnIOExceptionReadingBuildFile,
        shouldUseRepoDotBazel,
        threadStateReceiverFactoryForMetrics,
        cpuBoundSemaphore);
  }

  private static final class LoadedPackageWithoutDeps extends LoadedPackage {
    LoadedPackageWithoutDeps(Package.Builder builder, long loadTimeNanos) {
      super(builder, loadTimeNanos);
    }
  }

  @Override
  protected void handleGlobDepsAndPropagateFilesystemExceptions(
      PackageIdentifier packageIdentifier,
      LoadedPackage loadedPackage,
      Environment env,
      boolean packageWasInError) {
    // No-op for non-Skyframe globbing.
  }

  @Override
  protected Globber makeGlobber(
      NonSkyframeGlobber nonSkyframeGlobber,
      PackageIdentifier packageId,
      Root packageRoot,
      Environment env) {
    return nonSkyframeGlobber;
  }

  @Override
  protected LoadedPackage newLoadedPackage(
      Package.Builder packageBuilder, @Nullable Globber globber, long loadTimeNanos) {
    return new LoadedPackageWithoutDeps(packageBuilder, loadTimeNanos);
  }
}
