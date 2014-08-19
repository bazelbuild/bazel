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
package com.google.devtools.build.lib.packages;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A {@link Package} implementation that manages its own filesystem dependencies.
 */
public class LegacyPackage extends Package implements Serializable {

  /**
   * We store the glob patterns directly for now.
   */
  @Nullable
  private Collection<Pair<String, Boolean>> globPatterns;

  private LegacyPackage(String name) {
    super(name);
  }

  private void writeObject(ObjectOutputStream out) {
    // Nothing to do here. Java serialization is only used by skyframe and so we don't need to
    // worry about legacy data.
  }

  private void readObject(ObjectInputStream in) {
    // Nothing to do here. Java serialization is only used by skyframe and so we don't need to
    // worry about legacy data.
  }

  @Override
  protected Object readResolve() {
    // Even though LegacyPackage doesn't care about serialization, we still need to return the same
    // reference as Package#readResolve does, otherwise deserialization won't work (e.g. if we
    // omitted this method and Package#readResolve were private, then we'd get the default behavior
    // of 'return this', which would screw up the Package deserialization).
    return super.readResolve();
  }

  public void dropLegacyData() {
    this.globPatterns = null;
  }

  /**
   * Returns an (unordered, immutable) collection of all the glob patterns that
   * were evaluated (relative to the package directory) during the construction of
   * this package.
   */
  @VisibleForTesting
  public Collection<Pair<String, Boolean>> getGlobPatterns() {
    return globPatterns;
  }

  private static Set<PathFragment> checkLabelsCrossingSubpackages(
      Path buildFilePath, Map<Label, Location> labels,
      BulkPackageLocatorForCrossingSubpackageBoundaries locator,
      @Nullable EventHandler eventHandler)
          throws InterruptedException {
    if (labels.isEmpty()) {
      return ImmutableSet.of();
    }

    Set<PathFragment> subpackagesToCheck = getPossiblyAffectingPackages(labels.keySet());
    PathFragment packageFragment =
        Iterables.getFirst(labels.keySet(), null).getPackageFragment();

    Map<PathFragment, Path> existingPackages =
        locator.getExistingPackages(subpackagesToCheck);
    if (existingPackages.isEmpty()) {
      // Everything ok
      return ImmutableSet.of();
    }

    // This is slow, but we only get here when the package is in error
    for (Map.Entry<PathFragment, Path> inner : existingPackages.entrySet()) {
      for (Label label : labels.keySet()) {
        if (label.toPathFragment().getParentDirectory().startsWith(
            inner.getKey())) {
          String message = String.format("Label '%s' crosses boundary of subpackage '%s'",
              label, inner.getKey());

          // Both packages come from the same package-path root iff the outer
          // package path is a prefix of the inner:
          if (inner.getValue().startsWith(buildFilePath.getParentDirectory())) {
            // Same root
            PathFragment targetFragment = new PathFragment(label.getName());
            PathFragment innerFragment = targetFragment.subFragment(
                inner.getKey().segmentCount() - packageFragment.segmentCount(),
                targetFragment.segmentCount());
            message += " (perhaps you meant to put the colon here: "
                + "'//" + inner.getKey() + ":" + innerFragment + "'?)";
          } else { // different roots:
            message += " (have you deleted " + inner.getKey() + "/BUILD? "
                + "If so, use the --deleted_packages=" + inner.getKey() + " option)";
          }

          if (eventHandler != null) {
            eventHandler.handle(Event.error(labels.get(label), message));
          }

          break;
        }
      }
    }

    return ImmutableSet.copyOf(existingPackages.keySet());
  }

  private static Set<PathFragment> getPossiblyAffectingPackages(Iterable<Label> labels) {
    ImmutableSet.Builder<PathFragment> result = ImmutableSet.builder();

    for (Label label : labels) {
      PathFragment packageFragment = label.getPackageFragment();
      PathFragment targetPath = label.toPathFragment();
      // We can have targets with a name like '.', so we need to check this before blindly taking
      // the parent directory of the target.
      if (targetPath.equals(packageFragment)) {
        continue;
      }

      Preconditions.checkState(targetPath.startsWith(packageFragment));
      PathFragment candidate = targetPath.getParentDirectory();
      while (!candidate.equals(packageFragment)) {
        result.add(candidate);
        candidate = candidate.getParentDirectory();
        continue;
      }
    }

    return result.build();
  }

  private void finishInit(LegacyPackageBuilder builder) {
    this.globPatterns = builder.globCache.getKeySet();
  }

  /**
   * Builder for a package instance.
   *
   * <p>Package initialization is a bit of a circuitous process. Part of it
   * involves constructing member values that expect a valid reference to
   * the package instance that correctly fulfills {@link Package#getNameFragment()}.
   *
   * <p>So initialization is a multi-step process. First, the output package is
   * instantiated with its package name. Then, settings are applied and appropriate
   * mutations are made to these settings according to the logic in {@link PackageFactory}.
   * Finally, the package instance is finalized with the completed settings.
   *
   * <p>Once this process is done, the package is considered fully initialized and no
   * more mutations can be applied to it.
   */
  public static class LegacyPackageBuilder
      extends Package.AbstractPackageBuilder<LegacyPackage, LegacyPackageBuilder> {

    private GlobCache globCache = null;
    private Set<Label> targetsCrossingSubpackages = new HashSet<>();
    private Set<PathFragment> subpackagesCuttingOffLabels = new HashSet<>();

    // Set by #build and used by #beforeBuildInternal.
    private BulkPackageLocatorForCrossingSubpackageBoundaries bulkPackageLocator = null;
    private StoredEventHandler eventHandler = null;

    LegacyPackageBuilder(String packageName) {
      super(new LegacyPackage(packageName));
    }

    @Override
    protected LegacyPackageBuilder self() {
      return this;
    }

    /**
     * Sets the cache to use for this package's glob expansions.
     */
    LegacyPackageBuilder setGlobCache(GlobCache globCache) {
      this.globCache = globCache;
      return this;
    }

    /**
     * Evaluate the build language expression "glob(includes, excludes)" in the
     * context of this package.
     */
    List<String> glob(List<String> includes, List<String> excludes, boolean excludeDirs)
        throws IOException, GlobCache.BadGlobException, InterruptedException {
      if (globCache == null) {
        throw new NullPointerException("globCache is null");
      }

      return globCache.glob(includes, excludes, excludeDirs);
    }

    /**
     * Launches the given glob expressions, but does not block on their completion.
     */
    void globAsync(List<String> includes, List<String> excludes, boolean excludeDirs)
        throws GlobCache.BadGlobException {
      for (String pattern :  Iterables.concat(includes, excludes)) {
        globCache.getGlobAsync(pattern, excludeDirs);
      }
    }

    private void removeTargetsCrossingSubpackages(Set<PathFragment> subpackages) {
      Iterator<Map.Entry<String, Target>> iterator = targets.entrySet().iterator();
      while (iterator.hasNext()) {
        Map.Entry<String, Target> entry = iterator.next();
        boolean ok = true;
        PathFragment targetFragment =
            entry.getValue().getLabel().toPathFragment().getParentDirectory();
        for (PathFragment subpackage : subpackages) {
          if (targetFragment.startsWith(subpackage)) {
            ok = false;
            break;
          }
        }

        if (!ok) {
          iterator.remove();
          targetsCrossingSubpackages.add(entry.getValue().getLabel());
        }
      }
    }


    private static class InterruptedExceptionDuringBeforeBuildInternal extends RuntimeException {

      private InterruptedException e;

      public InterruptedExceptionDuringBeforeBuildInternal(InterruptedException e) {
        super(e.getMessage());
        this.e = e;
      }

      public InterruptedException getInterruptedException() {
        return e;
      }
    }

    @Override
    protected void beforeBuildInternal() {
      super.beforeBuildInternal();
      Map<Label, Location> labels = new HashMap<>();
      for (Target target : targets.values()) {
        labels.put(target.getLabel(), target.getLocation());
      }
      Map<Label, Location> subincludeLabels = Maps.newHashMap();
      for (Label subincludeLabel : subincludes.keySet()) {
        subincludeLabels.put(subincludeLabel, null);
      }
      try {
        subpackagesCuttingOffLabels.addAll(checkLabelsCrossingSubpackages(
            getFilename(), labels, bulkPackageLocator, eventHandler));
        subpackagesCuttingOffLabels.addAll(checkLabelsCrossingSubpackages(
            getFilename(), subincludeLabels, bulkPackageLocator, eventHandler));
      } catch (InterruptedException e) {
        throw new InterruptedExceptionDuringBeforeBuildInternal(e);
      }

      if (!subpackagesCuttingOffLabels.isEmpty()) {
        setContainsErrors();
        removeTargetsCrossingSubpackages(subpackagesCuttingOffLabels);
      }
    }

    /**
     * Builds and returns the {@link Package} instance from the builder's
     * Can only be called once per PackageBuilder instance.
     */
    @Override
    protected LegacyPackage buildInternal(StoredEventHandler eventHandler) {
      LegacyPackage pkg = super.buildInternal(eventHandler);
      pkg.finishInit(this);
      return pkg;
    }

    protected LegacyPackage build(
        BulkPackageLocatorForCrossingSubpackageBoundaries bulkPackageLocator,
        StoredEventHandler eventHandler)
            throws InterruptedException {
      this.bulkPackageLocator = bulkPackageLocator;
      this.eventHandler = eventHandler;
      try {
        return super.build(eventHandler);
      } catch (InterruptedExceptionDuringBeforeBuildInternal e) {
        throw e.getInterruptedException();
      }
    }
  }

  /**
   * A stub implementation of {@link BulkPackageLocatorForCrossingSubpackageBoundaries}
   * that always returns the empty set.
   */
  public static final BulkPackageLocatorForCrossingSubpackageBoundaries EMPTY_BULK_PACKAGE_LOCATOR =
      new BulkPackageLocatorForCrossingSubpackageBoundaries() {
        @Override
        public Map<PathFragment, Path> getExistingPackages(Set<PathFragment> candidates) {
          return ImmutableMap.of();
        }
      };
}
