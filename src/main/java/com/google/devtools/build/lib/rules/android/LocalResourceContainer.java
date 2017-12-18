// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.android.resources.ResourceFolderType;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.android.ResourceContainer.ResourceType;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * The collected resources and assets artifacts and roots.
 *
 * <p>This is used to encapsulate the logic and the data associated with the resources and assets
 * derived from an appropriate android rule in a reusable instance.
 */
@Immutable
public final class LocalResourceContainer {
  public static final String[] RESOURCES_ATTRIBUTES =
      new String[] {
        "manifest",
        "resource_files",
        "local_resource_files",
        "assets",
        "assets_dir",
        "inline_constants",
        "exports_manifest"
      };

  /** Set of allowable android directories prefixes. */
  public static final ImmutableSet<String> RESOURCE_DIRECTORY_TYPES =
      Arrays.stream(ResourceFolderType.values())
          .map(ResourceFolderType::getName)
          .collect(ImmutableSet.toImmutableSet());

  public static final String INCORRECT_RESOURCE_LAYOUT_MESSAGE =
      String.format(
          "'%%s' is not in the expected resource directory structure of "
              + "<resource directory>/{%s}/<file>",
          Joiner.on(',').join(RESOURCE_DIRECTORY_TYPES));

  /** Determines if the attributes contain resource and asset attributes. */
  public static boolean definesAndroidResources(AttributeMap attributes) {
    for (String attribute : RESOURCES_ATTRIBUTES) {
      if (attributes.isAttributeValueExplicitlySpecified(attribute)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Checks validity of a RuleContext to produce an AndroidData.
   *
   * @throws RuleErrorException if the RuleContext is invalid. Accumulated errors will be available
   *     via {@code ruleContext}
   */
  public static void validateRuleContext(RuleContext ruleContext) throws RuleErrorException {
    validateAssetsAndAssetsDir(ruleContext);
    validateNoResourcesAttribute(ruleContext);
    validateNoAndroidResourcesInSources(ruleContext);
    validateManifest(ruleContext);
  }

  private static void validateAssetsAndAssetsDir(RuleContext ruleContext)
      throws RuleErrorException {
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("assets")
        ^ ruleContext.attributes().isAttributeValueExplicitlySpecified("assets_dir")) {
      ruleContext.throwWithRuleError(
          "'assets' and 'assets_dir' should be either both empty or both non-empty");
    }
  }

  /** Validates that there are no resources defined if there are resource attributes defined. */
  private static void validateNoResourcesAttribute(RuleContext ruleContext)
      throws RuleErrorException {
    if (ruleContext.attributes().isAttributeValueExplicitlySpecified("resources")) {
      ruleContext.throwWithAttributeError(
          "resources",
          String.format(
              "resources cannot be set when any of %s are defined.",
              Joiner.on(", ").join(RESOURCES_ATTRIBUTES)));
    }
  }

  /**
   * Validates that there are no android_resources srcjars in the srcs, as android_resource rules
   * should not be used with the Android data logic.
   */
  private static void validateNoAndroidResourcesInSources(RuleContext ruleContext)
      throws RuleErrorException {
    Iterable<AndroidResourcesProvider> resources =
        ruleContext.getPrerequisites("srcs", Mode.TARGET, AndroidResourcesProvider.class);
    for (AndroidResourcesProvider provider : resources) {
      ruleContext.throwWithAttributeError(
          "srcs",
          String.format("srcs should not contain android_resource label %s", provider.getLabel()));
    }
  }

  private static void validateManifest(RuleContext ruleContext) throws RuleErrorException {
    if (ruleContext.getPrerequisiteArtifact("manifest", Mode.TARGET) == null) {
      ruleContext.throwWithAttributeError(
          "manifest", "manifest is required when resource_files or assets are defined.");
    }
  }

  /**
   * Creates a {@link LocalResourceContainer} containing this target's assets and resources.
   *
   * @param ruleContext the current context
   * @param assetsAttr the attribute used to refer to assets in this target
   * @param assetsDir the path to the assets directory
   * @param resourcesAttr the attribute used to refer to resource files in this target
   */
  public static LocalResourceContainer forAssetsAndResources(
      RuleContext ruleContext, String assetsAttr, PathFragment assetsDir, String resourcesAttr)
      throws RuleErrorException {
    ImmutableList.Builder<Artifact> assets = ImmutableList.builder();
    ImmutableList.Builder<PathFragment> assetRoots = ImmutableList.builder();

    for (TransitiveInfoCollection target :
        ruleContext.getPrerequisitesIf(assetsAttr, Mode.TARGET, FileProvider.class)) {
      for (Artifact file : target.getProvider(FileProvider.class).getFilesToBuild()) {
        PathFragment packageFragment =
            file.getArtifactOwner().getLabel().getPackageIdentifier().getSourceRoot();
        PathFragment packageRelativePath = file.getRootRelativePath().relativeTo(packageFragment);
        if (packageRelativePath.startsWith(assetsDir)) {
          PathFragment relativePath = packageRelativePath.relativeTo(assetsDir);
          assetRoots.add(trimTail(file.getExecPath(), relativePath));
        } else {
          ruleContext.attributeError(
              ResourceType.ASSETS.getAttribute(),
              String.format(
                  "'%s' (generated by '%s') is not beneath '%s'",
                  file.getRootRelativePath(), target.getLabel(), assetsDir));
          throw new RuleErrorException();
        }
        assets.add(file);
      }
    }

    ImmutableList<Artifact> resources =
        getResources(ruleContext.getPrerequisites(resourcesAttr, Mode.TARGET, FileProvider.class));

    return new LocalResourceContainer(
        resources,
        getResourceRoots(ruleContext, resources, resourcesAttr),
        assets.build(),
        assetRoots.build());
  }

  /**
   * Creates a {@link LocalResourceContainer} containing the resources included in a {@link
   * FileProvider}.
   *
   * <p>In general, {@link #forAssetsAndResources(RuleContext, String, PathFragment, String)} should
   * be used instead. No assets or transitive resources will be included in the container produced
   * by this method.
   *
   * @param ruleContext the current context
   * @param resourceFileProvider the provider containing resources
   * @param resourcesAttr the attribute used to refer to resource files in this target.
   */
  public static LocalResourceContainer forResourceFileProvider(
      RuleContext ruleContext, FileProvider resourceFileProvider, String resourcesAttr)
      throws RuleErrorException {
    ImmutableList<Artifact> resources = getResources(ImmutableList.of(resourceFileProvider));

    return new LocalResourceContainer(
        resources,
        getResourceRoots(ruleContext, resources, resourcesAttr),
        ImmutableList.of(),
        ImmutableList.of());
  }

  private static ImmutableList<Artifact> getResources(Iterable<FileProvider> targets) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (FileProvider target : targets) {
      builder.addAll(target.getFilesToBuild());
    }

    return builder.build();
  }

  /**
   * Gets the roots of some resources.
   *
   * @return a list of roots, or an empty list of the passed resources cannot all be contained in a
   *     single {@link LocalResourceContainer}. If that's the case, it will be reported to the
   *     {@link RuleErrorConsumer}.
   */
  @VisibleForTesting
  static ImmutableList<PathFragment> getResourceRoots(
      RuleErrorConsumer ruleErrorConsumer, Iterable<Artifact> files, String resourcesAttr)
      throws RuleErrorException {
    Artifact lastFile = null;
    PathFragment lastResourceDir = null;
    Set<PathFragment> resourceRoots = new LinkedHashSet<>();
    for (Artifact file : files) {
      PathFragment resourceDir =
          addResourceDir(
              file, lastFile, lastResourceDir, resourceRoots, resourcesAttr, ruleErrorConsumer);
      lastFile = file;
      lastResourceDir = resourceDir;
    }

    return ImmutableList.copyOf(resourceRoots);
  }

  /**
   * Inner method for adding resource roots to a collection. May fail and report to the {@link
   * RuleErrorConsumer} if the input is invalid.
   *
   * @param file the file to add the resource directory for
   * @param lastFile the last file this method was called on. May be null if this is the first call
   *     for this set of resources.
   * @param lastResourceDir the resource directory of the last file, as returned by the most recent
   *     call to this method, or null if this is the first call.
   * @param resourceRoots the collection to add resources to
   * @param resourcesAttr the attribute used to refer to resources. While we're moving towards
   *     "resource_files" everywhere, there are still uses of other attributes for different kinds
   *     of rules.
   * @param ruleErrorConsumer for reporting errors
   * @return the resource root of {@code file}.
   * @throws RuleErrorException if the current resource has no resource directory or if it is
   *     incompatible with {@code lastResourceDir}. An error will also be reported to the {@link
   *     RuleErrorConsumer} in this case.
   */
  private static PathFragment addResourceDir(
      Artifact file,
      @Nullable Artifact lastFile,
      @Nullable PathFragment lastResourceDir,
      Set<PathFragment> resourceRoots,
      String resourcesAttr,
      RuleErrorConsumer ruleErrorConsumer)
      throws RuleErrorException {
    PathFragment resourceDir = findResourceDir(file);
    if (resourceDir == null) {
      ruleErrorConsumer.attributeError(
          resourcesAttr,
          String.format(INCORRECT_RESOURCE_LAYOUT_MESSAGE, file.getRootRelativePath()));
      throw new RuleErrorException();
    }

    if (lastResourceDir != null && !resourceDir.equals(lastResourceDir)) {
      ruleErrorConsumer.attributeError(
          resourcesAttr,
          String.format(
              "'%s' (generated by '%s') is not in the same directory '%s' (derived from %s)."
                  + " All resources must share a common directory.",
              file.getRootRelativePath(),
              file.getArtifactOwner().getLabel(),
              lastResourceDir,
              lastFile.getRootRelativePath()));
      throw new RuleErrorException();
    }

    PathFragment packageFragment =
        file.getArtifactOwner().getLabel().getPackageIdentifier().getSourceRoot();
    PathFragment packageRelativePath = file.getRootRelativePath().relativeTo(packageFragment);
    try {
      resourceRoots.add(
          trimTail(file.getExecPath(), makeRelativeTo(resourceDir, packageRelativePath)));
    } catch (IllegalArgumentException e) {
      ruleErrorConsumer.attributeError(
          resourcesAttr,
          String.format(
              "'%s' (generated by '%s') is not under the directory '%s' (derived from %s).",
              file.getRootRelativePath(),
              file.getArtifactOwner().getLabel(),
              packageRelativePath,
              file.getRootRelativePath()));
      throw new RuleErrorException();
    }
    return resourceDir;
  }

  /**
   * Finds and validates the resource directory PathFragment from the artifact Path.
   *
   * <p>If the artifact is not a Fileset, the resource directory is presumed to be the second
   * directory from the end. Filesets are expect to have the last directory as the resource
   * directory.
   */
  public static PathFragment findResourceDir(Artifact artifact) {
    PathFragment fragment = artifact.getPath().asFragment();
    int segmentCount = fragment.segmentCount();
    if (segmentCount < 3) {
      return null;
    }
    // TODO(bazel-team): Expand Fileset to verify, or remove Fileset as an option for resources.
    if (artifact.isFileset() || artifact.isTreeArtifact()) {
      return fragment.subFragment(segmentCount - 1, segmentCount);
    }

    // Check the resource folder type layout.
    // get the prefix of the parent folder of the fragment.
    String parentDirectory = fragment.getSegment(segmentCount - 2);
    int dashIndex = parentDirectory.indexOf('-');
    String androidFolder =
        dashIndex == -1 ? parentDirectory : parentDirectory.substring(0, dashIndex);
    if (!RESOURCE_DIRECTORY_TYPES.contains(androidFolder)) {
      return null;
    }

    return fragment.subFragment(segmentCount - 3, segmentCount - 2);
  }

  /**
   * Returns the root-part of a given path by trimming off the end specified by a given tail.
   * Assumes that the tail is known to match, and simply relies on the segment lengths.
   */
  private static PathFragment trimTail(PathFragment path, PathFragment tail) {
    return path.subFragment(0, path.segmentCount() - tail.segmentCount());
  }

  private static PathFragment makeRelativeTo(PathFragment ancestor, PathFragment path) {
    String cutAtSegment = ancestor.getSegment(ancestor.segmentCount() - 1);
    int totalPathSegments = path.segmentCount() - 1;
    for (int i = totalPathSegments; i >= 0; i--) {
      if (path.getSegment(i).equals(cutAtSegment)) {
        return path.subFragment(i, totalPathSegments);
      }
    }
    throw new IllegalArgumentException("PathFragment " + path + " is not beneath " + ancestor);
  }

  private final ImmutableList<Artifact> resources;
  private final ImmutableList<Artifact> assets;
  private final ImmutableList<PathFragment> assetRoots;
  private final ImmutableList<PathFragment> resourceRoots;

  @VisibleForTesting
  public LocalResourceContainer(
      ImmutableList<Artifact> resources,
      ImmutableList<PathFragment> resourceRoots,
      ImmutableList<Artifact> assets,
      ImmutableList<PathFragment> assetRoots) {
    this.resources = resources;
    this.resourceRoots = resourceRoots;
    this.assets = assets;
    this.assetRoots = assetRoots;
  }

  public ImmutableList<Artifact> getResources() {
    return resources;
  }

  public ImmutableList<Artifact> getAssets() {
    return assets;
  }

  public ImmutableList<PathFragment> getAssetRoots() {
    return assetRoots;
  }

  public ImmutableList<PathFragment> getResourceRoots() {
    return resourceRoots;
  }

  /**
   * Filters this object.
   *
   * @return a new {@link LocalResourceContainer} with resources filtered by the passed {@link
   *     ResourceFilter}, or this object if no resources should be filtered.
   */
  public LocalResourceContainer filter(
      RuleErrorConsumer ruleErrorConsumer, ResourceFilter resourceFilter)
      throws RuleErrorException {
    Optional<ImmutableList<Artifact>> filtered =
        resourceFilter.maybeFilter(resources, /* isDependency= */ false);

    if (!filtered.isPresent()) {
      // Nothing was filtered out
      return this;
    }

    return withResources(ruleErrorConsumer, filtered.get(), "resource_files");
  }

  @VisibleForTesting
  static LocalResourceContainer forResources(
      RuleErrorConsumer ruleErrorConsumer, ImmutableList<Artifact> resources)
      throws RuleErrorException {
    return new LocalResourceContainer(
        resources,
        getResourceRoots(ruleErrorConsumer, resources, "resource_files"),
        ImmutableList.of(),
        ImmutableList.of());
  }

  private LocalResourceContainer withResources(
      RuleErrorConsumer ruleErrorConsumer, ImmutableList<Artifact> resources, String resourcesAttr)
      throws RuleErrorException {
    return new LocalResourceContainer(
        resources,
        getResourceRoots(ruleErrorConsumer, resources, resourcesAttr),
        assets,
        assetRoots);
  }
}
