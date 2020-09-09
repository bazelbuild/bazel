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
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleErrorConsumer;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingContext;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * The collected resources artifacts and roots.
 *
 * <p>This is used to encapsulate the logic and the data associated with the resources derived from
 * an appropriate android rule in a reusable instance.
 */
public class AndroidResources {
  private static final String DEFAULT_RESOURCES_ATTR = "resource_files";

  public static final String[] RESOURCES_ATTRIBUTES =
      new String[] {
        "manifest",
        DEFAULT_RESOURCES_ATTR,
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

  /**
   * Determines if the attributes contain resource and asset attributes.
   *
   * @deprecated We are moving towards processing Android assets, resources, and manifests
   *     separately. Use a separate method that just checks the attributes you need.
   */
  @Deprecated
  public static boolean definesAndroidResources(AttributeMap attributes) {
    for (String attribute : RESOURCES_ATTRIBUTES) {
      if (attributes.isAttributeValueExplicitlySpecified(attribute)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Checks validity of a RuleContext to produce Android resources, assets, and manifests.
   *
   * @throws RuleErrorException if the RuleContext is invalid. Accumulated errors will be available
   *     via {@code ruleContext}
   * @deprecated We are moving towards processing Android assets, resources, and manifests
   *     separately. Use a separate method that just checks the values you need.
   */
  @Deprecated
  public static void validateRuleContext(RuleContext ruleContext) throws RuleErrorException {
    AndroidAssets.validateAssetsAndAssetsDir(ruleContext);
    validateNoAndroidResourcesInSources(ruleContext);
    validateManifest(ruleContext);
  }

  /**
   * Validates that there are no targets with resources in the srcs, as they should not be used with
   * the Android data logic.
   */
  private static void validateNoAndroidResourcesInSources(RuleContext ruleContext)
      throws RuleErrorException {
    Iterable<AndroidResourcesInfo> resources =
        ruleContext.getPrerequisites("srcs", TransitionMode.TARGET, AndroidResourcesInfo.PROVIDER);
    for (AndroidResourcesInfo info : resources) {
      ruleContext.throwWithAttributeError(
          "srcs",
          String.format("srcs should not contain label with resources %s", info.getLabel()));
    }
  }

  private static void validateManifest(RuleContext ruleContext) throws RuleErrorException {
    if (ruleContext.getPrerequisiteArtifact("manifest", TransitionMode.TARGET) == null) {
      ruleContext.throwWithAttributeError(
          "manifest", "manifest is required when resource_files or assets are defined.");
    }
  }

  public static AndroidResources from(RuleContext ruleContext, String resourcesAttr)
      throws RuleErrorException {
    if (!hasLocalResourcesAttributes(ruleContext.attributes())) {
      return empty();
    }

    return from(
        ruleContext,
        ruleContext.getPrerequisites(resourcesAttr, TransitionMode.TARGET, FileProvider.class),
        resourcesAttr);
  }

  public static AndroidResources from(
      RuleErrorConsumer errorConsumer,
      Iterable<FileProvider> resourcesTargets,
      String resourcesAttr)
      throws RuleErrorException {
    return forResources(errorConsumer, getResources(resourcesTargets), resourcesAttr);
  }

  /** Returns an {@link AndroidResources} for a list of resource artifacts. */
  @VisibleForTesting
  public static AndroidResources forResources(
      RuleErrorConsumer ruleErrorConsumer, ImmutableList<Artifact> resources, String resourcesAttr)
      throws RuleErrorException {
    return new AndroidResources(
        resources, getResourceRoots(ruleErrorConsumer, resources, resourcesAttr));
  }

  /**
   * TODO(b/76218640): Whether local resources are built into a target should depend on that
   * target's resource attribute ("resource_files" in general, but local_resource_files for
   * android_test), not any other attributes.
   */
  private static boolean hasLocalResourcesAttributes(AttributeMap attrs) {
    return attrs.has("assets") || attrs.has("resource_files");
  }

  static AndroidResources empty() {
    return new AndroidResources(ImmutableList.of(), ImmutableList.of());
  }

  /**
   * Creates a {@link AndroidResources} containing all the resources in directory artifacts, for use
   * with AarImport rules.
   *
   * <p>In general, {@link #from(RuleContext, String)} should be used instead, but it can't be for
   * AarImport since we don't know about its individual assets at analysis time. No transitive
   * resources will be included in the container produced by this method.
   *
   * @param resourcesDir the tree artifact containing a {@code res/} directory
   */
  static AndroidResources forAarImport(SpecialArtifact resourcesDir) {
    Preconditions.checkArgument(resourcesDir.isTreeArtifact());
    return new AndroidResources(
        ImmutableList.of(resourcesDir),
        ImmutableList.of(resourcesDir.getExecPath().getChild("res")));
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
      throw ruleErrorConsumer.throwWithAttributeError(
          resourcesAttr,
          String.format(INCORRECT_RESOURCE_LAYOUT_MESSAGE, file.getRootRelativePath()));
    }

    if (lastResourceDir != null && !resourceDir.equals(lastResourceDir)) {
      throw ruleErrorConsumer.throwWithAttributeError(
          resourcesAttr,
          String.format(
              "'%s' (generated by '%s') is not in the same directory '%s' (derived from %s)."
                  + " All resources must share a common directory.",
              file.getRootRelativePath(),
              file.getOwnerLabel(),
              lastResourceDir,
              lastFile.getRootRelativePath()));
    }

    PathFragment packageFragment = file.getOwnerLabel().getPackageIdentifier().getSourceRoot();
    PathFragment packageRelativePath = file.getRootRelativePath().relativeTo(packageFragment);
    try {
      PathFragment path = file.getExecPath();
      resourceRoots.add(
          path.subFragment(
              0,
              path.segmentCount() - segmentCountAfterAncestor(resourceDir, packageRelativePath)));
    } catch (IllegalArgumentException e) {
      throw ruleErrorConsumer.throwWithAttributeError(
          resourcesAttr,
          String.format(
              "'%s' (generated by '%s') is not under the directory '%s' (derived from %s).",
              file.getRootRelativePath(),
              file.getOwnerLabel(),
              packageRelativePath,
              file.getRootRelativePath()));
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
    PathFragment fragment = artifact.getExecPath();
    int segmentCount = fragment.segmentCount();
    if (segmentCount < 3) {
      return null;
    }
    // TODO(bazel-team): Expand Fileset to verify, or remove Fileset as an option for resources.
    if (artifact.isFileset() || artifact.isTreeArtifact()) {
      return fragment.subFragment(segmentCount - 1);
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

  private static int segmentCountAfterAncestor(PathFragment ancestor, PathFragment path) {
    String cutAtSegment = ancestor.getSegment(ancestor.segmentCount() - 1);
    int index = -1;
    List<String> segments = path.getSegments();
    for (int i = segments.size() - 1; i >= 0; i--) {
      if (segments.get(i).equals(cutAtSegment)) {
        index = i;
        break;
      }
    }
    if (index == -1) {
      throw new IllegalArgumentException("PathFragment " + path + " is not beneath " + ancestor);
    }
    return segments.size() - index - 1;
  }

  private final ImmutableList<Artifact> resources;
  private final ImmutableList<PathFragment> resourceRoots;

  AndroidResources(AndroidResources other) {
    this(other.resources, other.resourceRoots);
  }

  @VisibleForTesting
  public AndroidResources(
      ImmutableList<Artifact> resources, ImmutableList<PathFragment> resourceRoots) {
    this.resources = resources;
    this.resourceRoots = resourceRoots;
  }

  private static ImmutableList<Artifact> getResources(Iterable<FileProvider> targets) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (FileProvider target : targets) {
      builder.addAll(target.getFilesToBuild().toList());
    }

    return builder.build();
  }

  public ImmutableList<Artifact> getResources() {
    return resources;
  }

  /**
   * Gets the roots of some resources.
   *
   * @return a list of roots, or an empty list of the passed resources cannot all be contained in a
   *     single {@link AndroidResources}. If that's the case, it will be reported to the {@link
   *     RuleErrorConsumer}.
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

  public ImmutableList<PathFragment> getResourceRoots() {
    return resourceRoots;
  }

  /**
   * Filters this object, assuming it contains the resources of the current target.
   *
   * <p>If this object contains the resources from a dependency of this target, use {@link
   * #maybeFilter(RuleErrorConsumer, ResourceFilter, boolean)} instead.
   *
   * @return a filtered {@link AndroidResources} object. If no filtering was done, this object will
   *     be returned.
   */
  public AndroidResources filterLocalResources(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter) throws RuleErrorException {
    Optional<? extends AndroidResources> filtered =
        maybeFilter(errorConsumer, resourceFilter, /* isDependency = */ false);
    return filtered.isPresent() ? filtered.get() : this;
  }

  /**
   * Filters this object.
   *
   * @return an optional wrapping a new {@link AndroidResources} with resources filtered by the
   *     passed {@link ResourceFilter}, or {@link Optional#empty()} if no resources should be
   *     filtered.
   */
  public Optional<? extends AndroidResources> maybeFilter(
      RuleErrorConsumer errorConsumer, ResourceFilter resourceFilter, boolean isDependency)
      throws RuleErrorException {
    Optional<ImmutableList<Artifact>> filtered =
        resourceFilter.maybeFilter(resources, /* isDependency= */ isDependency);

    if (!filtered.isPresent()) {
      // Nothing was filtered out
      return Optional.empty();
    }

    return Optional.of(
        new AndroidResources(
            filtered.get(),
            getResourceRoots(errorConsumer, filtered.get(), DEFAULT_RESOURCES_ATTR)));
  }

  /** Parses these resources. */
  public ParsedAndroidResources parse(
      AndroidDataContext dataContext,
      StampedAndroidManifest manifest,
      DataBindingContext dataBindingContext)
      throws InterruptedException {
    return ParsedAndroidResources.parseFrom(dataContext, this, manifest, dataBindingContext);
  }

  /**
   * Performs the complete resource processing pipeline - parsing, merging, and validation - on
   * these resources.
   */
  public ValidatedAndroidResources process(
      RuleContext ruleContext,
      AndroidDataContext dataContext,
      StampedAndroidManifest manifest,
      DataBindingContext dataBindingContext,
      boolean neverlink)
      throws RuleErrorException, InterruptedException {
    return process(
        dataContext,
        manifest,
        ResourceDependencies.fromRuleDeps(ruleContext, neverlink),
        dataBindingContext);
  }

  ValidatedAndroidResources process(
      AndroidDataContext dataContext,
      StampedAndroidManifest manifest,
      ResourceDependencies resourceDeps,
      DataBindingContext dataBindingContext)
      throws InterruptedException {
    return parse(dataContext, manifest, dataBindingContext)
        .merge(dataContext, resourceDeps)
        .validate(dataContext);
  }

  @Override
  public boolean equals(Object object) {
    if (!(object instanceof AndroidResources)) {
      return false;
    }

    AndroidResources other = (AndroidResources) object;
    return resources.equals(other.resources) && resourceRoots.equals(other.resourceRoots);
  }

  @Override
  public int hashCode() {
    return Objects.hash(resources, resourceRoots);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("resources", resources)
        .add("resourceRoots", resourceRoots)
        .toString();
  }
}
