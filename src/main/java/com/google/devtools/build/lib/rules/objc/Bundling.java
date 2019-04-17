// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_SWIFT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MULTI_ARCH_LINKED_BINARIES;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.NESTED_BUNDLE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ROOT_MERGE_ZIP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STORYBOARD;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XIB;

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Contains information regarding the creation of an iOS bundle.
 */
@Immutable
final class Bundling {

  /**
   * Names of top-level directories in dynamic frameworks (i.e. directly under the
   * {@code *.framework} directory) that should not be copied into the final bundle.
   */
  private static final ImmutableSet<String> STRIP_FRAMEWORK_DIRS =
      ImmutableSet.of("Headers", "PrivateHeaders", "Modules");

  static final class Builder {
    private String name;
    private String bundleDirFormat;
    private ImmutableList.Builder<BundleableFile> bundleFilesBuilder = ImmutableList.builder();
    private ObjcProvider objcProvider;
    private NestedSetBuilder<Artifact> infoplistInputs = NestedSetBuilder.stableOrder();
    private Artifact automaticEntriesInfoplistInput;
    private IntermediateArtifacts intermediateArtifacts;
    private String primaryBundleId;
    private String fallbackBundleId;
    private String architecture;
    private DottedVersion minimumOsVersion;
    private ImmutableSet<TargetDeviceFamily> families;
    private String artifactPrefix;
    @Nullable private String executableName;

    public Builder setName(String name) {
      this.name = name;
      return this;
    }

    /** Sets the name of the bundle's executable. */
    public Builder setExecutableName(String executableName) {
      this.executableName = executableName;
      return this;
    }

    /**
     * Sets the CPU architecture this bundling was constructed for. Legal value are any that may be
     * set on {@link AppleConfiguration#getIosCpu()}.
     */
    public Builder setArchitecture(String architecture) {
      this.architecture = architecture;
      return this;
    }

    public Builder setBundleDirFormat(String bundleDirFormat) {
      this.bundleDirFormat = bundleDirFormat;
      return this;
    }

    public Builder addExtraBundleFiles(ImmutableList<BundleableFile> extraBundleFiles) {
      this.bundleFilesBuilder.addAll(extraBundleFiles);
      return this;
    }

    public Builder setObjcProvider(ObjcProvider objcProvider) {
      this.objcProvider = objcProvider;
      return this;
    }

    /**
     * Adds an artifact representing an {@code Info.plist} as an input to this bundle's
     * {@code Info.plist} (which is merged from any such added plists plus the generated
     * automatic entries plist).
     */
    public Builder addInfoplistInput(Artifact infoplist) {
      this.infoplistInputs.add(infoplist);
      return this;
    }

    /**
     * Adds the given list of artifacts representing {@code Info.plist}s that are to be merged into
     * this bundle's {@code Info.plist}.
     */
    public Builder addInfoplistInputs(Iterable<Artifact> infoplists) {
      this.infoplistInputs.addAll(infoplists);
      return this;
    }

    /**
     * Adds an artifact representing an {@code Info.plist} that contains automatic entries
     * generated by xcode.
     */
    public Builder setAutomaticEntriesInfoplistInput(Artifact automaticEntriesInfoplist) {
      this.automaticEntriesInfoplistInput = automaticEntriesInfoplist;
      return this;
    }

    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    public Builder setPrimaryBundleId(String primaryId) {
      this.primaryBundleId = primaryId;
      return this;
    }

    public Builder setFallbackBundleId(String fallbackId) {
      this.fallbackBundleId = fallbackId;
      return this;
    }

    /**
     * Sets the minimum OS version for this bundle which will be used when constructing the bundle's
     * plist.
     */
    public Builder setMinimumOsVersion(DottedVersion minimumOsVersion) {
      this.minimumOsVersion = minimumOsVersion;
      return this;
    }

    public Builder setTargetDeviceFamilies(ImmutableSet<TargetDeviceFamily> families) {
      this.families = families;
      return this;
    }

    public Builder setArtifactPrefix(String artifactPrefix) {
      this.artifactPrefix = artifactPrefix;
      return this;
    }

    private static NestedSet<Artifact> nestedBundleContentArtifacts(Iterable<Bundling> bundles) {
      NestedSetBuilder<Artifact> artifacts = NestedSetBuilder.stableOrder();
      for (Bundling bundle : bundles) {
        artifacts.addTransitive(bundle.getBundleContentArtifacts());
      }
      return artifacts.build();
    }

    private NestedSet<Artifact> mergeZips(Optional<Artifact> actoolzipOutput) {
      NestedSetBuilder<Artifact> mergeZipBuilder =
          NestedSetBuilder.<Artifact>stableOrder()
              .addAll(actoolzipOutput.asSet())
              .addAll(
                  Xcdatamodel.outputZips(
                      Xcdatamodels.xcdatamodels(
                          intermediateArtifacts, objcProvider.get(XCDATAMODEL))))
              .addTransitive(objcProvider.get(MERGE_ZIP));
      for (Artifact xibFile : objcProvider.get(XIB)) {
        mergeZipBuilder.add(intermediateArtifacts.compiledXibFileZip(xibFile));
      }
      for (Artifact storyboard : objcProvider.get(STORYBOARD)) {
        mergeZipBuilder.add(intermediateArtifacts.compiledStoryboardZip(storyboard));
      }

      if (objcProvider.is(USES_SWIFT)) {
        mergeZipBuilder.add(intermediateArtifacts.swiftFrameworksFileZip());
      }

      return mergeZipBuilder.build();
    }

    private NestedSet<Artifact> rootMergeZips() {
      NestedSetBuilder<Artifact> rootMergeZipsBuilder =
          NestedSetBuilder.<Artifact>stableOrder().addTransitive(objcProvider.get(ROOT_MERGE_ZIP));

      if (objcProvider.is(USES_SWIFT)) {
        rootMergeZipsBuilder.add(intermediateArtifacts.swiftSupportZip());
      }

      return rootMergeZipsBuilder.build();
    }

    private NestedSet<Artifact> bundleInfoplistInputs() {
      if (objcProvider.hasAssetCatalogs()) {
        infoplistInputs.add(intermediateArtifacts.actoolPartialInfoplist());
      }
      return infoplistInputs.build();
    }

    private Optional<Artifact> bundleInfoplist(NestedSet<Artifact> bundleInfoplistInputs) {
      if (bundleInfoplistInputs.isEmpty()) {
        return Optional.absent();
      }
      if (needsToMerge(bundleInfoplistInputs, primaryBundleId, fallbackBundleId)) {
        return Optional.of(intermediateArtifacts.mergedInfoplist());
      }
      return Optional.of(Iterables.getOnlyElement(bundleInfoplistInputs));
    }

    private Optional<Artifact> combinedArchitectureBinary() {
      if (!Iterables.isEmpty(objcProvider.get(MULTI_ARCH_LINKED_BINARIES))) {
        return Optional.of(Iterables.getOnlyElement(objcProvider.get(MULTI_ARCH_LINKED_BINARIES)));
      } else if (!Iterables.isEmpty(objcProvider.get(LIBRARY))
          || !Iterables.isEmpty(objcProvider.get(IMPORTED_LIBRARY))) {
        return Optional.of(intermediateArtifacts.combinedArchitectureBinary());
      }
      return Optional.absent();
    }

    private Optional<Artifact> actoolzipOutput() {
      Optional<Artifact> actoolzipOutput = Optional.absent();
      if (!Iterables.isEmpty(objcProvider.get(ASSET_CATALOG))) {
        actoolzipOutput = Optional.of(intermediateArtifacts.actoolzipOutput());
      }
      return actoolzipOutput;
    }

    private NestedSet<BundleableFile> binaryStringsFiles() {
      NestedSetBuilder<BundleableFile> binaryStringsBuilder = NestedSetBuilder.stableOrder();
      for (Artifact stringsFile : objcProvider.get(STRINGS)) {
        BundleableFile bundleFile =
            new BundleableFile(
                intermediateArtifacts.convertedStringsFile(stringsFile),
                BundleableFile.flatBundlePath(stringsFile.getExecPath()));
        binaryStringsBuilder.add(bundleFile);
      }
      return binaryStringsBuilder.build();
    }

    private NestedSet<BundleableFile> dynamicFrameworkFiles() {
      NestedSetBuilder<BundleableFile> frameworkFilesBuilder = NestedSetBuilder.stableOrder();
      for (Artifact frameworkFile : objcProvider.get(DYNAMIC_FRAMEWORK_FILE)) {
        PathFragment frameworkDir =
            ObjcCommon.nearestContainerMatching(ObjcCommon.FRAMEWORK_CONTAINER_TYPE, frameworkFile)
                .get();
        String frameworkName = frameworkDir.getBaseName();
        PathFragment inFrameworkPath = frameworkFile.getExecPath().relativeTo(frameworkDir);
        if (inFrameworkPath.getFirstSegment(STRIP_FRAMEWORK_DIRS) == 0) {
          continue;
        }
        // If this is a top-level file in the framework set the executable bit (to make sure we set
        // the bit on the actual dylib binary - other files may also get it but we have no way to
        // distinguish them).
        int permissions =
            inFrameworkPath.segmentCount() == 1
                ? BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE
                : BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE;
        BundleableFile bundleFile =
            new BundleableFile(
                frameworkFile,
                "Frameworks/" + frameworkName + "/" + inFrameworkPath.getPathString(),
                permissions);
        frameworkFilesBuilder.add(bundleFile);
      }
      return frameworkFilesBuilder.build();
    }

    /**
     * Filters files that would map to the same location in the bundle, adding only one copy to the
     * set of files returned.
     *
     * <p>Files can have the same bundle path for various illegal reasons and errors are raised for
     * that separately. There are situations though where the same file exists multiple times (for
     * example in multi-architecture builds) and would conflict when creating the bundle. In all
     * these cases it shouldn't matter which one is included and this class will select the first
     * one.
     */
    ImmutableList<BundleableFile> deduplicateByBundlePaths(
        ImmutableList<BundleableFile> bundleFiles) {
      ImmutableList.Builder<BundleableFile> deduplicated = ImmutableList.builder();
      Set<String> bundlePaths = new HashSet<>();
      for (BundleableFile bundleFile : bundleFiles) {
        if (bundlePaths.add(bundleFile.getBundlePath())) {
          deduplicated.add(bundleFile);
        }
      }
      return deduplicated.build();
    }

    public Bundling build() {
      Preconditions.checkNotNull(intermediateArtifacts, "intermediateArtifacts should not be null");
      Preconditions.checkNotNull(families, "families should not be null");
      NestedSet<Artifact> bundleInfoplistInputs = bundleInfoplistInputs();
      Optional<Artifact> bundleInfoplist = bundleInfoplist(bundleInfoplistInputs);
      Optional<Artifact> actoolzipOutput = actoolzipOutput();
      Optional<Artifact> combinedArchitectureBinary = combinedArchitectureBinary();
      NestedSet<BundleableFile> binaryStringsFiles = binaryStringsFiles();
      NestedSet<BundleableFile> dynamicFrameworks = dynamicFrameworkFiles();
      NestedSet<Artifact> mergeZips = mergeZips(actoolzipOutput);
      NestedSet<Artifact> rootMergeZips = rootMergeZips();

      bundleFilesBuilder
          .addAll(binaryStringsFiles)
          .addAll(dynamicFrameworks)
          .addAll(objcProvider.get(BUNDLE_FILE));
      ImmutableList<BundleableFile> bundleFiles =
          deduplicateByBundlePaths(bundleFilesBuilder.build());

      NestedSetBuilder<Artifact> bundleContentArtifactsBuilder =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(nestedBundleContentArtifacts(objcProvider.get(NESTED_BUNDLE)))
              .addAll(combinedArchitectureBinary.asSet())
              .addAll(bundleInfoplist.asSet())
              .addTransitive(mergeZips)
              .addTransitive(rootMergeZips)
              .addAll(BundleableFile.toArtifacts(bundleFiles));

      return new Bundling(
          name,
          executableName,
          bundleDirFormat,
          combinedArchitectureBinary,
          bundleFiles,
          bundleInfoplist,
          actoolzipOutput,
          bundleContentArtifactsBuilder.build(),
          mergeZips,
          rootMergeZips,
          primaryBundleId,
          fallbackBundleId,
          architecture,
          minimumOsVersion,
          bundleInfoplistInputs,
          automaticEntriesInfoplistInput,
          objcProvider.get(NESTED_BUNDLE),
          families,
          intermediateArtifacts,
          artifactPrefix);
    }
  }

  private static boolean needsToMerge(
      NestedSet<Artifact> bundleInfoplistInputs, String primaryBundleId, String fallbackBundleId) {
    return primaryBundleId != null || fallbackBundleId != null
        || Iterables.size(bundleInfoplistInputs) > 1;
  }

  private final String name;
  @Nullable private final String executableName;
  private final String architecture;
  private final String bundleDirFormat;
  private final Optional<Artifact> combinedArchitectureBinary;
  private final ImmutableList<BundleableFile> bundleFiles;
  private final Optional<Artifact> bundleInfoplist;
  private final Optional<Artifact> actoolzipOutput;
  private final NestedSet<Artifact> bundleContentArtifacts;
  private final NestedSet<Artifact> mergeZips;
  private final NestedSet<Artifact> rootMergeZips;
  private final String primaryBundleId;
  private final String fallbackBundleId;
  private final DottedVersion minimumOsVersion;
  private final NestedSet<Artifact> infoplistInputs;
  private final NestedSet<Bundling> nestedBundlings;
  private Artifact automaticEntriesInfoplistInput;
  private final ImmutableSet<TargetDeviceFamily> families;
  private final IntermediateArtifacts intermediateArtifacts;
  private final String artifactPrefix;

  private Bundling(
      String name,
      String executableName,
      String bundleDirFormat,
      Optional<Artifact> combinedArchitectureBinary,
      ImmutableList<BundleableFile> bundleFiles,
      Optional<Artifact> bundleInfoplist,
      Optional<Artifact> actoolzipOutput,
      NestedSet<Artifact> bundleContentArtifacts,
      NestedSet<Artifact> mergeZips,
      NestedSet<Artifact> rootMergeZips,
      String primaryBundleId,
      String fallbackBundleId,
      String architecture,
      DottedVersion minimumOsVersion,
      NestedSet<Artifact> infoplistInputs,
      Artifact automaticEntriesInfoplistInput,
      NestedSet<Bundling> nestedBundlings,
      ImmutableSet<TargetDeviceFamily> families,
      IntermediateArtifacts intermediateArtifacts,
      String artifactPrefix) {
    this.nestedBundlings = Preconditions.checkNotNull(nestedBundlings);
    this.name = Preconditions.checkNotNull(name);
    this.executableName = executableName;
    this.bundleDirFormat = Preconditions.checkNotNull(bundleDirFormat);
    this.combinedArchitectureBinary = Preconditions.checkNotNull(combinedArchitectureBinary);
    this.bundleFiles = Preconditions.checkNotNull(bundleFiles);
    this.bundleInfoplist = Preconditions.checkNotNull(bundleInfoplist);
    this.actoolzipOutput = Preconditions.checkNotNull(actoolzipOutput);
    this.bundleContentArtifacts = Preconditions.checkNotNull(bundleContentArtifacts);
    this.mergeZips = Preconditions.checkNotNull(mergeZips);
    this.rootMergeZips = Preconditions.checkNotNull(rootMergeZips);
    this.fallbackBundleId = fallbackBundleId;
    this.primaryBundleId = primaryBundleId;
    this.architecture = Preconditions.checkNotNull(architecture);
    this.minimumOsVersion = Preconditions.checkNotNull(minimumOsVersion);
    this.infoplistInputs = Preconditions.checkNotNull(infoplistInputs);
    this.automaticEntriesInfoplistInput = automaticEntriesInfoplistInput;
    this.families = Preconditions.checkNotNull(families);
    this.intermediateArtifacts = intermediateArtifacts;
    this.artifactPrefix = artifactPrefix;
  }

  /**
   * The bundle directory. For apps, this would be {@code "Payload/TARGET_NAME.app"}, which is where
   * in the bundle zip archive every file is found, including the linked binary, nested bundles, and
   * everything returned by {@link #getBundleFiles()}.
   */
  public String getBundleDir() {
    return String.format(bundleDirFormat, name);
  }

  /**
   * The name of the bundle, from which the bundle root and the path of the linked binary in the
   * bundle archive are derived.
   */
  public String getName() {
    return name;
  }
  
  /** The name of the bundle's executable, or null if the bundle has no executable. */
  @Nullable public String getExecutableName() {
    return executableName;
  }

  /**
   * An {@link Optional} with the linked binary artifact, or {@link Optional#absent()} if it is
   * empty and should not be included in the bundle.
   */
  public Optional<Artifact> getCombinedArchitectureBinary() {
    return combinedArchitectureBinary;
  }

  /**
   * Bundle files to include in the bundle. These files are placed under the bundle root (possibly
   * nested, of course, depending on the bundle path of the files).
   */
  public ImmutableList<BundleableFile> getBundleFiles() {
    return bundleFiles;
  }

  /**
   * Returns any bundles nested in this one.
   */
  public NestedSet<Bundling> getNestedBundlings() {
    return nestedBundlings;
  }

  /**
   * Returns an artifact representing this bundle's {@code Info.plist} or {@link Optional#absent()}
   * if this bundle has no info plist inputs.
   */
  public Optional<Artifact> getBundleInfoplist() {
    return bundleInfoplist;
  }

  /**
   * Returns all info plists that need to be merged into this bundle's {@link #getBundleInfoplist()
   * info plist}, other than that plist that contains blaze-generated automatic entires.
   */
  public NestedSet<Artifact> getBundleInfoplistInputs() {
    return infoplistInputs;
  }

  /**
   * Returns an artifact representing a plist containing automatic entries generated by bazel.
   */
  public Artifact getAutomaticInfoPlist() {
    return automaticEntriesInfoplistInput;
  }

  /**
   * Returns all artifacts that are required as input to the merging of the final plist.
   */
  public NestedSet<Artifact> getMergingContentArtifacts() {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();
    result.addTransitive(infoplistInputs);
    if (automaticEntriesInfoplistInput != null) {
      result.add(automaticEntriesInfoplistInput);
    }
    return result.build();
  }

  /**
   * Returns {@code true} if this bundle requires merging of its {@link #getBundleInfoplist() info
   * plist}.
   */
  public boolean needsToMergeInfoplist() {
    return needsToMerge(infoplistInputs, primaryBundleId, fallbackBundleId);
  }

  /**
   * The location of the actoolzip output for this bundle. This is non-absent only included in the
   * bundle if there is at least one asset catalog artifact supplied by
   * {@link ObjcProvider#ASSET_CATALOG}.
   */
  public Optional<Artifact> getActoolzipOutput() {
    return actoolzipOutput;
  }

  /**
   * Returns all zip files whose contents should be merged into this bundle under the main bundle
   * directory. For instance, if a merge zip contains files a/b and c/d, then the resulting bundling
   * would have additional files at:
   * <ul>
   *   <li>{bundleDir}/a/b
   *   <li>{bundleDir}/c/d
   * </ul>
   */
  public NestedSet<Artifact> getMergeZips() {
    return mergeZips;
  }

  /**
   * Returns all zip files whose contents should be merged into final ipa and outside the
   * main bundle. For instance, if a merge zip contains files dir1/file1, then the resulting
   * bundling would have additional files at:
   * <ul>
   *   <li>dir1/file1
   *   <li>{bundleDir}/other_files
   * </ul>
   */
  public NestedSet<Artifact> getRootMergeZips() {
    return rootMergeZips;
  }

  /**
   * Returns the variable substitutions that should be used when merging the plist info file of
   * this bundle.
   */
  public Map<String, String> variableSubstitutions() {
    return ImmutableMap.of(
        "EXECUTABLE_NAME", Strings.nullToEmpty(executableName),
        "BUNDLE_NAME", PathFragment.create(getBundleDir()).getBaseName(),
        "PRODUCT_NAME", name);
  }

  /**
   * Returns the artifacts that are required to generate this bundle.
   */
  public NestedSet<Artifact> getBundleContentArtifacts() {
    return bundleContentArtifacts;
  }

  /**
   * Returns primary bundle ID to use, can be null.
   */
  public String getPrimaryBundleId() {
    return primaryBundleId;
  }
  
  /**
   * Returns fallback bundle ID to use when primary isn't set.
   */
  public String getFallbackBundleId() {
    return fallbackBundleId;
  }

  /**
   * Returns the iOS CPU architecture this bundle was constructed for.
   */
  public String getArchitecture() {
    return architecture;
  }

  /**
   * Returns the minimum iOS version this bundle's plist and resources should be generated for
   * (does <b>not</b> affect the minimum OS version its binary is compiled with).
   */
  public DottedVersion getMinimumOsVersion() {
    return minimumOsVersion;
  }

  /**
   * Returns the list of {@link TargetDeviceFamily} values this bundle is targeting. If empty, the
   * default values specified by "families" will be used.
   */
  public ImmutableSet<TargetDeviceFamily> getTargetDeviceFamilies() {
    return families;
  }

  /**
   * Returns {@link IntermediateArtifacts} required to create this bundle.
   */
  public IntermediateArtifacts getIntermediateArtifacts() {
    return intermediateArtifacts;
  }

  /**
   * Returns the prefix to be added to all generated artifact names, can be null. This is useful to
   * disambiguate artifacts for multiple bundles created with different names within same rule.
   */
  public String getArtifactPrefix() {
    return artifactPrefix;
  }
}
