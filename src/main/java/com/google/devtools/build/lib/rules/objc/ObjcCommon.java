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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_IMPORT_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DEFINE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FLAG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_FOR_XCODEGEN;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GCNO;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INSTRUMENTED_SOURCE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LINKED_BINARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SOURCE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STORYBOARD;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STRINGS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XIB;
import static com.google.devtools.build.lib.vfs.PathFragment.TO_PATH_FRAGMENT;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.RegexFilter;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Contains information common to multiple objc_* rules, and provides a unified API for extracting
 * and accessing it.
 */
// TODO(bazel-team): Decompose and subsume area-specific logic and data into the various *Support
// classes. Make sure to distinguish rule output (providers, runfiles, ...) from intermediate,
// rule-internal information. Any provider created by a rule should not be read, only published.
public final class ObjcCommon {
  /**
   * Provides a way to access attributes that are common to all compilation rules.
   */
  // TODO(bazel-team): Delete and move into support-specific attributes classes once ObjcCommon is
  // gone.
  static final class CompilationAttributes {
    private final RuleContext ruleContext;
    private final ObjcSdkFrameworks.Attributes sdkFrameworkAttributes;

    CompilationAttributes(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
      this.sdkFrameworkAttributes = new ObjcSdkFrameworks.Attributes(ruleContext);
    }

    ImmutableList<Artifact> hdrs() {
      return ImmutableList.copyOf(CcCommon.getHeaders(ruleContext));
    }

    Iterable<PathFragment> includes() {
      return Iterables.transform(
          ruleContext.attributes().get("includes", Type.STRING_LIST),
          PathFragment.TO_PATH_FRAGMENT);
    }

    Iterable<PathFragment> sdkIncludes() {
      return Iterables.transform(
          ruleContext.attributes().get("sdk_includes", Type.STRING_LIST),
          PathFragment.TO_PATH_FRAGMENT);
    }

    /**
     * Returns the value of the sdk_frameworks attribute plus frameworks that are included
     * automatically.
     */
    ImmutableSet<SdkFramework> sdkFrameworks() {
      return sdkFrameworkAttributes.sdkFrameworks();
    }

    /**
     * Returns the value of the weak_sdk_frameworks attribute.
     */
    ImmutableSet<SdkFramework> weakSdkFrameworks() {
      return sdkFrameworkAttributes.weakSdkFrameworks();
    }

    /**
     * Returns the value of the sdk_dylibs attribute.
     */
    ImmutableSet<String> sdkDylibs() {
      return sdkFrameworkAttributes.sdkDylibs();
    }

    /**
     * Returns the exec paths of all header search paths that should be added to this target and
     * dependers on this target, obtained from the {@code includes} attribute.
     */
    ImmutableList<PathFragment> headerSearchPaths() {
      ImmutableList.Builder<PathFragment> paths = new ImmutableList.Builder<>();
      PathFragment packageFragment =
          ruleContext.getLabel().getPackageIdentifier().getPathFragment();
      List<PathFragment> rootFragments = ImmutableList.of(
          packageFragment,
          ruleContext.getConfiguration().getGenfilesFragment().getRelative(packageFragment));

      Iterable<PathFragment> relativeIncludes =
          Iterables.filter(includes(), Predicates.not(PathFragment.IS_ABSOLUTE));
      for (PathFragment include : relativeIncludes) {
        for (PathFragment rootFragment : rootFragments) {
          paths.add(rootFragment.getRelative(include).normalize());
        }
      }
      return paths.build();
    }
  }

  /**
   * Provides a way to access attributes that are common to all resources rules.
   */
  // TODO(bazel-team): Delete and move into support-specific attributes classes once ObjcCommon is
  // gone.
  static final class ResourceAttributes {
    private final RuleContext ruleContext;

    ResourceAttributes(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    ImmutableList<Artifact> strings() {
      return ruleContext.getPrerequisiteArtifacts("strings", Mode.TARGET).list();
    }

    ImmutableList<Artifact> xibs() {
      return ruleContext.getPrerequisiteArtifacts("xibs", Mode.TARGET)
          .errorsForNonMatching(ObjcRuleClasses.XIB_TYPE)
          .list();
    }

    ImmutableList<Artifact> storyboards() {
      return ruleContext.getPrerequisiteArtifacts("storyboards", Mode.TARGET).list();
    }

    ImmutableList<Artifact> resources() {
      return ruleContext.getPrerequisiteArtifacts("resources", Mode.TARGET).list();
    }

    ImmutableList<Artifact> structuredResources() {
      return ruleContext.getPrerequisiteArtifacts("structured_resources", Mode.TARGET).list();
    }

    ImmutableList<Artifact> datamodels() {
      return ruleContext.getPrerequisiteArtifacts("datamodels", Mode.TARGET).list();
    }

    ImmutableList<Artifact> assetCatalogs() {
      return ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET).list();
    }
  }

  static class Builder {
    private RuleContext context;
    private Optional<CompilationAttributes> compilationAttributes = Optional.absent();
    private Optional<ResourceAttributes> resourceAttributes = Optional.absent();
    private Iterable<SdkFramework> extraSdkFrameworks = ImmutableList.of();
    private Iterable<SdkFramework> extraWeakSdkFrameworks = ImmutableList.of();
    private Iterable<String> extraSdkDylibs = ImmutableList.of();
    private Iterable<Artifact> frameworkImports = ImmutableList.of();
    private Optional<CompilationArtifacts> compilationArtifacts = Optional.absent();
    private Iterable<ObjcProvider> depObjcProviders = ImmutableList.of();
    private Iterable<ObjcProvider> directDepObjcProviders = ImmutableList.of();
    private Iterable<String> defines = ImmutableList.of();
    private Iterable<PathFragment> userHeaderSearchPaths = ImmutableList.of();
    private Iterable<Artifact> headers = ImmutableList.of();
    private IntermediateArtifacts intermediateArtifacts;
    private boolean alwayslink;
    private Iterable<Artifact> extraImportLibraries = ImmutableList.of();
    private Optional<Artifact> linkedBinary = Optional.absent();

    Builder(RuleContext context) {
      this.context = Preconditions.checkNotNull(context);
    }

    public Builder setCompilationAttributes(CompilationAttributes baseCompilationAttributes) {
      Preconditions.checkState(!this.compilationAttributes.isPresent(),
          "compilationAttributes is already set to: %s", this.compilationAttributes);
      this.compilationAttributes = Optional.of(baseCompilationAttributes);
      return this;
    }

    public Builder setResourceAttributes(ResourceAttributes baseResourceAttributes) {
      Preconditions.checkState(!this.resourceAttributes.isPresent(),
          "resourceAttributes is already set to: %s", this.resourceAttributes);
      this.resourceAttributes = Optional.of(baseResourceAttributes);
      return this;
    }

    Builder addExtraSdkFrameworks(Iterable<SdkFramework> extraSdkFrameworks) {
      this.extraSdkFrameworks = Iterables.concat(this.extraSdkFrameworks, extraSdkFrameworks);
      return this;
    }

    Builder addExtraWeakSdkFrameworks(Iterable<SdkFramework> extraWeakSdkFrameworks) {
      this.extraWeakSdkFrameworks =
          Iterables.concat(this.extraWeakSdkFrameworks, extraWeakSdkFrameworks);
      return this;
    }

    Builder addExtraSdkDylibs(Iterable<String> extraSdkDylibs) {
      this.extraSdkDylibs = Iterables.concat(this.extraSdkDylibs, extraSdkDylibs);
      return this;
    }

    Builder addFrameworkImports(Iterable<Artifact> frameworkImports) {
      this.frameworkImports = Iterables.concat(this.frameworkImports, frameworkImports);
      return this;
    }

    Builder setCompilationArtifacts(CompilationArtifacts compilationArtifacts) {
      Preconditions.checkState(!this.compilationArtifacts.isPresent(),
          "compilationArtifacts is already set to: %s", this.compilationArtifacts);
      this.compilationArtifacts = Optional.of(compilationArtifacts);
      return this;
    }

    /**
     * Add providers which will be exposed both to the declaring rule and to any dependers on the
     * declaring rule.
     */
    Builder addDepObjcProviders(Iterable<ObjcProvider> depObjcProviders) {
      this.depObjcProviders = Iterables.concat(this.depObjcProviders, depObjcProviders);
      return this;
    }

    /**
     * Add providers which will only be used by the declaring rule, and won't be propagated to any
     * dependers on the declaring rule.
     */
    Builder addNonPropagatedDepObjcProviders(Iterable<ObjcProvider> directDepObjcProviders) {
      this.directDepObjcProviders = Iterables.concat(
          this.directDepObjcProviders, directDepObjcProviders);
      return this;
    }

    public Builder addUserHeaderSearchPaths(Iterable<PathFragment> userHeaderSearchPaths) {
      this.userHeaderSearchPaths =
          Iterables.concat(this.userHeaderSearchPaths, userHeaderSearchPaths);
      return this;
    }

    public Builder addDefines(Iterable<String> defines) {
      this.defines = Iterables.concat(this.defines, defines);
      return this;
    }

    public Builder addHeaders(Iterable<Artifact> headers) {
      this.headers = Iterables.concat(this.headers, headers);
      return this;
    }

    Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    Builder setAlwayslink(boolean alwayslink) {
      this.alwayslink = alwayslink;
      return this;
    }

    /**
     * Adds additional static libraries to be linked into the final ObjC application bundle.
     */
    Builder addExtraImportLibraries(Iterable<Artifact> extraImportLibraries) {
      this.extraImportLibraries = Iterables.concat(this.extraImportLibraries, extraImportLibraries);
      return this;
    }

    /**
     * Sets a linked binary generated by this rule to be propagated to dependers.
     */
    Builder setLinkedBinary(Artifact linkedBinary) {
      this.linkedBinary = Optional.of(linkedBinary);
      return this;
    }

    ObjcCommon build() {
      Iterable<BundleableFile> bundleImports = BundleableFile.bundleImportsFromRule(context);

      ObjcProvider.Builder objcProvider = new ObjcProvider.Builder()
          .addAll(IMPORTED_LIBRARY, extraImportLibraries)
          .addAll(BUNDLE_FILE, bundleImports)
          .addAll(BUNDLE_IMPORT_DIR,
              uniqueContainers(BundleableFile.toArtifacts(bundleImports), BUNDLE_CONTAINER_TYPE))
          .addAll(SDK_FRAMEWORK, extraSdkFrameworks)
          .addAll(WEAK_SDK_FRAMEWORK, extraWeakSdkFrameworks)
          .addAll(SDK_DYLIB, extraSdkDylibs)
          .addAll(FRAMEWORK_FILE, frameworkImports)
          .addAll(FRAMEWORK_DIR, uniqueContainers(frameworkImports, FRAMEWORK_CONTAINER_TYPE))
          .addAll(INCLUDE, userHeaderSearchPaths)
          .addAll(DEFINE, defines)
          .addAll(HEADER, headers)
          .addTransitiveAndPropagate(depObjcProviders)
          .addTransitiveWithoutPropagating(directDepObjcProviders);

      if (compilationAttributes.isPresent()) {
        CompilationAttributes attributes = compilationAttributes.get();
        ObjcConfiguration objcConfiguration = ObjcRuleClasses.objcConfiguration(context);
        Iterable<PathFragment> sdkIncludes = Iterables.transform(
            Interspersing.prependEach(
                IosSdkCommands.sdkDir(objcConfiguration) + "/usr/include/",
                PathFragment.safePathStrings(attributes.sdkIncludes())),
            TO_PATH_FRAGMENT);
        objcProvider
            .addAll(HEADER, attributes.hdrs())
            .addAll(INCLUDE, attributes.headerSearchPaths())
            .addAll(INCLUDE, sdkIncludes)
            .addAll(SDK_FRAMEWORK, attributes.sdkFrameworks())
            .addAll(WEAK_SDK_FRAMEWORK, attributes.weakSdkFrameworks())
            .addAll(SDK_DYLIB, attributes.sdkDylibs());
      }

      if (resourceAttributes.isPresent()) {
        ResourceAttributes attributes = resourceAttributes.get();
        objcProvider
            .addAll(GENERAL_RESOURCE_FILE, attributes.storyboards())
            .addAll(GENERAL_RESOURCE_FILE, attributes.resources())
            .addAll(GENERAL_RESOURCE_FILE, attributes.strings())
            .addAll(GENERAL_RESOURCE_FILE, attributes.xibs())
            .addAll(BUNDLE_FILE, BundleableFile.flattenedRawResourceFiles(attributes.resources()))
            .addAll(BUNDLE_FILE,
                BundleableFile.structuredRawResourceFiles(attributes.structuredResources()))
            .addAll(XCASSETS_DIR,
                uniqueContainers(attributes.assetCatalogs(), ASSET_CATALOG_CONTAINER_TYPE))
            .addAll(ASSET_CATALOG, attributes.assetCatalogs())
            .addAll(XCDATAMODEL, attributes.datamodels())
            .addAll(XIB, attributes.xibs())
            .addAll(STRINGS, attributes.strings())
            .addAll(STORYBOARD, attributes.storyboards());
      }

      for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
        Iterable<Artifact> allSources =
            Iterables.concat(artifacts.getSrcs(), artifacts.getNonArcSrcs());
        objcProvider.addAll(LIBRARY, artifacts.getArchive().asSet());
        objcProvider.addAll(SOURCE, allSources);
        BuildConfiguration configuration = context.getConfiguration();
        RegexFilter filter = configuration.getInstrumentationFilter();
        if (configuration.isCodeCoverageEnabled()
            && filter.isIncluded(context.getLabel().toString())) {
          for (Artifact source : allSources) {
            objcProvider.add(INSTRUMENTED_SOURCE, source);
            objcProvider.add(GCNO, intermediateArtifacts.gcnoFile(source));
          }
        }

        boolean usesCpp = false;
        for (Artifact sourceFile :
            Iterables.concat(artifacts.getSrcs(), artifacts.getNonArcSrcs())) {
          usesCpp = usesCpp || ObjcRuleClasses.CPP_SOURCES.matches(sourceFile.getExecPath());
        }
        if (usesCpp) {
          objcProvider.add(FLAG, USES_CPP);
        }
      }

      if (alwayslink) {
        for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
          for (Artifact archive : artifacts.getArchive().asSet()) {
            objcProvider.add(FORCE_LOAD_LIBRARY, archive);
            objcProvider.add(FORCE_LOAD_FOR_XCODEGEN, String.format(
                "$(BUILT_PRODUCTS_DIR)/lib%s.a",
                XcodeProvider.xcodeTargetName(context.getLabel())));
          }
        }
        for (Artifact archive : extraImportLibraries) {
          objcProvider.add(FORCE_LOAD_LIBRARY, archive);
          objcProvider.add(FORCE_LOAD_FOR_XCODEGEN,
              "$(WORKSPACE_ROOT)/" + archive.getExecPath().getSafePathString());
        }
      }

      objcProvider.addAll(LINKED_BINARY, linkedBinary.asSet());

      return new ObjcCommon(context, objcProvider.build(), compilationArtifacts);
    }

  }

  static final FileType BUNDLE_CONTAINER_TYPE = FileType.of(".bundle");

  static final FileType ASSET_CATALOG_CONTAINER_TYPE = FileType.of(".xcassets");

  static final FileType FRAMEWORK_CONTAINER_TYPE = FileType.of(".framework");
  private final RuleContext context;
  private final ObjcProvider objcProvider;

  private final Optional<CompilationArtifacts> compilationArtifacts;

  private ObjcCommon(
      RuleContext context,
      ObjcProvider objcProvider,
      Optional<CompilationArtifacts> compilationArtifacts) {
    this.context = Preconditions.checkNotNull(context);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.compilationArtifacts = Preconditions.checkNotNull(compilationArtifacts);
  }

  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }

  public Optional<CompilationArtifacts> getCompilationArtifacts() {
    return compilationArtifacts;
  }

  /**
   * Returns an {@link Optional} containing the compiled {@code .a} file, or
   * {@link Optional#absent()} if this object contains no {@link CompilationArtifacts} or the
   * compilation information has no sources.
   */
  public Optional<Artifact> getCompiledArchive() {
    for (CompilationArtifacts justCompilationArtifacts : compilationArtifacts.asSet()) {
      return justCompilationArtifacts.getArchive();
    }
    return Optional.absent();
  }

  /**
   * Reports any known errors to the {@link RuleContext}. This should be called exactly once for
   * a target.
   */
  public void reportErrors() {

    // TODO(bazel-team): Report errors for rules that are not actually useful (i.e. objc_library
    // without sources or resources, empty objc_bundles)
  }

  static ImmutableList<PathFragment> userHeaderSearchPaths(BuildConfiguration configuration) {
    return ImmutableList.of(
        new PathFragment("."),
        configuration.getGenfilesFragment());
  }

  /**
   * Returns the first directory in the sequence of parents of the exec path of the given artifact
   * that matches {@code type}. For instance, if {@code type} is FileType.of(".foo") and the exec
   * path of {@code artifact} is {@code a/b/c/bar.foo/d/e}, then the return value is
   * {@code a/b/c/bar.foo}.
   */
  static Optional<PathFragment> nearestContainerMatching(FileType type, Artifact artifact) {
    PathFragment container = artifact.getExecPath();
    do {
      if (type.matches(container)) {
        return Optional.of(container);
      }
      container = container.getParentDirectory();
    } while (container != null);
    return Optional.absent();
  }

  /**
   * Similar to {@link #nearestContainerMatching(FileType, Artifact)}, but tries matching several
   * file types in {@code types}, and returns a path for the first match in the sequence.
   */
  static Optional<PathFragment> nearestContainerMatching(
      Iterable<FileType> types, Artifact artifact) {
    for (FileType type : types) {
      for (PathFragment container : nearestContainerMatching(type, artifact).asSet()) {
        return Optional.of(container);
      }
    }
    return Optional.absent();
  }

  /**
   * Returns all directories matching {@code containerType} that contain the items in
   * {@code artifacts}. This function ignores artifacts that are not in any directory matching
   * {@code containerType}.
   */
  static Iterable<PathFragment> uniqueContainers(
      Iterable<Artifact> artifacts, FileType containerType) {
    ImmutableSet.Builder<PathFragment> containers = new ImmutableSet.Builder<>();
    for (Artifact artifact : artifacts) {
      containers.addAll(ObjcCommon.nearestContainerMatching(containerType, artifact).asSet());
    }
    return containers.build();
  }

  /**
   * Similar to {@link #nearestContainerMatching(FileType, Artifact)}, but returns the container
   * closest to the root that matches the given type.
   */
  static Optional<PathFragment> farthestContainerMatching(FileType type, Artifact artifact) {
    PathFragment container = artifact.getExecPath();
    Optional<PathFragment> lastMatch = Optional.absent();
    do {
      if (type.matches(container)) {
        lastMatch = Optional.of(container);
      }
      container = container.getParentDirectory();
    } while (container != null);
    return lastMatch;
  }

  static Iterable<String> notInContainerErrors(
      Iterable<Artifact> artifacts, FileType containerType) {
    return notInContainerErrors(artifacts, ImmutableList.of(containerType));
  }

  @VisibleForTesting
  static final String NOT_IN_CONTAINER_ERROR_FORMAT =
      "File '%s' is not in a directory of one of these type(s): %s";

  static Iterable<String> notInContainerErrors(
      Iterable<Artifact> artifacts, Iterable<FileType> containerTypes) {
    Set<String> errors = new HashSet<>();
    for (Artifact artifact : artifacts) {
      boolean inContainer = nearestContainerMatching(containerTypes, artifact).isPresent();
      if (!inContainer) {
        errors.add(String.format(NOT_IN_CONTAINER_ERROR_FORMAT,
            artifact.getExecPath(), Iterables.toString(containerTypes)));
      }
    }
    return errors;
  }

  /**
   * Returns a {@link RuleConfiguredTargetBuilder}.
   *
   * @param filesToBuild files to build for this target. These also become the data runfiles. Note
   *     that this method may add more files to create the complete list of files to build for this
   *     target.
   * @param maybeTargetProvider the provider for this target.
   * @param maybeExportedProvider the {@link ObjcProvider} for this target. This should generally be
   *     present whenever {@code objc_} rules may depend on this target.
   * @param maybeJ2ObjcSrcsProvider the {@link J2ObjcSrcsProvider} for this target.
   */
  public RuleConfiguredTargetBuilder configuredTargetBuilder(NestedSet<Artifact> filesToBuild,
      Optional<XcodeProvider> maybeTargetProvider, Optional<ObjcProvider> maybeExportedProvider,
      Optional<XcTestAppProvider> maybeXcTestAppProvider,
      Optional<J2ObjcSrcsProvider> maybeJ2ObjcSrcsProvider) {
    NestedSet<Artifact> allFilesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(filesToBuild)
        .build();

    RunfilesProvider runfilesProvider = RunfilesProvider.withData(
        new Runfiles.Builder()
            .addRunfiles(context, RunfilesProvider.DEFAULT_RUNFILES)
            .build(),
        new Runfiles.Builder().addTransitiveArtifacts(allFilesToBuild).build());

    RuleConfiguredTargetBuilder target = new RuleConfiguredTargetBuilder(context)
        .setFilesToBuild(allFilesToBuild)
        .add(RunfilesProvider.class, runfilesProvider);
    for (ObjcProvider exportedProvider : maybeExportedProvider.asSet()) {
      target.addProvider(ObjcProvider.class, exportedProvider);
    }
    for (XcTestAppProvider xcTestAppProvider : maybeXcTestAppProvider.asSet()) {
      target.addProvider(XcTestAppProvider.class, xcTestAppProvider);
    }
    for (XcodeProvider targetProvider : maybeTargetProvider.asSet()) {
      target.addProvider(XcodeProvider.class, targetProvider);
    }
    for (J2ObjcSrcsProvider j2ObjcSrcsProvider : maybeJ2ObjcSrcsProvider.asSet()) {
      target.addProvider(J2ObjcSrcsProvider.class, j2ObjcSrcsProvider);
    }
    return target;
  }

  /**
   * Creates a {@link ConfiguredTarget}.
   *
   * @param filesToBuild files to build for this target. These also become the data runfiles. Note
   *     that this method may add more files to create the complete list of files to build for this
   *     target.
   * @param maybeTargetProvider the provider for this target.
   * @param maybeExportedProvider the {@link ObjcProvider} for this target. This should generally be
   *     present whenever {@code objc_} rules may depend on this target.
   * @param maybeJ2ObjcSrcsProvider the {@link J2ObjcSrcsProvider} for this target.
   */
  public ConfiguredTarget configuredTarget(NestedSet<Artifact> filesToBuild,
      Optional<XcodeProvider> maybeTargetProvider, Optional<ObjcProvider> maybeExportedProvider,
      Optional<XcTestAppProvider> maybeXcTestAppProvider,
      Optional<J2ObjcSrcsProvider> maybeJ2ObjcSrcsProvider) {
    return configuredTargetBuilder(filesToBuild, maybeTargetProvider, maybeExportedProvider,
        maybeXcTestAppProvider, maybeJ2ObjcSrcsProvider).build();
  }
}
