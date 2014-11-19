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

import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.ARCHIVES;
import static com.google.devtools.build.lib.rules.objc.ArtifactListAttribute.BUNDLE_IMPORTS;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.ASSET_CATALOG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_IMPORT_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FLAG;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_FOR_XCODEGEN;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FORCE_LOAD_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.Flag.USES_CPP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.GENERAL_RESOURCE_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.INCLUDE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STORYBOARD_OUTPUT_ZIP;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCASSETS_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.XCDATAMODEL;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.HashSet;
import java.util.Set;

/**
 * Contains information common to multiple objc_* rules, and provides a unified API for extracting
 * and accessing it.
 */
final class ObjcCommon {
  static class Builder {
    private RuleContext context;
    private Optional<ObjcBase.Attributes> baseAttributes = Optional.absent();
    private Iterable<SdkFramework> extraSdkFrameworks = ImmutableList.of();
    private Iterable<Artifact> frameworkImports = ImmutableList.of();
    private Optional<CompilationArtifacts> compilationArtifacts = Optional.absent();
    private Iterable<ObjcProvider> depObjcProviders = ImmutableList.of();
    private Iterable<PathFragment> userHeaderSearchPaths = ImmutableList.of();
    private Iterable<Artifact> headers = ImmutableList.of();
    private IntermediateArtifacts intermediateArtifacts;
    private boolean alwayslink;

    Builder(RuleContext context) {
      this.context = Preconditions.checkNotNull(context);
    }

    public Builder setBaseAttributes(ObjcBase.Attributes baseAttributes) {
      Preconditions.checkState(!this.baseAttributes.isPresent(),
          "baseAttributes is already set to: %s", this.baseAttributes);
      this.baseAttributes = Optional.of(baseAttributes);
      return this;
    }

    Builder addExtraSdkFrameworks(Iterable<SdkFramework> extraSdkFrameworks) {
      this.extraSdkFrameworks = Iterables.concat(this.extraSdkFrameworks, extraSdkFrameworks);
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

    Builder addDepObjcProviders(Iterable<ObjcProvider> depObjcProviders) {
      this.depObjcProviders = Iterables.concat(this.depObjcProviders, depObjcProviders);
      return this;
    }

    public Builder addUserHeaderSearchPaths(Iterable<PathFragment> userHeaderSearchPaths) {
      this.userHeaderSearchPaths =
          Iterables.concat(this.userHeaderSearchPaths, userHeaderSearchPaths);
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

    ObjcCommon build() {
      Iterable<BundleableFile> bundleImports = BundleableFile.bundleImportsFromRule(context);

      ObjcProvider.Builder objcProvider = new ObjcProvider.Builder()
          .addAll(IMPORTED_LIBRARY, ARCHIVES.get(context))
          .addAll(BUNDLE_FILE, bundleImports)
          .addAll(BUNDLE_IMPORT_DIR,
              uniqueContainers(BundleableFile.toArtifacts(bundleImports), BUNDLE_CONTAINER_TYPE))
          .addAll(SDK_FRAMEWORK, extraSdkFrameworks)
          .addAll(FRAMEWORK_FILE, frameworkImports)
          .addAll(FRAMEWORK_DIR, uniqueContainers(frameworkImports, FRAMEWORK_CONTAINER_TYPE))
          .addAll(INCLUDE, userHeaderSearchPaths)
          .addAll(HEADER, headers)
          .addTransitive(depObjcProviders);

      Storyboards storyboards;
      if (baseAttributes.isPresent()) {
        ObjcBase.Attributes attributes = baseAttributes.get();
        storyboards = Storyboards.fromInputs(attributes.storyboards(), intermediateArtifacts);
        Iterable<CompiledResourceFile> compiledResources = Iterables.concat(
            CompiledResourceFile.fromXibFiles(intermediateArtifacts, attributes.xibs()),
            CompiledResourceFile.fromStringsFiles(intermediateArtifacts, attributes.strings()));

        objcProvider
            .addAll(HEADER, attributes.hdrs())
            .addAll(INCLUDE, attributes.headerSearchPaths())
            .addAll(XCASSETS_DIR,
                uniqueContainers(attributes.assetCatalogs(), ASSET_CATALOG_CONTAINER_TYPE))
            .addAll(ASSET_CATALOG, attributes.assetCatalogs())
            .addTransitive(STORYBOARD_OUTPUT_ZIP, storyboards.getOutputZips())
            .addAll(GENERAL_RESOURCE_FILE, storyboards.getInputs())
            .addAll(GENERAL_RESOURCE_FILE, attributes.resources())
            .addAll(GENERAL_RESOURCE_FILE, attributes.strings())
            .addAll(GENERAL_RESOURCE_FILE, attributes.xibs())
            .addAll(BUNDLE_FILE, BundleableFile.nonCompiledResourceFiles(attributes.resources()))
            .addAll(BUNDLE_FILE,
                Iterables.transform(compiledResources, CompiledResourceFile.TO_BUNDLED))
            .addAll(SDK_FRAMEWORK, attributes.sdkFrameworks())
            .addAll(WEAK_SDK_FRAMEWORK, attributes.weakSdkFrameworks())
            .addAll(SDK_DYLIB, attributes.sdkDylibs())
            .addAll(XCDATAMODEL,
                Xcdatamodels.xcdatamodels(intermediateArtifacts, attributes.datamodels()));
      } else {
        storyboards = Storyboards.empty();
      }

      for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
        objcProvider.addAll(LIBRARY, artifacts.getArchive().asSet());

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
            objcProvider.add(FORCE_LOAD_FOR_XCODEGEN,
                "$(BUILT_PRODUCTS_DIR)/" + archive.getExecPath().getBaseName());
          }
        }
        for (Artifact archive : ARCHIVES.get(context)) {
          objcProvider.add(FORCE_LOAD_LIBRARY, archive);
          objcProvider.add(FORCE_LOAD_FOR_XCODEGEN,
              "$(WORKSPACE_ROOT)/" + archive.getExecPath().getSafePathString());
        }
      }

      Iterable<String> ruleErrors =
          notInContainerErrors(frameworkImports, FRAMEWORK_CONTAINER_TYPE);

      return new ObjcCommon(context, objcProvider.build(), storyboards, baseAttributes,
          compilationArtifacts, ruleErrors);
    }
  }

  @VisibleForTesting
  static final String ABSOLUTE_INCLUDES_PATH_FORMAT =
      "The path '%s' is absolute, but only relative paths are allowed.";

  static final FileType BUNDLE_CONTAINER_TYPE = FileType.of(".bundle");

  @VisibleForTesting
  static final FileType ASSET_CATALOG_CONTAINER_TYPE = FileType.of(".xcassets");

  @VisibleForTesting
  static final FileType FRAMEWORK_CONTAINER_TYPE = FileType.of(".framework");

  private final RuleContext context;
  private final ObjcProvider objcProvider;
  private final Storyboards storyboards;
  private final Optional<ObjcBase.Attributes> baseAttributes;
  private final Optional<CompilationArtifacts> compilationArtifacts;
  private final Iterable<String> ruleErrors;

  private ObjcCommon(
      RuleContext context,
      ObjcProvider objcProvider,
      Storyboards storyboards,
      Optional<ObjcBase.Attributes> baseAttributes,
      Optional<CompilationArtifacts> compilationArtifacts,
      Iterable<String> ruleErrors) {
    this.context = Preconditions.checkNotNull(context);
    this.objcProvider = Preconditions.checkNotNull(objcProvider);
    this.storyboards = Preconditions.checkNotNull(storyboards);
    this.baseAttributes = Preconditions.checkNotNull(baseAttributes);
    this.compilationArtifacts = Preconditions.checkNotNull(compilationArtifacts);
    this.ruleErrors = Preconditions.checkNotNull(ruleErrors);
  }

  public ObjcProvider getObjcProvider() {
    return objcProvider;
  }

  public Iterable<Artifact> getHdrs() {
    return baseAttributes.isPresent() ? baseAttributes.get().hdrs() : ImmutableList.<Artifact>of();
  }

  public Optional<CompilationArtifacts> getCompilationArtifacts() {
    return compilationArtifacts;
  }

  /**
   * Returns all storyboards declared in this rule (not including others in the transitive
   * dependency tree).
   */
  public Storyboards getStoryboards() {
    return storyboards;
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
    for (String error : ruleErrors) {
      context.ruleError(error);
    }

    for (String error : notInContainerErrors(BUNDLE_IMPORTS.get(context), BUNDLE_CONTAINER_TYPE)) {
      context.attributeError(BUNDLE_IMPORTS.attrName(), error);
    }

    for (ObjcBase.Attributes attributes : baseAttributes.asSet()) {
      for (String error :
          notInContainerErrors(attributes.datamodels(), Xcdatamodels.CONTAINER_TYPES)) {
        context.attributeError("datamodels", error);
      }

      for (String error :
          notInContainerErrors(attributes.assetCatalogs(), ASSET_CATALOG_CONTAINER_TYPE)) {
        context.attributeError("asset_catalogs", error);
      }

      for (PathFragment absoluteInclude :
          Iterables.filter(attributes.includes(), PathFragment.IS_ABSOLUTE)) {
        context.attributeError(
            "includes", String.format(ABSOLUTE_INCLUDES_PATH_FORMAT, absoluteInclude));
      }
    }

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

  private NestedSet<Artifact> inputsToLegacyRules() {
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(objcProvider.allArtifactsForObjcFilegroup());

    for (CompilationArtifacts artifacts : compilationArtifacts.asSet()) {
      inputs
          .addAll(artifacts.getNonArcSrcs())
          .addAll(artifacts.getSrcs())
          .addAll(artifacts.getPchFile().asSet());
    }

    return inputs.build();
  }

  /**
   * @param filesToBuild files to build for this target. These also become the data runfiles. Note
   *     that this method may add more files to create the complete list of files to build for this
   *     target.
   * @param maybeTargetProvider the provider for this target.
   * @param maybeExportedProvider the {@link ObjcProvider} for this target. This should generally be
   *     present whenever {@code objc_} rules may depend on this target.
   */
  public ConfiguredTarget configuredTarget(NestedSet<Artifact> filesToBuild,
      Optional<XcodeProvider> maybeTargetProvider, Optional<ObjcProvider> maybeExportedProvider) {
    NestedSet<Artifact> allFilesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addTransitive(filesToBuild)
        .addTransitive(storyboards.getOutputZips())
        .build();

    RunfilesProvider runfilesProvider = RunfilesProvider.withData(
        new Runfiles.Builder()
            .addRunfiles(context, RunfilesProvider.DEFAULT_RUNFILES)
            .build(),
        new Runfiles.Builder().addArtifacts(allFilesToBuild).build());

    RuleConfiguredTargetBuilder target = new RuleConfiguredTargetBuilder(context)
        .setFilesToBuild(allFilesToBuild)
        .add(RunfilesProvider.class, runfilesProvider)
        // TODO(bazel-team): Remove this when legacy dependencies have been removed.
        .addProvider(LegacyObjcSourceFileProvider.class,
            new LegacyObjcSourceFileProvider(inputsToLegacyRules()));
    for (ObjcProvider exportedProvider : maybeExportedProvider.asSet()) {
      target.addProvider(ObjcProvider.class, exportedProvider);
    }
    for (XcodeProvider targetProvider : maybeTargetProvider.asSet()) {
      target.addProvider(XcodeProvider.class, targetProvider);
    }
    return target.build();
  }
}
