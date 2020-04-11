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

package com.google.devtools.build.lib.rules.proto;

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Utility functions for proto_library and proto aspect implementations.
 */
public class ProtoCommon {
  private ProtoCommon() {
    throw new UnsupportedOperationException();
  }

  // Keep in sync with the migration label in
  // https://github.com/bazelbuild/rules_proto/blob/master/proto/defs.bzl.
  @VisibleForTesting
  public static final String PROTO_RULES_MIGRATION_LABEL =
      "__PROTO_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__";

  /**
   * Gets the direct sources of a proto library. If protoSources is not empty, the value is just
   * protoSources. Otherwise, it's the combined sources of all direct dependencies of the given
   * RuleContext.
   *
   * @param sources the direct proto sources.
   * @param deps the proto dependencies.
   * @return the direct sources of a proto library.
   */
  private static NestedSet<Artifact> computeStrictImportableProtosForDependents(
      ImmutableList<ProtoSource> sources, ImmutableList<ProtoInfo> deps) {

    if (sources.isEmpty()) {
      /* a proxy/alias library, return the sources of the direct deps */
      NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
      for (ProtoInfo provider : deps) {
        builder.addTransitive(provider.getStrictImportableProtoSourcesForDependents());
      }
      return builder.build();
    } else {
      return NestedSetBuilder.wrap(
          STABLE_ORDER, Iterables.transform(sources, s -> s.getSourceFile()));
    }
  }

  private static NestedSet<ProtoSource> computeExportedProtos(
      ImmutableList<ProtoSource> directSources, ImmutableList<ProtoInfo> deps) {
    if (!directSources.isEmpty()) {
      return NestedSetBuilder.wrap(STABLE_ORDER, directSources);
    }

    /* a proxy/alias library, return the sources of the direct deps */
    NestedSetBuilder<ProtoSource> builder = NestedSetBuilder.stableOrder();
    for (ProtoInfo provider : deps) {
      builder.addTransitive(provider.getExportedSources());
    }
    return builder.build();
  }

  /**
   * Gets the direct sources and import paths of a proto library. If protoSourcesImportPaths is not
   * empty, the value is just protoSourcesImportPaths. Otherwise, it's the combined sources of all
   * direct dependencies of the given RuleContext.
   *
   * @param sources the direct proto sources.
   * @param deps the proto dependencies.
   * @return the direct sources and import paths of a proto library.
   */
  private static NestedSet<Pair<Artifact, String>>
      computeStrictImportableProtosImportPathsForDependents(
          ImmutableList<ProtoSource> sources, ImmutableList<ProtoInfo> deps) {
    if (sources.isEmpty()) {
      /* a proxy/alias library, return the sources of the direct deps */
      NestedSetBuilder<Pair<Artifact, String>> builder = NestedSetBuilder.stableOrder();
      for (ProtoInfo provider : deps) {
        builder.addTransitive(provider.getStrictImportableProtoSourcesImportPathsForDependents());
      }
      return builder.build();
    } else {
      return NestedSetBuilder.wrap(
          STABLE_ORDER, Iterables.transform(sources, s -> toProtoImportPathPair(s)));
    }
  }

  private static Pair<Artifact, String> toProtoImportPathPair(ProtoSource source) {
    Optional<PathFragment> importPath =
        source.getImportPathForStrictImportableProtosImportPathsForDependents();
    if (importPath.isPresent()) {
      return new Pair<>(source.getOriginalSourceFile(), importPath.get().toString());
    }
    return new Pair<>(source.getOriginalSourceFile(), null);
  }

  private static NestedSet<ProtoSource> computeTransitiveProtoSources(
      ImmutableList<ProtoInfo> protoDeps, Library library) {
    NestedSetBuilder<ProtoSource> result = NestedSetBuilder.naiveLinkOrder();
    result.addAll(library.getSources());
    for (ProtoInfo dep : protoDeps) {
      result.addTransitive(dep.getTransitiveSources());
    }
    return result.build();
  }

  /**
   * Collects all .proto files in this lib and its transitive dependencies.
   *
   * <p>Each import is a Artifact/Label pair.
   */
  private static NestedSet<Artifact> computeTransitiveProtoSourceArtifacts(
      ImmutableList<ProtoSource> sources, ImmutableList<ProtoInfo> deps) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.naiveLinkOrder();
    result.addAll(Iterables.transform(sources, s -> s.getSourceFile()));
    for (ProtoInfo dep : deps) {
      result.addTransitive(dep.getTransitiveProtoSources());
    }
    return result.build();
  }

  private static NestedSet<Artifact> computeTransitiveOriginalProtoSources(
      ImmutableList<ProtoInfo> protoDeps, ImmutableList<Artifact> originalProtoSources) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.naiveLinkOrder();

    result.addAll(originalProtoSources);

    for (ProtoInfo dep : protoDeps) {
      result.addTransitive(dep.getOriginalTransitiveProtoSources());
    }

    return result.build();
  }

  static NestedSet<Artifact> computeDependenciesDescriptorSets(ImmutableList<ProtoInfo> deps) {
    return computeTransitiveDescriptorSets(null, deps);
  }

  private static NestedSet<Artifact> computeTransitiveDescriptorSets(
      @Nullable Artifact directDescriptorSet, ImmutableList<ProtoInfo> deps) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();
    if (directDescriptorSet != null) {
      result.add(directDescriptorSet);
    }
    for (ProtoInfo dep : deps) {
      result.addTransitive(dep.getTransitiveDescriptorSets());
    }
    return result.build();
  }

  /**
   * Returns all proto source roots in this lib ({@code currentProtoSourceRoot}) and in its
   * transitive dependencies.
   *
   * <p>Assumes {@code currentProtoSourceRoot} is the same as the package name.
   */
  private static NestedSet<String> computeTransitiveProtoSourceRoots(
      ImmutableList<ProtoInfo> protoDeps, String currentProtoSourceRoot) {
    NestedSetBuilder<String> protoPath = NestedSetBuilder.stableOrder();

    protoPath.add(currentProtoSourceRoot);
    for (ProtoInfo provider : protoDeps) {
      protoPath.addTransitive(provider.getTransitiveProtoSourceRoots());
    }

    return protoPath.build();
  }

  /** Basically a {@link Pair}. */
  private static final class Library {
    private final ImmutableList<ProtoSource> sources;
    private final PathFragment sourceRoot;

    Library(ImmutableList<ProtoSource> sources, PathFragment sourceRoot) {
      this.sources = sources;
      this.sourceRoot = sourceRoot;
    }

    public ImmutableList<ProtoSource> getSources() {
      return sources;
    }

    public PathFragment getSourceRoot() {
      return sourceRoot;
    }
  }

  /**
   * Returns the {@link Library} representing this <code>proto_library</code> rule.
   *
   * <p>Assumes that <code>strip_import_prefix</code> and <code>import_prefix</code> are unset and
   * that there are no generated .proto files that need to be compiled.
   */
  @Nullable
  public static Library createLibraryWithoutVirtualSourceRoot(
      PathFragment protoSourceRoot, ImmutableList<Artifact> directSources) {
    ImmutableList.Builder<ProtoSource> sources = ImmutableList.builder();
    for (Artifact protoSource : directSources) {
      sources.add(
          new ProtoSource(
              /* sourceFile */ protoSource,
              /* sourceRoot */ protoSource.getRoot().getExecPath(),
              /* importPath */ Optional.empty()));
    }
    return new Library(sources.build(), protoSourceRoot);
  }

  private static PathFragment getPathFragmentAttribute(
      RuleContext ruleContext, String attributeName) {
    if (!ruleContext.attributes().has(attributeName)) {
      return null;
    }

    if (!ruleContext.attributes().isAttributeValueExplicitlySpecified(attributeName)) {
      return null;
    }

    String asString = ruleContext.attributes().get(attributeName, STRING);
    if (!PathFragment.isNormalized(asString)) {
      ruleContext.attributeError(
          attributeName, "should be normalized (without uplevel references or '.' path segments)");
      return null;
    }

    return PathFragment.create(asString);
  }

  /**
   * Returns the {@link Library} representing this <code>proto_library</code> rule if import prefix
   * munging is done. Otherwise, returns null.
   */
  private static Library createLibraryWithVirtualSourceRootMaybe(
      RuleContext ruleContext,
      ImmutableList<Artifact> protoSources,
      boolean generatedProtosInVirtualImports)
      throws InterruptedException {
    PathFragment importPrefixAttribute = getPathFragmentAttribute(ruleContext, "import_prefix");
    PathFragment stripImportPrefixAttribute =
        getPathFragmentAttribute(ruleContext, "strip_import_prefix");
    boolean hasGeneratedSources = false;

    if (generatedProtosInVirtualImports) {
      for (Artifact protoSource : protoSources) {
        if (!protoSource.isSourceArtifact()) {
          hasGeneratedSources = true;
          break;
        }
      }
    }

    if (importPrefixAttribute == null
        && stripImportPrefixAttribute == null
        && !hasGeneratedSources) {
      // Simple case, no magic required.
      return null;
    }

    PathFragment stripImportPrefix;
    PathFragment importPrefix;

    StarlarkSemantics starlarkSemantics =
        ruleContext.getAnalysisEnvironment().getSkylarkSemantics();
    boolean siblingRepositoryLayout = starlarkSemantics.experimentalSiblingRepositoryLayout();
    if (stripImportPrefixAttribute != null || importPrefixAttribute != null) {
      if (stripImportPrefixAttribute == null) {
        stripImportPrefix =
            PathFragment.create(ruleContext.getLabel().getWorkspaceRoot(starlarkSemantics));
      } else if (stripImportPrefixAttribute.isAbsolute()) {
        stripImportPrefix =
            ruleContext
                .getLabel()
                .getPackageIdentifier()
                .getRepository()
                .getExecPath(siblingRepositoryLayout)
                .getRelative(stripImportPrefixAttribute.toRelative());
      } else {
        stripImportPrefix =
            ruleContext
                .getLabel()
                .getPackageIdentifier()
                .getExecPath(siblingRepositoryLayout)
                .getRelative(stripImportPrefixAttribute);
      }

      if (importPrefixAttribute != null) {
        importPrefix = importPrefixAttribute;
      } else {
        importPrefix = PathFragment.EMPTY_FRAGMENT;
      }

      if (importPrefix.isAbsolute()) {
        ruleContext.attributeError("import_prefix", "should be a relative path");
        return null;
      }
    } else {
      // Has generated sources, but neither strip_import_prefix nor import_prefix
      stripImportPrefix =
          ruleContext
              .getLabel()
              .getPackageIdentifier()
              .getRepository()
              .getDerivedArtifactSourceRoot();

      importPrefix = PathFragment.EMPTY_FRAGMENT;
    }

    PathFragment sourceRootPath = ruleContext.getUniqueDirectory("_virtual_imports");
    PathFragment sourceRoot =
        ruleContext.getBinOrGenfilesDirectory().getExecPath().getRelative(sourceRootPath);

    ImmutableList.Builder<ProtoSource> sources = ImmutableList.builder();
    for (Artifact realProtoSource : protoSources) {
      if (siblingRepositoryLayout && realProtoSource.isSourceArtifact()
          ? !realProtoSource.getExecPath().startsWith(stripImportPrefix)
          : !realProtoSource.getRootRelativePath().startsWith(stripImportPrefix)) {
        ruleContext.ruleError(
            String.format(
                ".proto file '%s' is not under the specified strip prefix '%s'",
                realProtoSource.getExecPathString(), stripImportPrefix.getPathString()));
        continue;
      }
      Pair<PathFragment, Artifact> importsPair =
          computeImports(
              ruleContext,
              realProtoSource,
              sourceRootPath,
              importPrefix,
              stripImportPrefix,
              starlarkSemantics.experimentalSiblingRepositoryLayout());
      sources.add(
          new ProtoSource(
              /* sourceFile */ importsPair.second,
              /* originalSourceFile */ realProtoSource,
              /* sourceRoot */ sourceRoot,
              /* importPath */ Optional.of(importsPair.first)));
    }
    return new Library(sources.build(), sourceRoot);
  }

  private static Pair<PathFragment, Artifact> computeImports(
      RuleContext ruleContext,
      Artifact realProtoSource,
      PathFragment sourceRootPath,
      PathFragment importPrefix,
      PathFragment stripImportPrefix,
      boolean siblingRepositoryLayout) {
    PathFragment importPath;

    if (siblingRepositoryLayout && realProtoSource.isSourceArtifact()) {
      importPath =
          importPrefix.getRelative(realProtoSource.getExecPath().relativeTo(stripImportPrefix));
    } else {
      importPath =
          importPrefix.getRelative(
              realProtoSource.getRootRelativePath().relativeTo(stripImportPrefix));
    }

    Artifact virtualProtoSource =
        ruleContext.getDerivedArtifact(
            sourceRootPath.getRelative(importPath), ruleContext.getBinOrGenfilesDirectory());

      ruleContext.registerAction(
          SymlinkAction.toArtifact(
              ruleContext.getActionOwner(),
              realProtoSource,
              virtualProtoSource,
              "Symlinking virtual .proto sources for " + ruleContext.getLabel()));

    return Pair.of(importPath, virtualProtoSource);
  }

  /**
   * Returns a set of the {@code proto_source_root} collected from the current library and the
   * specified attribute.
   *
   * <p>Assumes {@code currentProtoSourceRoot} is the same as the package name.
   */
  private static NestedSet<String> getProtoSourceRootsOfAttribute(
      ImmutableList<ProtoInfo> protoInfos, String currentProtoSourceRoot) {
    NestedSetBuilder<String> protoSourceRoots = NestedSetBuilder.stableOrder();
    protoSourceRoots.add(currentProtoSourceRoot);

    for (ProtoInfo provider : protoInfos) {
      protoSourceRoots.add(provider.getDirectProtoSourceRoot());
    }

    return protoSourceRoots.build();
  }

  /**
   * Check that .proto files in sources are from the same package. This is done to avoid clashes
   * with the generated sources.
   */
  public static void checkSourceFilesAreInSamePackage(RuleContext ruleContext) {
    // TODO(bazel-team): this does not work with filegroups that contain files
    // that are not in the package
    for (Label source : ruleContext.attributes().get("srcs", BuildType.LABEL_LIST)) {
      if (!isConfiguredTargetInSamePackage(ruleContext, source)) {
        ruleContext.attributeError(
            "srcs",
            "Proto source with label '" + source + "' must be in same package as consuming rule.");
      }
    }
  }

  private static boolean isConfiguredTargetInSamePackage(RuleContext ruleContext, Label source) {
    return ruleContext.getLabel().getPackageIdentifier().equals(source.getPackageIdentifier());
  }

  public static void checkRuleHasValidMigrationTag(RuleContext ruleContext) {
    if (!ruleContext.getFragment(ProtoConfiguration.class).loadProtoRulesFromBzl()) {
      return;
    }

    if (!hasValidMigrationTag(ruleContext)) {
      ruleContext.ruleError(
          "The native Protobuf rules are deprecated. Please load "
              + ruleContext.getRule().getRuleClass()
              + " from the rules_proto repository."
              + " See http://github.com/bazelbuild/rules_proto.");
    }
  }

  private static boolean hasValidMigrationTag(RuleContext ruleContext) {
    return ruleContext
        .attributes()
        .get("tags", Type.STRING_LIST)
        .contains(PROTO_RULES_MIGRATION_LABEL);
  }

  /**
   * Creates the {@link ProtoInfo} for the {@code proto_library} rule associated with {@code
   * ruleContext}.
   */
  public static ProtoInfo createProtoInfo(
      RuleContext ruleContext, boolean generatedProtosInVirtualImports)
      throws InterruptedException {
    ImmutableList<Artifact> originalDirectProtoSources =
        ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();
    ImmutableList<ProtoInfo> deps =
        ImmutableList.copyOf(ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoInfo.PROVIDER));
    ImmutableList<ProtoInfo> exports =
        ImmutableList.copyOf(
            ruleContext.getPrerequisites("exports", Mode.TARGET, ProtoInfo.PROVIDER));

    Library library =
        createLibraryWithVirtualSourceRootMaybe(
            ruleContext, originalDirectProtoSources, generatedProtosInVirtualImports);
    if (ruleContext.hasErrors()) {
      return null;
    }

    if (library == null) {
      PathFragment contextProtoSourceRoot =
          ruleContext
              .getLabel()
              .getPackageIdentifier()
              .getRepository()
              .getExecPath(
                  ruleContext
                      .getAnalysisEnvironment()
                      .getSkylarkSemantics()
                      .experimentalSiblingRepositoryLayout());
      library =
          createLibraryWithoutVirtualSourceRoot(contextProtoSourceRoot, originalDirectProtoSources);
    }

    // Direct.
    ImmutableList<ProtoSource> directSources = library.getSources();
    Artifact directDescriptorSet =
        ruleContext.getGenfilesArtifact(
            ruleContext.getLabel().getName() + "-descriptor-set.proto.bin");
    PathFragment directProtoSourceRoot = library.getSourceRoot();

    // Transitive.
    NestedSet<ProtoSource> transitiveSources = computeTransitiveProtoSources(deps, library);
    NestedSet<Artifact> transitiveDescriptorSets =
        computeTransitiveDescriptorSets(directDescriptorSet, deps);
    NestedSet<Artifact> transitiveProtoSources =
        computeTransitiveProtoSourceArtifacts(directSources, deps);
    NestedSet<Artifact> transitiveOriginalProtoSources =
        computeTransitiveOriginalProtoSources(deps, originalDirectProtoSources);
    NestedSet<String> transitiveProtoSourceRoots =
        computeTransitiveProtoSourceRoots(deps, directProtoSourceRoot.getSafePathString());

    // Layering checks.
    NestedSet<ProtoSource> exportedSources = computeExportedProtos(directSources, deps);
    NestedSet<ProtoSource> strictImportableSources =
        computeStrictImportableProtos(directSources, deps);
    NestedSet<ProtoSource> publicImportSources = computePublicImportProtos(directSources, exports);

    // Misc (deprecated).
    NestedSet<Artifact> strictImportableProtoSourcesForDependents =
        computeStrictImportableProtosForDependents(directSources, deps);
    NestedSet<Pair<Artifact, String>> strictImportableProtoSourcesImportPathsForDependents =
        computeStrictImportableProtosImportPathsForDependents(directSources, deps);

    return new ProtoInfo(
        directSources,
        directDescriptorSet,
        directProtoSourceRoot,
        transitiveSources,
        transitiveDescriptorSets,
        transitiveProtoSources,
        transitiveOriginalProtoSources,
        transitiveProtoSourceRoots,
        exportedSources,
        strictImportableSources,
        publicImportSources,
        strictImportableProtoSourcesForDependents,
        strictImportableProtoSourcesImportPathsForDependents,
        Location.BUILTIN);
  }

  public static Runfiles.Builder createDataRunfilesProvider(
      final NestedSet<Artifact> transitiveProtoSources, RuleContext ruleContext) {
    // We assume that the proto sources will not have conflicting artifacts
    // with the same root relative path
    return new Runfiles.Builder(
            ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
        .addTransitiveArtifactsWrappedInStableOrder(transitiveProtoSources);
  }

  // =================================================================
  // Protocol compiler invocation stuff.

  /**
   * Each language-specific initialization method will call this to construct
   * Artifacts representing its protocol compiler outputs.
   *
   * @param extension Remove ".proto" and replace it with this to produce
   *                  the output file name, e.g. ".pb.cc".
   * @param pythonNames If true, replace hyphens in the file name
   *              with underscores, as required for Python modules.
   */
  public static ImmutableList<Artifact> getGeneratedOutputs(RuleContext ruleContext,
      ImmutableList<Artifact> protoSources, String extension, boolean pythonNames) {
    ImmutableList.Builder<Artifact> outputsBuilder = new ImmutableList.Builder<>();
    ArtifactRoot genfiles =
        ruleContext.getConfiguration().getGenfilesDirectory(ruleContext.getRule().getRepository());
    for (Artifact src : protoSources) {
      PathFragment srcPath = src.getRootRelativePath();
      if (pythonNames) {
        srcPath = srcPath.replaceName(srcPath.getBaseName().replace('-', '_'));
      }
      // Note that two proto_library rules can have the same source file, so this is actually a
      // shared action. NB: This can probably result in action conflicts if the proto_library rules
      // are not the same.
      outputsBuilder.add(
          ruleContext.getShareableArtifact(FileSystemUtils.replaceExtension(srcPath, extension),
              genfiles));
    }
    return outputsBuilder.build();
  }

  /**
   * Each language-specific initialization method will call this to construct
   * Artifacts representing its protocol compiler outputs.
   *
   * @param extension Remove ".proto" and replace it with this to produce
   *                  the output file name, e.g. ".pb.cc".
   */
  public static ImmutableList<Artifact> getGeneratedOutputs(RuleContext ruleContext,
      ImmutableList<Artifact> protoSources, String extension) {
    return getGeneratedOutputs(ruleContext, protoSources, extension, false);
  }

  public static ImmutableList<Artifact> getGeneratedTreeArtifactOutputs(
      RuleContext ruleContext, ImmutableList<Artifact> protoSources, PathFragment directory) {
    ImmutableList.Builder<Artifact> outputsBuilder = new ImmutableList.Builder<>();
    if (!protoSources.isEmpty()) {
      ArtifactRoot genfiles =
          ruleContext
              .getConfiguration()
              .getGenfilesDirectory(ruleContext.getRule().getRepository());
      outputsBuilder.add(ruleContext.getTreeArtifact(directory, genfiles));
    }
    return outputsBuilder.build();
  }

  private static NestedSet<ProtoSource> computeStrictImportableProtos(
      ImmutableList<ProtoSource> directSources, ImmutableList<ProtoInfo> deps) {
    NestedSetBuilder<ProtoSource> builder = NestedSetBuilder.stableOrder();
    if (!directSources.isEmpty()) {
      builder.addAll(directSources);
      for (ProtoInfo provider : deps) {
        builder.addTransitive(provider.getExportedSources());
      }
    }
    return builder.build();
  }

  /**
   * Returns the .proto files that are the direct srcs of the exported dependencies of this rule.
   */
  private static NestedSet<ProtoSource> computePublicImportProtos(
      ImmutableList<ProtoSource> directSources, ImmutableList<ProtoInfo> exports) {
    NestedSetBuilder<ProtoSource> result = NestedSetBuilder.stableOrder();
    result.addAll(directSources);
    for (ProtoInfo export : exports) {
      result.addTransitive(export.getExportedSources());
    }
    return result.build();
  }

  /**
   * Decides whether this proto_library should check for strict proto deps.
   *
   * <p>Only takes into account the command-line flag --strict_proto_deps.
   */
  @VisibleForTesting
  public static boolean areDepsStrict(RuleContext ruleContext) {
    StrictDepsMode flagValue = ruleContext.getFragment(ProtoConfiguration.class).strictProtoDeps();
    return flagValue != StrictDepsMode.OFF && flagValue != StrictDepsMode.DEFAULT;
  }
}
