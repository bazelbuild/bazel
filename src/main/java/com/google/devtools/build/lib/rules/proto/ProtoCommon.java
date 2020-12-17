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
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
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

  private static final Interner<PathFragment> PROTO_SOURCE_ROOT_INTERNER =
      BlazeInterners.newWeakInterner();

  /**
   * Returns a memory efficient version of the passed protoSourceRoot.
   *
   * <p>Any sizable proto graph will contain many {@code .proto} sources with the same source root.
   * We can't afford to have all of them represented as individual objects in memory.
   *
   * @param protoSourceRoot
   * @return
   */
  static PathFragment memoryEfficientProtoSourceRoot(PathFragment protoSourceRoot) {
    return PROTO_SOURCE_ROOT_INTERNER.intern(protoSourceRoot);
  }

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
          STABLE_ORDER, Iterables.transform(sources, ProtoSource::getSourceFile));
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
    result.addAll(Iterables.transform(sources, ProtoSource::getSourceFile));
    for (ProtoInfo dep : deps) {
      result.addTransitive(dep.getTransitiveProtoSources());
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
              /* sourceRoot */ memoryEfficientProtoSourceRoot(
                  protoSourceRoot.getRelative(protoSource.getRoot().getExecPath()))));
    }
    return new Library(sources.build(), memoryEfficientProtoSourceRoot(protoSourceRoot));
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
    if (stripImportPrefixAttribute == null) {
      stripImportPrefix = PathFragment.EMPTY_FRAGMENT;
    } else if (stripImportPrefixAttribute.isAbsolute()) {
      stripImportPrefix = stripImportPrefixAttribute.toRelative();
    } else {
      stripImportPrefix =
          ruleContext.getLabel().getPackageFragment().getRelative(stripImportPrefixAttribute);
    }

    PathFragment importPrefix =
        importPrefixAttribute != null ? importPrefixAttribute : PathFragment.EMPTY_FRAGMENT;
    if (importPrefix.isAbsolute()) {
      ruleContext.attributeError("import_prefix", "should be a relative path");
      return null;
    }

    PathFragment sourceRootPath = ruleContext.getUniqueDirectory("_virtual_imports");
    PathFragment sourceRoot =
        memoryEfficientProtoSourceRoot(
            ruleContext.getBinOrGenfilesDirectory().getExecPath().getRelative(sourceRootPath));

    ImmutableList.Builder<ProtoSource> sources = ImmutableList.builder();
    for (Artifact realProtoSource : protoSources) {
      if (!realProtoSource.getRepositoryRelativePath().startsWith(stripImportPrefix)) {
        ruleContext.ruleError(
            String.format(
                ".proto file '%s' is not under the specified strip prefix '%s'",
                realProtoSource.getExecPathString(), stripImportPrefix.getPathString()));
        continue;
      }
      Artifact virtualProtoSource =
          createVirtualProtoSource(
              ruleContext, realProtoSource, sourceRootPath, importPrefix, stripImportPrefix);
      sources.add(
          new ProtoSource(
              /* sourceFile */ virtualProtoSource,
              /* originalSourceFile */ realProtoSource,
              /* sourceRoot */ sourceRoot));
    }
    return new Library(sources.build(), sourceRoot);
  }

  private static Artifact createVirtualProtoSource(
      RuleContext ruleContext,
      Artifact realProtoSource,
      PathFragment sourceRootPath,
      PathFragment importPrefix,
      PathFragment stripImportPrefix) {
    PathFragment importPath =
        importPrefix.getRelative(
            realProtoSource.getRepositoryRelativePath().relativeTo(stripImportPrefix));

    Artifact virtualProtoSource =
        ruleContext.getDerivedArtifact(
            sourceRootPath.getRelative(importPath), ruleContext.getBinOrGenfilesDirectory());

    ruleContext.registerAction(
        SymlinkAction.toArtifact(
            ruleContext.getActionOwner(),
            realProtoSource,
            virtualProtoSource,
            "Symlinking virtual .proto sources for " + ruleContext.getLabel()));

    return virtualProtoSource;
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

  /**
   * Creates the {@link ProtoInfo} for the {@code proto_library} rule associated with {@code
   * ruleContext}.
   */
  public static ProtoInfo createProtoInfo(
      RuleContext ruleContext, boolean generatedProtosInVirtualImports)
      throws InterruptedException {
    ImmutableList<Artifact> originalDirectProtoSources =
        ruleContext.getPrerequisiteArtifacts("srcs").list();
    ImmutableList<ProtoInfo> deps =
        ImmutableList.copyOf(ruleContext.getPrerequisites("deps", ProtoInfo.PROVIDER));
    ImmutableList<ProtoInfo> exports =
        ImmutableList.copyOf(ruleContext.getPrerequisites("exports", ProtoInfo.PROVIDER));

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
              .getRepository()
              .getExecPath(ruleContext.getConfiguration().isSiblingRepositoryLayout());
      library =
          createLibraryWithoutVirtualSourceRoot(contextProtoSourceRoot, originalDirectProtoSources);
    }

    ImmutableList<ProtoSource> directSources = library.getSources();
    PathFragment directProtoSourceRoot = library.getSourceRoot();
    NestedSet<ProtoSource> transitiveSources = computeTransitiveProtoSources(deps, library);
    NestedSet<Artifact> transitiveProtoSources =
        computeTransitiveProtoSourceArtifacts(directSources, deps);
    NestedSet<String> transitiveProtoSourceRoots =
        computeTransitiveProtoSourceRoots(deps, directProtoSourceRoot.getSafePathString());
    NestedSet<Artifact> strictImportableProtosForDependents =
        computeStrictImportableProtosForDependents(directSources, deps);
    Artifact directDescriptorSet =
        ruleContext.getGenfilesArtifact(
            ruleContext.getLabel().getName() + "-descriptor-set.proto.bin");
    NestedSet<Artifact> transitiveDescriptorSets =
        computeTransitiveDescriptorSets(directDescriptorSet, deps);

    // Layering checks.
    NestedSet<ProtoSource> exportedSources = computeExportedProtos(directSources, deps);
    NestedSet<ProtoSource> strictImportableSources =
        computeStrictImportableProtos(directSources, deps);
    NestedSet<ProtoSource> publicImportSources = computePublicImportProtos(exports);

    return new ProtoInfo(
        directSources,
        directProtoSourceRoot,
        transitiveSources,
        transitiveProtoSources,
        transitiveProtoSourceRoots,
        strictImportableProtosForDependents,
        directDescriptorSet,
        transitiveDescriptorSets,
        exportedSources,
        strictImportableSources,
        publicImportSources);
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
   * Each language-specific initialization method will call this to construct Artifacts representing
   * its protocol compiler outputs.
   *
   * @param extension Remove ".proto" and replace it with this to produce the output file name, e.g.
   *     ".pb.cc".
   * @param pythonNames If true, replace hyphens in the file name with underscores, as required for
   *     Python modules.
   */
  public static ImmutableList<Artifact> getGeneratedOutputs(
      RuleContext ruleContext,
      ImmutableList<Artifact> protoSources,
      String extension,
      boolean pythonNames) {
    ImmutableList.Builder<Artifact> outputsBuilder = new ImmutableList.Builder<>();
    ArtifactRoot genfiles = ruleContext.getGenfilesDirectory();
    for (Artifact src : protoSources) {
      PathFragment srcPath =
          src.getOutputDirRelativePath(ruleContext.getConfiguration().isSiblingRepositoryLayout());
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
   * Each language-specific initialization method will call this to construct Artifacts representing
   * its protocol compiler outputs.
   *
   * @param extension Remove ".proto" and replace it with this to produce the output file name, e.g.
   *     ".pb.cc".
   */
  public static ImmutableList<Artifact> getGeneratedOutputs(
      RuleContext ruleContext, ImmutableList<Artifact> protoSources, String extension) {
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
      ImmutableList<ProtoInfo> exports) {
    NestedSetBuilder<ProtoSource> result = NestedSetBuilder.stableOrder();
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
    StrictDepsMode getBool = ruleContext.getFragment(ProtoConfiguration.class).strictProtoDeps();
    return getBool != StrictDepsMode.OFF && getBool != StrictDepsMode.DEFAULT;
  }
}
