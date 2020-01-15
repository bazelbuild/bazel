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

import static com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode.TARGET;
import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.StrictDepsMode;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.Type;
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

  /**
   * Gets the direct sources of a proto library. If protoSources is not empty, the value is just
   * protoSources. Otherwise, it's the combined sources of all direct dependencies of the given
   * RuleContext.
   *
   * @param ruleContext the proto library rule context.
   * @param protoSources the direct proto sources.
   * @return the direct sources of a proto library.
   */
  private static NestedSet<Artifact> computeStrictImportableProtosForDependents(
      RuleContext ruleContext, ImmutableList<Artifact> protoSources) {

    if (protoSources.isEmpty()) {
      /* a proxy/alias library, return the sources of the direct deps */
      NestedSetBuilder<Artifact> builder = NestedSetBuilder.stableOrder();
      for (TransitiveInfoCollection provider : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
        ProtoInfo sources = provider.get(ProtoInfo.PROVIDER);
        if (sources != null) {
          builder.addTransitive(sources.getStrictImportableProtoSourcesForDependents());
        }
      }
      return builder.build();
    } else {
      return NestedSetBuilder.wrap(STABLE_ORDER, protoSources);
    }
  }

  /**
   * Gets the direct sources and import paths of a proto library. If protoSourcesImportPaths is not
   * empty, the value is just protoSourcesImportPaths. Otherwise, it's the combined sources of all
   * direct dependencies of the given RuleContext.
   *
   * @param ruleContext the proto library rule context.
   * @param protoSourcesImportPaths the direct proto sources.
   * @return the direct sources and import paths of a proto library.
   */
  private static NestedSet<Pair<Artifact, String>>
      computeStrictImportableProtosImportPathsForDependents(
          RuleContext ruleContext, ImmutableList<Pair<Artifact, String>> protoSourcesImportPaths) {

    if (protoSourcesImportPaths.isEmpty()) {
      /* a proxy/alias library, return the sources of the direct deps */
      NestedSetBuilder<Pair<Artifact, String>> builder = NestedSetBuilder.stableOrder();
      for (TransitiveInfoCollection provider : ruleContext.getPrerequisites("deps", Mode.TARGET)) {
        ProtoInfo sources = provider.get(ProtoInfo.PROVIDER);
        if (sources != null) {
          builder.addTransitive(sources.getStrictImportableProtoSourcesImportPathsForDependents());
        }
      }
      return builder.build();
    } else {
      return NestedSetBuilder.wrap(STABLE_ORDER, protoSourcesImportPaths);
    }
  }

  /**
   * Collects all .proto files in this lib and its transitive dependencies.
   *
   * <p>Each import is a Artifact/Label pair.
   */
  private static NestedSet<Artifact> computeTransitiveProtoSources(
      RuleContext ruleContext, ImmutableList<Artifact> protoSources) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.naiveLinkOrder();

    result.addAll(protoSources);

    for (ProtoInfo dep : ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoInfo.PROVIDER)) {
      result.addTransitive(dep.getTransitiveProtoSources());
    }

    return result.build();
  }

  private static NestedSet<Artifact> computeTransitiveOriginalProtoSources(
      RuleContext ruleContext, ImmutableList<Artifact> originalProtoSources) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.naiveLinkOrder();

    result.addAll(originalProtoSources);

    for (ProtoInfo dep : ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoInfo.PROVIDER)) {
      result.addTransitive(dep.getOriginalTransitiveProtoSources());
    }

    return result.build();
  }

  static NestedSet<Artifact> computeDependenciesDescriptorSets(RuleContext ruleContext) {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();

    for (ProtoInfo provider :
        ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoInfo.PROVIDER)) {
      result.addTransitive(provider.getTransitiveDescriptorSets());
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
      RuleContext ruleContext, String currentProtoSourceRoot) {
    NestedSetBuilder<String> protoPath = NestedSetBuilder.stableOrder();

    protoPath.add(currentProtoSourceRoot);
    for (ProtoInfo provider :
        ruleContext.getPrerequisites("deps", Mode.TARGET, ProtoInfo.PROVIDER)) {
      protoPath.addTransitive(provider.getTransitiveProtoSourceRoots());
    }

    return protoPath.build();
  }

  /**
   * The set of .proto files in a single <code>proto_library</code> rule.
   *
   * <p>In addition to the artifacts of the .proto files, this also includes the proto source root
   * so that rules depending on this know how to include them.
   */
  // TODO(lberki): Would be nice if had these in ProtoInfo instead of that haphazard set of fields
  // Unfortunately, ProtoInfo has a Starlark interface so that requires a migration.
  static final class Library {
    private final ImmutableList<Artifact> sources;
    private final ImmutableList<Pair<Artifact, String>> importPathSourcePair;
    private final String sourceRoot;

    Library(
        ImmutableList<Artifact> sources,
        String sourceRoot,
        ImmutableList<Pair<Artifact, String>> importPathSourcePair) {
      this.sources = sources;
      this.sourceRoot = sourceRoot;
      this.importPathSourcePair = importPathSourcePair;
    }

    public ImmutableList<Artifact> getSources() {
      return sources;
    }

    public ImmutableList<Pair<Artifact, String>> getImportPathSourcePair() {
      return importPathSourcePair;
    }

    public String getSourceRoot() {
      return sourceRoot;
    }
  }

  /**
   * Returns the {@link Library} representing this <code>proto_library</code> rule.
   *
   * <p>Assumes that <code>strip_import_prefix</code> and <code>import_prefix</code> are unset and
   * that there are no generated .proto files that need to be compiled.
   */
  // TODO(lberki): This should really be a PathFragment. Unfortunately, it's on the Starlark API of
  // ProtoInfo so it's not an easy change :(
  @Nullable
  private static Library createLibraryWithoutVirtualSourceRoot(
      RuleContext ruleContext, ImmutableList<Artifact> directSources) {
    String protoSourceRoot =
        ruleContext
            .getLabel()
            .getPackageIdentifier()
            .getRepository()
            .getPathUnderExecRoot()
            .getPathString();

    ImmutableList.Builder<Pair<Artifact, String>> builder = ImmutableList.builder();
    for (Artifact protoSource : directSources) {
      builder.add(new Pair<Artifact, String>(protoSource, null));
    }
    return new Library(
        directSources, protoSourceRoot.isEmpty() ? "." : protoSourceRoot, builder.build());
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
      boolean generatedProtosInVirtualImports) {
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

    if (stripImportPrefixAttribute != null || importPrefixAttribute != null) {
      if (stripImportPrefixAttribute == null) {
        stripImportPrefix = PathFragment.create(ruleContext.getLabel().getWorkspaceRoot());
      } else if (stripImportPrefixAttribute.isAbsolute()) {
        stripImportPrefix =
            ruleContext
                .getLabel()
                .getPackageIdentifier()
                .getRepository()
                .getSourceRoot()
                .getRelative(stripImportPrefixAttribute.toRelative());
      } else {
        stripImportPrefix =
            ruleContext.getPackageDirectory().getRelative(stripImportPrefixAttribute);
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
          ruleContext.getLabel().getPackageIdentifier().getRepository().getPathUnderExecRoot();

      importPrefix = PathFragment.EMPTY_FRAGMENT;
    }

    ImmutableList.Builder<Artifact> symlinks = ImmutableList.builder();
    ImmutableList.Builder<Pair<Artifact, String>> protoSourceImportPair = ImmutableList.builder();

    PathFragment sourceRootPath = ruleContext.getUniqueDirectory("_virtual_imports");

    for (Artifact realProtoSource : protoSources) {
      if (!realProtoSource.getRootRelativePath().startsWith(stripImportPrefix)) {
        ruleContext.ruleError(
            String.format(
                ".proto file '%s' is not under the specified strip prefix '%s'",
                realProtoSource.getExecPathString(), stripImportPrefix.getPathString()));
        continue;
      }
      Pair<PathFragment, Artifact> importsPair =
          computeImports(
              ruleContext, realProtoSource, sourceRootPath, importPrefix, stripImportPrefix);
      protoSourceImportPair.add(new Pair<>(realProtoSource, importsPair.first.toString()));
      symlinks.add(importsPair.second);
    }

    String sourceRoot =
        ruleContext
            .getBinOrGenfilesDirectory()
            .getExecPath()
            .getRelative(sourceRootPath)
            .getPathString();
    return new Library(symlinks.build(), sourceRoot, protoSourceImportPair.build());
  }

  private static Pair<PathFragment, Artifact> computeImports(
      RuleContext ruleContext,
      Artifact realProtoSource,
      PathFragment sourceRootPath,
      PathFragment importPrefix,
      PathFragment stripImportPrefix) {
    PathFragment importPath =
        importPrefix.getRelative(
            realProtoSource.getRootRelativePath().relativeTo(stripImportPrefix));

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
      RuleContext ruleContext, String currentProtoSourceRoot, String attributeName) {
    NestedSetBuilder<String> protoSourceRoots = NestedSetBuilder.stableOrder();
    protoSourceRoots.add(currentProtoSourceRoot);

    for (ProtoInfo provider :
        ruleContext.getPrerequisites(attributeName, Mode.TARGET, ProtoInfo.PROVIDER)) {
      protoSourceRoots.add(provider.getDirectProtoSourceRoot());
    }

    return protoSourceRoots.build();
  }

  /**
   * Returns a set of the {@code proto_source_root} collected from the current library and the
   * direct dependencies.
   *
   * <p>Assumes {@code currentProtoSourceRoot} is the same as the package name.
   */
  private static NestedSet<String> computeStrictImportableProtoSourceRoots(
      RuleContext ruleContext, String currentProtoSourceRoot) {
    return getProtoSourceRootsOfAttribute(ruleContext, currentProtoSourceRoot, "deps");
  }

  /**
   * Returns a set of the {@code proto_source_root} collected from the current library and the
   * exported dependencies.
   *
   * <p>Assumes {@code currentProtoSourceRoot} is the same as the package name.
   */
  private static NestedSet<String> computeExportedProtoSourceRoots(
      RuleContext ruleContext, String currentProtoSourceRoot) {
    return getProtoSourceRootsOfAttribute(ruleContext, currentProtoSourceRoot, "exports");
  }

  /**
   * Check that .proto files in sources are from the same package. This is done to avoid clashes
   * with the generated sources.
   */
  private static void checkSourceFilesAreInSamePackage(RuleContext ruleContext) {
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

  public static void checkRuleHasValidMigrationTag(RuleContext ruleContext)
      throws RuleErrorException {
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
      RuleContext ruleContext, boolean generatedProtosInVirtualImports) {
    checkSourceFilesAreInSamePackage(ruleContext);
    ImmutableList<Artifact> directProtoSources =
        ruleContext.getPrerequisiteArtifacts("srcs", Mode.TARGET).list();
    Library library =
        createLibraryWithVirtualSourceRootMaybe(
            ruleContext, directProtoSources, generatedProtosInVirtualImports);
    if (ruleContext.hasErrors()) {
      return null;
    }

    if (library == null) {
      library = createLibraryWithoutVirtualSourceRoot(ruleContext, directProtoSources);
    }

    NestedSet<Artifact> transitiveProtoSources =
        computeTransitiveProtoSources(ruleContext, library.getSources());
    NestedSet<Artifact> transitiveOriginalProtoSources =
        computeTransitiveOriginalProtoSources(ruleContext, directProtoSources);
    NestedSet<String> transitiveProtoSourceRoots =
        computeTransitiveProtoSourceRoots(ruleContext, library.getSourceRoot());

    NestedSet<Artifact> strictImportableProtosForDependents =
        computeStrictImportableProtosForDependents(ruleContext, library.getSources());
    NestedSet<Pair<Artifact, String>> strictImportableProtosImportPathsForDependents =
        computeStrictImportableProtosImportPathsForDependents(
            ruleContext, library.getImportPathSourcePair());
    NestedSet<Pair<Artifact, String>> strictImportableProtos =
        computeStrictImportableProtos(ruleContext, library.getImportPathSourcePair());
    NestedSet<String> strictImportableProtoSourceRoots =
        computeStrictImportableProtoSourceRoots(ruleContext, library.getSourceRoot());

    NestedSet<Pair<Artifact, String>> exportedProtos =
        ProtoCommon.computeExportedProtos(ruleContext);
    NestedSet<String> exportedProtoSourceRoots =
        computeExportedProtoSourceRoots(ruleContext, library.getSourceRoot());

    Artifact directDescriptorSet =
        ruleContext.getGenfilesArtifact(
            ruleContext.getLabel().getName() + "-descriptor-set.proto.bin");
    NestedSet<Artifact> dependenciesDescriptorSets =
        ProtoCommon.computeDependenciesDescriptorSets(ruleContext);
    NestedSet<Artifact> transitiveDescriptorSets =
        NestedSetBuilder.fromNestedSet(dependenciesDescriptorSets).add(directDescriptorSet).build();

    ProtoInfo protoInfo =
        new ProtoInfo(
            library.getSources(),
            directProtoSources,
            library.getSourceRoot(),
            transitiveProtoSources,
            transitiveOriginalProtoSources,
            transitiveProtoSourceRoots,
            strictImportableProtosForDependents,
            strictImportableProtos,
            strictImportableProtosImportPathsForDependents,
            strictImportableProtoSourceRoots,
            exportedProtos,
            exportedProtoSourceRoots,
            directDescriptorSet,
            transitiveDescriptorSets,
            Location.BUILTIN);

    return protoInfo;
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

  /**
   * Returns the .proto files that are the direct srcs of the direct-dependencies of this rule. If
   * the current rule is an alias proto_library (=no srcs), we use the direct srcs of the
   * direct-dependencies of our direct-dependencies.
   */
  @Nullable
  private static NestedSet<Pair<Artifact, String>> computeStrictImportableProtos(
      RuleContext ruleContext, ImmutableList<Pair<Artifact, String>> importPathSourcePair) {
    NestedSetBuilder<Pair<Artifact, String>> result = NestedSetBuilder.stableOrder();
    if (importPathSourcePair.isEmpty()) {
      for (ProtoInfo provider : ruleContext.getPrerequisites("deps", TARGET, ProtoInfo.PROVIDER)) {
        result.addTransitive(provider.getStrictImportableProtoSourcesImportPaths());
      }
    } else {
      for (ProtoInfo provider : ruleContext.getPrerequisites("deps", TARGET, ProtoInfo.PROVIDER)) {
        result.addTransitive(provider.getStrictImportableProtoSourcesImportPathsForDependents());
      }
      result.addAll(importPathSourcePair);
    }
    return result.build();
  }

  /**
   * Returns the .proto files that are the direct srcs of the exported dependencies of this rule.
   */
  private static NestedSet<Pair<Artifact, String>> computeExportedProtos(RuleContext ruleContext) {
    NestedSetBuilder<Pair<Artifact, String>> result = NestedSetBuilder.stableOrder();
    for (ProtoInfo provider : ruleContext.getPrerequisites("exports", TARGET, ProtoInfo.PROVIDER)) {
      result.addTransitive(provider.getStrictImportableProtoSourcesImportPaths());
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
