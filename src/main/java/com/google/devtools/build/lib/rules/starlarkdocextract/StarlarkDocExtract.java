// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.starlarkdocextract;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Verify.verify;
import static com.google.common.base.Verify.verifyNotNull;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static java.util.stream.Collectors.partitioningBy;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.BinaryFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.TextFormat;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import net.starlark.java.eval.Module;

/** Implementation of the {@code starlark_doc_extract} rule. */
public class StarlarkDocExtract implements RuleConfiguredTargetFactory {
  static final String SRC_ATTR = "src";
  static final String DEPS_ATTR = "deps";
  static final String SYMBOL_NAMES_ATTR = "symbol_names";
  static final SafeImplicitOutputsFunction BINARYPROTO_OUT = fromTemplates("%{name}.binaryproto");
  static final SafeImplicitOutputsFunction TEXTPROTO_OUT = fromTemplates("%{name}.textproto");

  /** Configuration fragment for the {@code starlark_doc_extract} rule. */
  // TODO(b/276733504): remove once non-experimental.
  @RequiresOptions(options = {Configuration.Options.class})
  public static final class Configuration extends Fragment {

    /** Options for the {@code starlark_doc_extract} rule. */
    public static final class Options extends FragmentOptions {
      @Option(
          name = "experimental_enable_starlark_doc_extract",
          defaultValue = "false",
          documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
          effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
          metadataTags = {OptionMetadataTag.EXPERIMENTAL},
          help = "If set to true, enables the experimental starlark_doc_extract rule.")
      public boolean experimentalEnableBzlDocDump;
    }

    private final boolean enabled;

    public Configuration(BuildOptions buildOptions) {
      enabled = buildOptions.get(Options.class).experimentalEnableBzlDocDump;
    }

    public boolean enabled() {
      return enabled;
    }
  }

  @Override
  @Nullable
  public ConfiguredTarget create(RuleContext ruleContext)
      throws ActionConflictException, InterruptedException, RuleErrorException {
    if (!ruleContext.getFragment(Configuration.class).enabled()) {
      RuleErrorException exception =
          new RuleErrorException(
              "The experimental starlark_doc_extract rule is disabled; use"
                  + " --experimental_enable_starlark_doc_extract flag to enable.");
      ruleContext.ruleError(exception.getMessage());
      throw exception;
    }

    RepositoryMapping repositoryMapping = getTargetRepositoryMapping(ruleContext);
    Module module = loadModule(ruleContext, repositoryMapping);
    if (module == null) {
      // Skyframe restart
      verify(
          ruleContext.getAnalysisEnvironment().getSkyframeEnv().valuesMissing()
              && !ruleContext.hasErrors());
      return null;
    }
    verifyModuleDeps(ruleContext, module, repositoryMapping);
    ModuleInfo moduleInfo = getModuleInfo(ruleContext, module, repositoryMapping);

    NestedSet<Artifact> filesToBuild =
        new NestedSetBuilder<Artifact>(Order.STABLE_ORDER)
            .add(createBinaryProtoOutput(ruleContext, moduleInfo))
            .build();
    // Textproto output isn't in filesToBuild: we want to create it only if explicitly requested.
    createTextProtoOutput(ruleContext, moduleInfo);

    return new RuleConfiguredTargetBuilder(ruleContext)
        .setFilesToBuild(filesToBuild)
        .addProvider(
            RunfilesProvider.class,
            RunfilesProvider.simple(
                new Runfiles.Builder(ruleContext.getWorkspaceName())
                    .addRunfiles(ruleContext, RunfilesProvider.DEFAULT_RUNFILES)
                    .addTransitiveArtifacts(filesToBuild)
                    .build()))
        .build();
  }

  /**
   * Loads the Starlark module from the source file given by the rule's {@code src} attribute.
   *
   * @throws RuleErrorException and reports an error in the rule if the {@code src} attribute refers
   *     to multiple or zero files, a generated file, or a source file which cannot be loaded or
   *     parsed
   * @return the module object, or null on Skyframe restart
   */
  @Nullable
  private static Module loadModule(RuleContext ruleContext, RepositoryMapping repositoryMapping)
      throws RuleErrorException, InterruptedException {
    try (SilentCloseable c = Profiler.instance().profile("BzlDocDump.loadModule")) {
      // Note attr schema validates that src is a .bzl or .scl file.
      Label label = getSourceFileLabel(ruleContext, SRC_ATTR, repositoryMapping);

      // Note getSkyframeEnv() cannot be null while creating a configured target.
      SkyFunction.Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();

      BzlLoadValue bzlLoadValue;
      try {
        // TODO(b/276733504): support loading modules in @_builtins
        bzlLoadValue =
            (BzlLoadValue)
                env.getValueOrThrow(BzlLoadValue.keyForBuild(label), BzlLoadFailedException.class);
      } catch (BzlLoadFailedException e) {
        ruleContext.attributeError(SRC_ATTR, e.getMessage());
        throw new RuleErrorException(e);
      }
      if (bzlLoadValue == null) {
        // Skyframe restart
        return null;
      }
      return bzlLoadValue.getModule();
    }
  }

  /**
   * Retrieves the label of the singular source artifact from a given attribute. Note that we can't
   * simply use {@code ruleContext.attributes().get(attrName, LABEL)} because that does not resolve
   * aliases and filegroups.
   *
   * @throws RuleErrorException if the source is not a singular source artifact, meaning its label
   *     cannot be used as a label for a Starlark load()
   */
  private static Label getSourceFileLabel(
      RuleContext ruleContext, String attrName, RepositoryMapping repositoryMapping)
      throws RuleErrorException {
    Artifact artifact = ruleContext.getPrerequisiteArtifact(attrName);
    ruleContext.assertNoErrors();
    // If ruleContext.getPrerequisiteArtifact() set no errors, we know artifact != null
    if (!artifact.isSourceArtifact()) {
      RuleErrorException error =
          new RuleErrorException(
              String.format(
                  "%s is not a source file and cannot be loaded in Starlark",
                  formatDerivedArtifact(artifact, repositoryMapping)));
      ruleContext.attributeError(attrName, error.getMessage());
      throw error;
    }
    return verifyNotNull(artifact.getOwner());
  }

  private static String formatDerivedArtifact(
      Artifact artifact, RepositoryMapping repositoryMapping) {
    checkArgument(!artifact.isSourceArtifact());
    return String.format(
        "%s (generated by rule %s)",
        artifact.getRepositoryRelativePath(),
        artifact.getOwner().getDisplayForm(repositoryMapping));
  }

  /**
   * Verifies that the module's transitive loads are a subset of the source artifacts in
   * files-to-build of the rule's deps.
   *
   * @throws RuleErrorException if that is not the case.
   */
  private static void verifyModuleDeps(
      RuleContext ruleContext, Module module, RepositoryMapping repositoryMapping)
      throws RuleErrorException {
    // Note attr schema validates that deps are .bzl or .scl files.
    Map<Boolean, ImmutableSet<Artifact>> flattenedDepsPartitionedByIsSource =
        ruleContext.getPrerequisites(DEPS_ATTR).stream()
            // TODO(https://github.com/bazelbuild/bazel/issues/18599): ideally we should use
            // StarlarkLibraryInfo here instead of FileProvider#getFilesToBuild; that requires a
            // native StarlarkLibraryInfo in Bazel.
            .flatMap(dep -> dep.getProvider(FileProvider.class).getFilesToBuild().toList().stream())
            .collect(partitioningBy(Artifact::isSourceArtifact, toImmutableSet()));
    // bzl_library targets may contain both source artifacts and derived artifacts (e.g. generated
    // .bzl files for tests); only the source artifacts can be load()-ed by Bazel.
    ImmutableSet<Artifact> flattenedDepsSourceArtifacts =
        flattenedDepsPartitionedByIsSource.getOrDefault(true, ImmutableSet.of());
    ImmutableSet<Artifact> flattenedDepsDerivedArtifacts =
        flattenedDepsPartitionedByIsSource.getOrDefault(false, ImmutableSet.of());

    ImmutableList<String> topmostUnknownLoads =
        getTopmostUnknownLoads(
            module,
            flattenedDepsSourceArtifacts.stream()
                .map(artifact -> verifyNotNull(artifact.getOwner()))
                .collect(toImmutableSet()),
            repositoryMapping);

    if (!topmostUnknownLoads.isEmpty()) {
      StringBuilder errorMessageBuilder =
          new StringBuilder("missing bzl_library targets for Starlark module(s) ")
              .append(Joiner.on(", ").join(topmostUnknownLoads));
      if (!flattenedDepsDerivedArtifacts.isEmpty()) {
        // TODO(arostovtsev): we ought to print only the derived artifacts having the same
        // root-relative path as topmostUnknownLoads.
        errorMessageBuilder
            .append("\nNote the following are generated file(s) and cannot be loaded in Starlark: ")
            .append(
                Joiner.on(", ")
                    .join(
                        flattenedDepsDerivedArtifacts.stream()
                            .map(artifact -> formatDerivedArtifact(artifact, repositoryMapping))
                            .iterator()));
      }
      RuleErrorException error = new RuleErrorException(errorMessageBuilder.toString());
      ruleContext.attributeError(DEPS_ATTR, error.getMessage());
      throw error;
    }
  }

  /**
   * Finds the topmost modules that are transitively loaded by the given module but not mentioned in
   * the given set of known modules, and returns these modules' display forms.
   *
   * <p>Unknown modules that are only referenced by other unknown modules are not included.
   */
  private static ImmutableList<String> getTopmostUnknownLoads(
      Module module, ImmutableSet<Label> knownModules, RepositoryMapping repositoryMapping) {
    ImmutableList.Builder<String> unknown = ImmutableList.builder();
    Set<Label> visited = new LinkedHashSet<>();
    BazelModuleContext.visitLoadGraphRecursively(
        BazelModuleContext.of(module).loads(),
        label -> {
          if (!visited.add(label)) {
            return false;
          }
          if (!knownModules.contains(label)) {
            unknown.add(label.getDisplayForm(repositoryMapping));
            return false;
          }
          return true;
        });
    return unknown.build();
  }

  /**
   * Returns the starlark_doc_extract target's repository's repo mapping.
   *
   * <p>We return the target's repository's repo mapping, as opposed to the main repo mapping, to
   * ensure label stringification does not change regardless of whether we are the main repo or a
   * dependency. However, this does mean that label stringifactions we produce could be invalid in
   * the main repo.
   */
  private static RepositoryMapping getTargetRepositoryMapping(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    RepositoryMappingValue repositoryMappingValue;
    try {
      repositoryMappingValue =
          (RepositoryMappingValue)
              ruleContext
                  .getAnalysisEnvironment()
                  .getSkyframeEnv()
                  .getValueOrThrow(
                      RepositoryMappingValue.key(ruleContext.getRepository()),
                      RepositoryMappingResolutionException.class);
    } catch (RepositoryMappingResolutionException e) {
      ruleContext.ruleError(e.getMessage());
      throw new RuleErrorException(e);
    }
    verifyNotNull(repositoryMappingValue);
    return repositoryMappingValue.getRepositoryMapping();
  }

  private static ModuleInfo getModuleInfo(
      RuleContext ruleContext, Module module, RepositoryMapping repositoryMapping)
      throws RuleErrorException {
    ModuleInfo moduleInfo;
    try {
      moduleInfo =
          new ModuleInfoExtractor(getWantedSymbolPredicate(ruleContext), repositoryMapping)
              .extractFrom(module);
    } catch (ModuleInfoExtractor.ExtractionException e) {
      ruleContext.ruleError(e.getMessage());
      throw new RuleErrorException(e);
    }
    return moduleInfo;
  }

  private static Predicate<String> getWantedSymbolPredicate(RuleContext ruleContext) {
    ImmutableList<String> symbolNames =
        ImmutableList.copyOf(ruleContext.attributes().get(SYMBOL_NAMES_ATTR, STRING_LIST));
    if (symbolNames.isEmpty()) {
      return name -> true;
    } else {
      return symbolNames::contains;
    }
  }

  private static Artifact createBinaryProtoOutput(RuleContext ruleContext, ModuleInfo moduleInfo)
      throws InterruptedException {
    Artifact binaryProtoOutput = ruleContext.getImplicitOutputArtifact(BINARYPROTO_OUT);
    ruleContext.registerAction(
        new BinaryFileWriteAction(
            ruleContext.getActionOwner(),
            binaryProtoOutput,
            ByteSource.wrap(moduleInfo.toByteArray()),
            /* makeExecutable= */ false));
    return binaryProtoOutput;
  }

  @Nullable
  @CanIgnoreReturnValue
  private static Artifact createTextProtoOutput(RuleContext ruleContext, ModuleInfo moduleInfo)
      throws InterruptedException, RuleErrorException {
    Artifact textProtoOutput = ruleContext.getImplicitOutputArtifact(TEXTPROTO_OUT);

    StringBuilder textprotoBuilder = new StringBuilder();
    try {
      TextFormat.printer().print(moduleInfo, textprotoBuilder);
    } catch (IOException e) {
      ruleContext.ruleError(e.getMessage());
      throw new RuleErrorException(e);
    }
    ruleContext.registerAction(
        FileWriteAction.create(
            ruleContext,
            textProtoOutput,
            textprotoBuilder.toString(),
            /* makeExecutable= */ false));
    return textProtoOutput;
  }
}
