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

import static com.google.common.base.Verify.verify;
import static com.google.common.base.Verify.verifyNotNull;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
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
import com.google.devtools.build.lib.cmdline.Label;
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
import java.util.function.Predicate;
import javax.annotation.Nullable;
import net.starlark.java.eval.Module;

/** Implementation of the {@code starlark_doc_extract} rule. */
public class StarlarkDocExtract implements RuleConfiguredTargetFactory {
  static final String SRC_ATTR = "src";
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

    Module module = loadModule(ruleContext);
    if (module == null) {
      // Skyframe restart
      verify(
          ruleContext.getAnalysisEnvironment().getSkyframeEnv().valuesMissing()
              && !ruleContext.hasErrors());
      return null;
    }
    ModuleInfo moduleInfo = getModuleInfo(ruleContext, module);

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

  @Nullable
  private static Module loadModule(RuleContext ruleContext)
      throws RuleErrorException, InterruptedException {
    try (SilentCloseable c = Profiler.instance().profile("BzlDocDump.loadModule")) {
      final Label label = ruleContext.attributes().get(SRC_ATTR, LABEL);
      // Note getSkyframeEnv() cannot be null while creating a configured target.
      SkyFunction.Environment env = ruleContext.getAnalysisEnvironment().getSkyframeEnv();

      BzlLoadValue bzlLoadValue;
      try {
        // TODO(b/276733504): support loading modules in non-BUILD context (bzlmod, prelude,
        // builtins, and possibly workspace).
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

  private static ModuleInfo getModuleInfo(RuleContext ruleContext, Module module)
      throws RuleErrorException, InterruptedException {
    RepositoryMappingValue repositoryMappingValue;
    try {
      // We get the starlark_doc_extract target's repository's repo mapping to ensure label
      // stringification does not change regardless of whether we are the main repo or a dependency.
      // However, this does mean that label stringifactions we produce could be invalid in the main
      // repo.
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

    ModuleInfo moduleInfo;
    try {
      moduleInfo =
          new ModuleInfoExtractor(
                  getWantedSymbolPredicate(ruleContext),
                  repositoryMappingValue.getRepositoryMapping())
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
