// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.trimming;

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses.BaseRule;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.ResolvedToolchainContext;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.ToolchainType.ToolchainTypeRule;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.repository.BindRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.util.List;
import java.util.function.Function;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;

/** Set of trimmable fragments for testing automatic trimming. */
public final class TrimmableTestConfigurationFragments {

  private TrimmableTestConfigurationFragments() {
    // Utility class, non-instantiable
  }

  public static void installStarlarkRules(Scratch scratch, String path)
      throws LabelSyntaxException, IOException {
    installStarlarkRules(
        scratch, path, Label.parseAbsolute("//:undefined_toolchain_type", ImmutableMap.of()));
  }

  public static void installStarlarkRules(Scratch scratch, String path, Label toolchainTypeLabel)
      throws LabelSyntaxException, IOException {
    scratch.file(
        path,
        "toolchainTypeLabel = " + Starlark.repr(toolchainTypeLabel),
        "def _impl(ctx):",
        "  ctx.actions.write(ctx.outputs.main, '')",
        "  files = depset(",
        "      direct = [ctx.outputs.main],",
        "      transitive = [dep.files for dep in ctx.attr.deps])",
        "  return [DefaultInfo(files=files)]",
        "alpha_starlark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['alpha'],",
        "    outputs = {'main': '%{name}.sa'},",
        ")",
        "bravo_starlark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['bravo'],",
        "    outputs = {'main': '%{name}.sb'},",
        ")",
        "charlie_starlark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['charlie'],",
        "    outputs = {'main': '%{name}.sc'},",
        ")",
        "delta_starlark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['delta'],",
        "    outputs = {'main': '%{name}.sd'},",
        ")",
        "echo_starlark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['echo'],",
        "    outputs = {'main': '%{name}.se'},",
        ")",
        "platformer_starlark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    outputs = {'main': '%{name}.sp'},",
        ")",
        "def _uses_toolchains_impl(ctx):",
        "  ctx.actions.write(ctx.outputs.main, '')",
        "  transitive_depsets = [dep.files for dep in ctx.attr.deps]",
        "  toolchain_deps = ctx.toolchains[toolchainTypeLabel].files",
        "  files = depset(",
        "      direct = [ctx.outputs.main],",
        "      transitive = transitive_depsets + [toolchain_deps])",
        "  return [DefaultInfo(files=files)]",
        "uses_toolchains_starlark = rule(",
        "    implementation = _uses_toolchains_impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    outputs = {'main': '%{name}.su'},",
        "    toolchains = [str(toolchainTypeLabel)],",
        ")",
        "def _toolchain_impl(ctx):",
        "  ctx.actions.write(ctx.outputs.main, '')",
        "  files = depset(",
        "      direct = [ctx.outputs.main],",
        "      transitive = [dep.files for dep in ctx.attr.deps])",
        "  return [DefaultInfo(files=files), platform_common.ToolchainInfo(files=files)]",
        "toolchain_starlark = rule(",
        "    implementation = _toolchain_impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    outputs = {'main': '%{name}.st'},",
        ")",
        "def _group_impl(ctx):",
        "  files = depset(transitive = [dep.files for dep in ctx.attr.deps])",
        "  return [DefaultInfo(files=files)]",
        "group = rule(",
        "    implementation = _group_impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        ")");
  }

  public static void installFragmentsAndNativeRules(ConfiguredRuleClassProvider.Builder builder) {
    installFragmentsAndNativeRules(builder, null);
  }

  public static void installFragmentsAndNativeRules(
      ConfiguredRuleClassProvider.Builder builder, @Nullable Label toolchainTypeLabel) {
    // boilerplate:
    builder
        // must be set, but it doesn't matter here what it's set to
        .setToolsRepository("@")
        // must be set, but it doesn't matter here what it's set to
        .setRunfilesPrefix("runfiles")
        // must be set, but it doesn't matter here what it's set to
        .setPrerequisiteValidator((contextBuilder, prerequisite, attribute) -> {})
        // must be set, but it doesn't matter here what it's set to
        .setPrelude("//:prelude.bzl")
        // must be part of BuildOptions for various reasons e.g. dynamic configs
        .addConfigurationOptions(CoreOptions.class)
        .addConfigurationFragment(new TestConfiguration.Loader())
        // needed for the default workspace
        .addRuleDefinition(new WorkspaceBaseRule())
        .addRuleDefinition(new BindRule())
        // needed for our native rules
        .addRuleDefinition(new BaseRule())
        // needed to define toolchains
        .addRuleDefinition(new ToolchainTypeRule())
        // needs to be set to something
        .addUniversalConfigurationFragment(TestConfiguration.class);

    CoreRules.INSTANCE.init(builder);

    MockRule transitionRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "with_configuration",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .requiresConfigurationFragments(
                              AConfig.class,
                              BConfig.class,
                              CConfig.class,
                              DConfig.class,
                              EConfig.class,
                              PlatformConfiguration.class)
                          .cfg(new TestFragmentTransitionFactory())
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .add(
                              attr("alpha", Type.STRING)
                                  .value((String) null)
                                  .nonconfigurable("used in transition"))
                          .add(
                              attr("bravo", Type.STRING)
                                  .value((String) null)
                                  .nonconfigurable("used in transition"))
                          .add(
                              attr("charlie", Type.STRING)
                                  .value((String) null)
                                  .nonconfigurable("used in transition"))
                          .add(
                              attr("delta", Type.STRING)
                                  .value((String) null)
                                  .nonconfigurable("used in transition"))
                          .add(
                              attr("echo", Type.STRING)
                                  .value((String) null)
                                  .nonconfigurable("used in transition"))
                          .add(
                              attr("platforms", BuildType.NODEP_LABEL_LIST)
                                  .value((List<Label>) null)
                                  .nonconfigurable("used in transition"))
                          .add(
                              attr("extra_execution_platforms", Type.STRING_LIST)
                                  .value((List<String>) null)
                                  .nonconfigurable("used in transition"))
                          .add(
                              attr("extra_toolchains", Type.STRING_LIST)
                                  .value((List<String>) null)
                                  .nonconfigurable("used in transition"));
                    });

    MockRule alphaRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "alpha_native",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .requiresConfigurationFragments(AConfig.class)
                          .setImplicitOutputsFunction(
                              ImplicitOutputsFunction.fromTemplates("%{name}.a"));
                    });

    MockRule bravoRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "bravo_native",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .requiresConfigurationFragments(BConfig.class)
                          .setImplicitOutputsFunction(
                              ImplicitOutputsFunction.fromTemplates("%{name}.b"));
                    });

    MockRule charlieRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "charlie_native",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .requiresConfigurationFragments(CConfig.class)
                          .setImplicitOutputsFunction(
                              ImplicitOutputsFunction.fromTemplates("%{name}.c"));
                    });

    MockRule deltaRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "delta_native",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .requiresConfigurationFragments(DConfig.class)
                          .setImplicitOutputsFunction(
                              ImplicitOutputsFunction.fromTemplates("%{name}.d"));
                    });

    MockRule echoRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "echo_native",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .requiresConfigurationFragments(EConfig.class)
                          .setImplicitOutputsFunction(
                              ImplicitOutputsFunction.fromTemplates("%{name}.e"));
                    });

    MockRule platformlessRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "platformless_native",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .useToolchainResolution(false)
                          .setImplicitOutputsFunction(
                              ImplicitOutputsFunction.fromTemplates("%{name}.np"));
                    });

    MockRule platformerRule =
        () ->
            MockRule.ancestor(BaseRule.class)
                .factory(DepsCollectingFactory.class)
                .define(
                    "platformer_native",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .add(
                              attr("deps", BuildType.LABEL_LIST)
                                  .allowedFileTypes(FileTypeSet.ANY_FILE))
                          .useToolchainResolution(true)
                          .setImplicitOutputsFunction(
                              ImplicitOutputsFunction.fromTemplates("%{name}.p"));
                    });

    builder
        .addConfigurationFragment(AConfig.FACTORY)
        .addConfigurationFragment(BConfig.FACTORY)
        .addConfigurationFragment(CConfig.FACTORY)
        .addConfigurationFragment(DConfig.FACTORY)
        .addConfigurationFragment(EConfig.FACTORY)
        .addRuleDefinition(transitionRule)
        .addRuleDefinition(alphaRule)
        .addRuleDefinition(bravoRule)
        .addRuleDefinition(charlieRule)
        .addRuleDefinition(deltaRule)
        .addRuleDefinition(echoRule)
        .addRuleDefinition(platformlessRule)
        .addRuleDefinition(platformerRule);

    if (toolchainTypeLabel != null) {
      MockRule usesToolchainsRule =
          () ->
              MockRule.ancestor(BaseRule.class)
                  .factory(DepsCollectingFactory.class)
                  .define(
                      "uses_toolchains_native",
                      (ruleBuilder, env) -> {
                        ruleBuilder
                            .add(
                                attr("deps", BuildType.LABEL_LIST)
                                    .allowedFileTypes(FileTypeSet.ANY_FILE))
                            .useToolchainResolution(true)
                            .addRequiredToolchains(toolchainTypeLabel)
                            .setImplicitOutputsFunction(
                                ImplicitOutputsFunction.fromTemplates("%{name}.u"));
                      });
      builder.addRuleDefinition(usesToolchainsRule);
    }
  }

  /** General purpose fragment loader for the test fragments in this file. */
  private static final class FragmentLoader<
          OptionsT extends FragmentOptions, FragmentT extends Fragment>
      implements ConfigurationFragmentFactory {
    private final Class<FragmentT> fragmentType;
    private final Class<OptionsT> optionsType;
    private final Function<OptionsT, FragmentT> fragmentMaker;

    FragmentLoader(
        Class<FragmentT> fragmentType,
        Class<OptionsT> optionsType,
        Function<OptionsT, FragmentT> fragmentMaker) {
      this.fragmentType = fragmentType;
      this.optionsType = optionsType;
      this.fragmentMaker = fragmentMaker;
    }

    @Override
    public Class<? extends Fragment> creates() {
      return fragmentType;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(optionsType);
    }

    @Override
    public Fragment create(BuildOptions buildOptions) {
      return fragmentMaker.apply(buildOptions.get(optionsType));
    }
  }

  /** Set of test options. */
  public static final class AOptions extends FragmentOptions {
    @Option(
        name = "alpha",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "0")
    public String alpha;
  }

  /** Test configuration fragment. */
  @StarlarkBuiltin(name = "alpha", doc = "Test config fragment.")
  public static final class AConfig extends Fragment implements StarlarkValue {
    public static final ConfigurationFragmentFactory FACTORY =
        new FragmentLoader<>(
            AConfig.class, AOptions.class, (options) -> new AConfig(options.alpha));

    private final String value;

    public AConfig(String value) {
      this.value = value;
    }

    @Override
    public String getOutputDirectoryName() {
      return "A" + value;
    }
  }

  /** Set of test options. */
  public static final class BOptions extends FragmentOptions {
    @Option(
        name = "bravo",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "0")
    public String bravo;
  }

  /** Test configuration fragment. */
  @StarlarkBuiltin(name = "bravo", doc = "Test config fragment.")
  public static final class BConfig extends Fragment implements StarlarkValue {
    public static final ConfigurationFragmentFactory FACTORY =
        new FragmentLoader<>(
            BConfig.class, BOptions.class, (options) -> new BConfig(options.bravo));

    private final String value;

    public BConfig(String value) {
      this.value = value;
    }

    @Override
    public String getOutputDirectoryName() {
      return "B" + value;
    }
  }

  /** Set of test options. */
  public static final class COptions extends FragmentOptions {
    @Option(
        name = "charlie",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "0")
    public String charlie;
  }

  /** Test configuration fragment. */
  @StarlarkBuiltin(name = "charlie", doc = "Test config fragment.")
  public static final class CConfig extends Fragment implements StarlarkValue {
    public static final ConfigurationFragmentFactory FACTORY =
        new FragmentLoader<>(
            CConfig.class, COptions.class, (options) -> new CConfig(options.charlie));

    private final String value;

    public CConfig(String value) {
      this.value = value;
    }

    @Override
    public String getOutputDirectoryName() {
      return "C" + value;
    }
  }

  /** Set of test options. */
  public static final class DOptions extends FragmentOptions {
    @Option(
        name = "delta",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "0")
    public String delta;
  }

  /** Test configuration fragment. */
  @StarlarkBuiltin(name = "delta", doc = "Test config fragment.")
  public static final class DConfig extends Fragment implements StarlarkValue {
    public static final ConfigurationFragmentFactory FACTORY =
        new FragmentLoader<>(
            DConfig.class, DOptions.class, (options) -> new DConfig(options.delta));

    private final String value;

    public DConfig(String value) {
      this.value = value;
    }

    @Override
    public String getOutputDirectoryName() {
      return "D" + value;
    }
  }

  /** Set of test options. */
  public static final class EOptions extends FragmentOptions {
    public static final OptionDefinition ECHO =
        OptionsParser.getOptionDefinitionByName(EOptions.class, "echo");

    @Option(
        name = "echo",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "0")
    public String echo;
  }

  /** Test configuration fragment. */
  @StarlarkBuiltin(name = "echo", doc = "Test config fragment.")
  public static final class EConfig extends Fragment implements StarlarkValue {
    public static final ConfigurationFragmentFactory FACTORY =
        new FragmentLoader<>(EConfig.class, EOptions.class, (options) -> new EConfig(options.echo));

    private final String value;

    public EConfig(String value) {
      this.value = value;
    }

    @Override
    public String getOutputDirectoryName() {
      return "E" + value;
    }
  }

  private static final class TestFragmentTransitionFactory implements TransitionFactory<Rule> {
    private static final class SetValuesTransition implements PatchTransition {
      private final String alpha;
      private final String bravo;
      private final String charlie;
      private final String delta;
      private final String echo;
      private final List<Label> platforms;
      private final List<String> extraExecutionPlatforms;
      private final List<String> extraToolchains;

      public SetValuesTransition(
          String alpha,
          String bravo,
          String charlie,
          String delta,
          String echo,
          List<Label> platforms,
          List<String> extraExecutionPlatforms,
          List<String> extraToolchains) {
        this.alpha = alpha;
        this.bravo = bravo;
        this.charlie = charlie;
        this.delta = delta;
        this.echo = echo;
        this.platforms = platforms;
        this.extraExecutionPlatforms = extraExecutionPlatforms;
        this.extraToolchains = extraToolchains;
      }

      @Override
      public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
        return ImmutableSet.of(
            AOptions.class,
            BOptions.class,
            COptions.class,
            DOptions.class,
            EOptions.class,
            PlatformOptions.class);
      }

      @Override
      public BuildOptions patch(BuildOptionsView target, EventHandler eventHandler) {
        BuildOptionsView output = target.clone();
        if (alpha != null) {
          output.get(AOptions.class).alpha = alpha;
        }
        if (bravo != null) {
          output.get(BOptions.class).bravo = bravo;
        }
        if (charlie != null) {
          output.get(COptions.class).charlie = charlie;
        }
        if (delta != null) {
          output.get(DOptions.class).delta = delta;
        }
        if (echo != null) {
          output.get(EOptions.class).echo = echo;
        }
        if (platforms != null) {
          output.get(PlatformOptions.class).platforms = platforms;
        }
        if (extraExecutionPlatforms != null) {
          output.get(PlatformOptions.class).extraExecutionPlatforms = extraExecutionPlatforms;
        }
        if (extraToolchains != null) {
          output.get(PlatformOptions.class).extraToolchains = extraToolchains;
        }
        return output.underlying();
      }
    }

    @Override
    public PatchTransition create(Rule rule) {
      AttributeMap attributes = NonconfigurableAttributeMapper.of(rule);
      return new SetValuesTransition(
          attributes.get("alpha", Type.STRING),
          attributes.get("bravo", Type.STRING),
          attributes.get("charlie", Type.STRING),
          attributes.get("delta", Type.STRING),
          attributes.get("echo", Type.STRING),
          attributes.get("platforms", BuildType.NODEP_LABEL_LIST),
          attributes.get("extra_execution_platforms", Type.STRING_LIST),
          attributes.get("extra_toolchains", Type.STRING_LIST));
    }
  }

  /** RuleConfiguredTargetFactory which collects dependencies. */
  public static final class DepsCollectingFactory implements RuleConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();
      filesToBuild.addAll(ruleContext.getOutputArtifacts());
      for (FileProvider dep :
          ruleContext.getPrerequisites("deps", TransitionMode.TARGET, FileProvider.class)) {
        filesToBuild.addTransitive(dep.getFilesToBuild());
      }
      for (Artifact artifact : ruleContext.getOutputArtifacts()) {
        ruleContext.registerAction(
            FileWriteAction.createEmptyWithInputs(
                ruleContext.getActionOwner(),
                NestedSetBuilder.emptySet(Order.STABLE_ORDER),
                artifact));
      }
      if (ruleContext.getToolchainContext() != null) {
        ResolvedToolchainContext toolchainContext = ruleContext.getToolchainContext();
        for (ToolchainTypeInfo toolchainType : toolchainContext.requiredToolchainTypes()) {
          ToolchainInfo toolchainInfo = toolchainContext.forToolchainType(toolchainType);
          try {
            filesToBuild.addTransitive(
                ((Depset) toolchainInfo.getValue("files")).getSet(Artifact.class));
          } catch (EvalException | Depset.TypeException ex) {
            throw new AssertionError(ex);
          }
        }
      }
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(filesToBuild.build())
          .setRunfilesSupport(null, null)
          .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
          .build();
    }
  }
}
