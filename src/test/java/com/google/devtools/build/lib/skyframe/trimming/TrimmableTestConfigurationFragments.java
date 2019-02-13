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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFragmentFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.rules.repository.BindRule;
import com.google.devtools.build.lib.rules.repository.WorkspaceBaseRule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import java.io.IOException;
import java.util.function.Function;

/** Set of trimmable fragments for testing automatic trimming. */
public final class TrimmableTestConfigurationFragments {

  private TrimmableTestConfigurationFragments() {
    // Utility class, non-instantiable
  }

  public static void installSkylarkRules(Scratch scratch, String path) throws IOException {
    scratch.file(
        path,
        "def _impl(ctx):",
        "  ctx.actions.write(ctx.outputs.main, '')",
        "  files = depset(",
        "      direct = [ctx.outputs.main],",
        "      transitive = [dep.files for dep in ctx.attr.deps])",
        "  return [DefaultInfo(files=files)]",
        "alpha_skylark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['alpha'],",
        "    outputs = {'main': '%{name}.sa'},",
        ")",
        "bravo_skylark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['bravo'],",
        "    outputs = {'main': '%{name}.sb'},",
        ")",
        "charlie_skylark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['charlie'],",
        "    outputs = {'main': '%{name}.sc'},",
        ")",
        "delta_skylark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['delta'],",
        "    outputs = {'main': '%{name}.sd'},",
        ")",
        "echo_skylark = rule(",
        "    implementation = _impl,",
        "    attrs = {'deps': attr.label_list(allow_files=True)},",
        "    fragments = ['echo'],",
        "    outputs = {'main': '%{name}.se'},",
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
        .addConfigurationOptions(BuildConfiguration.Options.class)
        .addConfigurationFragment(new TestConfiguration.Loader())
        // needed for the other rules to build on and the default workspace
        .addRuleDefinition(new BaseRuleClasses.RootRule())
        .addRuleDefinition(new BaseRuleClasses.BaseRule())
        .addRuleDefinition(new WorkspaceBaseRule())
        .addRuleDefinition(new BindRule())
        // needs to be set to something
        .addUniversalConfigurationFragment(TestConfiguration.class);

    MockRule transitionRule =
        () ->
            MockRule.factory(DepsCollectingFactory.class)
                .define(
                    "with_configuration",
                    (ruleBuilder, env) -> {
                      ruleBuilder
                          .requiresConfigurationFragments(
                              AConfig.class,
                              BConfig.class,
                              CConfig.class,
                              DConfig.class,
                              EConfig.class)
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
                                  .nonconfigurable("used in transition"));
                    });

    MockRule alphaRule =
        () ->
            MockRule.factory(DepsCollectingFactory.class)
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
            MockRule.factory(DepsCollectingFactory.class)
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
            MockRule.factory(DepsCollectingFactory.class)
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
            MockRule.factory(DepsCollectingFactory.class)
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
            MockRule.factory(DepsCollectingFactory.class)
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
        .addRuleDefinition(echoRule);
  }

  /** General purpose fragment loader for the test fragments in this file. */
  private static final class FragmentLoader<
          OptionsT extends FragmentOptions, FragmentT extends BuildConfiguration.Fragment>
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
    public Class<? extends BuildConfiguration.Fragment> creates() {
      return fragmentType;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiredOptions() {
      return ImmutableSet.<Class<? extends FragmentOptions>>of(optionsType);
    }

    @Override
    public BuildConfiguration.Fragment create(BuildOptions buildOptions) {
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
  @SkylarkModule(name = "alpha", doc = "Test config fragment.")
  public static final class AConfig extends BuildConfiguration.Fragment {
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
  @SkylarkModule(name = "bravo", doc = "Test config fragment.")
  public static final class BConfig extends BuildConfiguration.Fragment {
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
  @SkylarkModule(name = "charlie", doc = "Test config fragment.")
  public static final class CConfig extends BuildConfiguration.Fragment {
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
  @SkylarkModule(name = "delta", doc = "Test config fragment.")
  public static final class DConfig extends BuildConfiguration.Fragment {
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
    @Option(
        name = "echo",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "0")
    public String echo;
  }

  /** Test configuration fragment. */
  @SkylarkModule(name = "echo", doc = "Test config fragment.")
  public static final class EConfig extends BuildConfiguration.Fragment {
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

  private static final class TestFragmentTransitionFactory implements RuleTransitionFactory {
    private static final class SetValuesTransition implements PatchTransition {
      private final String alpha;
      private final String bravo;
      private final String charlie;
      private final String delta;
      private final String echo;

      public SetValuesTransition(
          String alpha, String bravo, String charlie, String delta, String echo) {
        this.alpha = alpha;
        this.bravo = bravo;
        this.charlie = charlie;
        this.delta = delta;
        this.echo = echo;
      }

      @Override
      public BuildOptions patch(BuildOptions target) {
        BuildOptions output = target.clone();
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
        return output;
      }
    }

    @Override
    public PatchTransition buildTransitionFor(Rule rule) {
      NonconfigurableAttributeMapper attributes = NonconfigurableAttributeMapper.of(rule);
      return new SetValuesTransition(
          attributes.get("alpha", Type.STRING),
          attributes.get("bravo", Type.STRING),
          attributes.get("charlie", Type.STRING),
          attributes.get("delta", Type.STRING),
          attributes.get("echo", Type.STRING));
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
          ruleContext.getPrerequisites("deps", Mode.TARGET, FileProvider.class)) {
        filesToBuild.addTransitive(dep.getFilesToBuild());
      }
      for (Artifact artifact : ruleContext.getOutputArtifacts()) {
        ruleContext.registerAction(
            FileWriteAction.createEmptyWithInputs(
                ruleContext.getActionOwner(), ImmutableList.of(), artifact));
      }
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(filesToBuild.build())
          .setRunfilesSupport(null, null)
          .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
          .build();
    }
  }
}
