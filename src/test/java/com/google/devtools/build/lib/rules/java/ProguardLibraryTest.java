// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredAspectFactory;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the actions built by ProguardLibrary. */
@RunWith(JUnit4.class)
public class ProguardLibraryTest extends BuildViewTestCase {

  /** Fake rule for testing {@link ProguardLibrary#collectProguardSpecs()} with all fields. */
  public static final class StandardProguardLibraryRule
      implements RuleDefinition, RuleConfiguredTargetFactory {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.NO_FILE))
          .add(attr("exports", LABEL_LIST).allowedFileTypes(FileTypeSet.NO_FILE))
          .add(attr("runtime_deps", LABEL_LIST).allowedFileTypes(FileTypeSet.NO_FILE))
          .add(
              attr("plugins", LABEL_LIST)
                  .cfg(ExecutionTransitionFactory.create())
                  .allowedFileTypes(FileTypeSet.NO_FILE))
          .add(
              attr("exported_plugins", LABEL_LIST)
                  .cfg(ExecutionTransitionFactory.create())
                  .allowedFileTypes(FileTypeSet.NO_FILE))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("standard_proguard_library")
          .ancestors(BaseRuleClasses.NativeBuildRule.class, ProguardLibraryRule.class)
          .factoryClass(StandardProguardLibraryRule.class)
          .build();
    }

    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      NestedSet<Artifact> proguardSpecs = new ProguardLibrary(ruleContext).collectProguardSpecs();
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(proguardSpecs)
          .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
          .addNativeDeclaredProvider(new ProguardSpecProvider(proguardSpecs))
          .build();
    }
  }

  /**
   * Fake rule for testing {@link ProguardLibrary#collectProguardSpecs()} with custom/absent fields.
   */
  public static final class CustomProguardLibraryRule
      implements RuleDefinition, RuleConfiguredTargetFactory {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(attr("target_libs", LABEL_LIST).allowedFileTypes(FileTypeSet.NO_FILE))
          .add(
              attr("host_libs", LABEL_LIST)
                  .cfg(ExecutionTransitionFactory.create())
                  .allowedFileTypes(FileTypeSet.NO_FILE))
          .add(attr("target_attrs", STRING_LIST))
          .add(attr("host_attrs", STRING_LIST))
          .add(
              attr("$implicit_target", LABEL)
                  .value(Label.parseCanonicalUnchecked("//test/implicit:implicit_target")))
          .add(
              attr("$implicit_host", LABEL)
                  .cfg(ExecutionTransitionFactory.create())
                  .value(Label.parseCanonicalUnchecked("//test/implicit:implicit_host")))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("custom_proguard_library")
          .ancestors(BaseRuleClasses.NativeBuildRule.class, ProguardLibraryRule.class)
          .factoryClass(CustomProguardLibraryRule.class)
          .build();
    }

    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws ActionConflictException, InterruptedException {
      NestedSet<Artifact> proguardSpecs =
          new ProguardLibrary(ruleContext)
              .collectProguardSpecs(
                  new ImmutableSet.Builder<String>()
                      .addAll(ruleContext.attributes().get("target_attrs", STRING_LIST))
                      .addAll(ruleContext.attributes().get("host_attrs", STRING_LIST))
                      .build());

      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(proguardSpecs)
          .addProvider(RunfilesProvider.class, RunfilesProvider.EMPTY)
          .addNativeDeclaredProvider(new ProguardSpecProvider(proguardSpecs))
          .build();
    }
  }

  /** Fake rule for testing {@link ProguardLibrary#collectProguardSpecs()} with aspect fields. */
  public static final class AspectProguardLibraryRule
      implements RuleDefinition, RuleConfiguredTargetFactory {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .add(
              attr("deps", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.NO_FILE)
                  .aspect(PROGUARD_LIBRARY_ASPECT))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("aspect_proguard_library")
          .ancestors(BaseRuleClasses.NativeBuildRule.class)
          .factoryClass(AspectProguardLibraryRule.class)
          .build();
    }

    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      NestedSet<Artifact> proguardSpecs = new ProguardLibrary(ruleContext).collectProguardSpecs();

      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(proguardSpecs)
          .addProvider(RunfilesProvider.EMPTY)
          .addNativeDeclaredProvider(new ProguardSpecProvider(proguardSpecs))
          .build();
    }
  }

  static final class ProguardLibraryAspect extends NativeAspectClass
      implements ConfiguredAspectFactory {
    @Override
    public AspectDefinition getDefinition(AspectParameters params) {
      return new AspectDefinition.Builder(this)
          .add(
              attr("$implicit_target", LABEL)
                  .value(Label.parseCanonicalUnchecked("//test/implicit:implicit_target")))
          .build();
    }

    @Override
    public ConfiguredAspect create(
        ConfiguredTargetAndData ctadBase,
        RuleContext ruleContext,
        AspectParameters params,
        RepositoryName toolsRepository)
        throws ActionConflictException, InterruptedException {
      NestedSet<Artifact> proguardSpecs =
          new ProguardLibrary(ruleContext)
              .collectProguardSpecs(ImmutableSet.of("$implicit_target"));

      return new ConfiguredAspect.Builder(ruleContext)
          .addNativeDeclaredProvider(new ProguardSpecProvider(proguardSpecs))
          .build();
    }
  }

  public static final ProguardLibraryAspect PROGUARD_LIBRARY_ASPECT = new ProguardLibraryAspect();

  @Before
  public final void createBuildFile() throws Exception {
    scratch.file(
        "test/implicit/BUILD",
        "standard_proguard_library(",
        "    name = 'implicit_target',",
        "    proguard_specs = ['implicit_target.cfg'])",
        "standard_proguard_library(",
        "    name = 'implicit_host',",
        "    proguard_specs = ['implicit_host.cfg'])");
  }

  /** Make the test rule class provider understand our rules in addition to the standard ones. */
  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(new StandardProguardLibraryRule())
            .addRuleDefinition(new CustomProguardLibraryRule())
            .addRuleDefinition(new AspectProguardLibraryRule())
            .addNativeAspectClass(PROGUARD_LIBRARY_ASPECT);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void testProguardSpecs_outputsAllowlistedVersions() throws Exception {
    scratch.file(
        "test/BUILD",
        "standard_proguard_library(",
        "    name = 'lib',",
        "    proguard_specs = ['optimizations.cfg', 'configs.cfg'])");

    ConfiguredTarget target = getConfiguredTarget("//test:lib");

    assertThat(getFilesToBuild(target).toList())
        .containsExactly(
            getBinArtifact("validated_proguard/lib/test/optimizations.cfg_valid", target),
            getBinArtifact("validated_proguard/lib/test/configs.cfg_valid", target));
  }

  @Test
  public void testProguardSpecs_originalConfigSentAsInputToAllowlister() throws Exception {
    scratch.file(
        "test/BUILD",
        "standard_proguard_library(",
        "    name = 'lib',",
        "    proguard_specs = ['optimizations.cfg', 'configs.cfg'])");

    ConfiguredTarget target = getConfiguredTarget("//test:lib");
    Artifact output = getBinArtifact("validated_proguard/lib/test/optimizations.cfg_valid", target);
    Artifact source = getFileConfiguredTarget("//test:optimizations.cfg").getArtifact();

    SpawnAction action = getGeneratingSpawnAction(output);

    assertThat(action).isNotNull();
    assertThat(action.getInputs().toList()).contains(source);

    List<String> inputs =
        action.getInputs().toList().stream()
            .map(Artifact::getExecPathString)
            // Strip the bazel-out/CONFIG/ portion of the paths.
            .map(path -> path.replaceFirst(TestConstants.PRODUCT_NAME + "-out/[^/]+/", ""))
            .collect(Collectors.toList());
    List<String> expectedFilesToRun =
        getFilesToRun(getConfiguredTarget(TestConstants.PROGUARD_ALLOWLISTER_TARGET))
            .toList()
            .stream()
            .map(Artifact::getExecPathString)
            // Strip the bazel-out/CONFIG/ portion of the paths.
            .map(path -> path.replaceFirst(TestConstants.PRODUCT_NAME + "-out/[^/]+/", ""))
            .collect(Collectors.toList());
    assertThat(inputs).containsAtLeastElementsIn(expectedFilesToRun);
  }

  @Test
  public void testProguardSpecs_allowlisterPathsPassedAsFlags() throws Exception {
    scratch.file(
        "test/BUILD",
        "standard_proguard_library(",
        "    name = 'lib',",
        "    proguard_specs = ['optimizations.cfg', 'configs.cfg'])");

    ConfiguredTarget target = getConfiguredTarget("//test:lib");
    Artifact output = getBinArtifact("validated_proguard/lib/test/optimizations.cfg_valid", target);
    Artifact source = getFileConfiguredTarget("//test:optimizations.cfg").getArtifact();

    SpawnAction action = getGeneratingSpawnAction(output);

    assertContainsSublist(
        action.getArguments(), ImmutableList.of("--path", source.getExecPathString()));
    assertContainsSublist(
        action.getArguments(), ImmutableList.of("--output", output.getExecPathString()));
  }

  @Test
  public void testProguardSpecs_pickedUpFromDependencyAttributes() throws Exception {
    scratch.file(
        "test/BUILD",
        "standard_proguard_library(",
        "    name = 'lib',",
        "    deps = [':dep'],",
        "    exports = [':export'],",
        "    runtime_deps = [':runtime'],",
        "    plugins = [':plugin'],",
        "    exported_plugins = [':exported_plugin'])",
        "standard_proguard_library(name = 'dep', proguard_specs = ['dep.cfg'])",
        "standard_proguard_library(name = 'export', proguard_specs = ['export.cfg'])",
        "standard_proguard_library(name = 'runtime', proguard_specs = ['runtime.cfg'])",
        "standard_proguard_library(name = 'plugin', proguard_specs = ['plugin.cfg'])",
        "standard_proguard_library(",
        "    name = 'exported_plugin',",
        "    proguard_specs = ['exported_plugin.cfg']",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//test:lib");
    Artifact validatedDep =
        getBinArtifact(
            "validated_proguard/dep/test/dep.cfg_valid", getConfiguredTarget("//test:dep"));
    Artifact validatedExport =
        getBinArtifact(
            "validated_proguard/export/test/export.cfg_valid",
            getConfiguredTarget("//test:export"));
    Artifact validatedRuntimeDep =
        getBinArtifact(
            "validated_proguard/runtime/test/runtime.cfg_valid",
            getConfiguredTarget("//test:runtime"));
    Artifact validatedPlugin =
        getBinArtifact(
            "validated_proguard/plugin/test/plugin.cfg_valid",
            getDirectPrerequisite(target, "//test:plugin"));
    Artifact validatedExportedPlugin =
        getBinArtifact(
            "validated_proguard/exported_plugin/test/exported_plugin.cfg_valid",
            getDirectPrerequisite(target, "//test:exported_plugin"));

    assertThat(getFilesToBuild(target).toList())
        .containsExactly(
            validatedDep,
            validatedExport,
            validatedRuntimeDep,
            validatedPlugin,
            validatedExportedPlugin);
  }

  @Test
  public void testProguardSpecs_customAttributes() throws Exception {
    scratch.file(
        "test/BUILD",
        "custom_proguard_library(",
        "    name = 'lib',",
        "    target_libs = [':target'],",
        "    host_libs = [':host'],",
        "    target_attrs = ['target_libs', '$implicit_target'],",
        "    host_attrs = ['host_libs', '$implicit_host'])",
        "standard_proguard_library(name = 'target', proguard_specs = ['target.cfg'])",
        "standard_proguard_library(name = 'host', proguard_specs = ['host.cfg'])");

    ConfiguredTarget target = getConfiguredTarget("//test:lib");
    Artifact validatedTarget =
        getBinArtifact(
            "validated_proguard/target/test/target.cfg_valid",
            getConfiguredTarget("//test:target"));
    Artifact validatedHost =
        getBinArtifact(
            "validated_proguard/host/test/host.cfg_valid",
            getDirectPrerequisite(target, "//test:host"));
    Artifact validatedImplicitTarget =
        getBinArtifact(
            "validated_proguard/implicit_target/test/implicit/implicit_target.cfg_valid",
            getConfiguredTarget("//test/implicit:implicit_target"));
    Artifact validatedImplicitHost =
        getBinArtifact(
            "validated_proguard/implicit_host/test/implicit/implicit_host.cfg_valid",
            getDirectPrerequisite(target, "//test/implicit:implicit_host"));

    assertThat(getFilesToBuild(target).toList())
        .containsExactly(
            validatedTarget, validatedHost, validatedImplicitTarget, validatedImplicitHost);
  }

  @Test
  public void testProguardSpecs_customAttributes_allowsAttributeMismatch() throws Exception {
    scratch.file(
        "test/BUILD",
        "custom_proguard_library(",
        "    name = 'lib',",
        "    target_libs = [':target'],",
        "    host_libs = [':host'],",
        "    target_attrs = ['deps', 'exports', 'runtime_deps'],",
        "    host_attrs = ['plugins', 'exported_plugins'])",
        "standard_proguard_library(name = 'target', proguard_specs = ['target.cfg'])",
        "standard_proguard_library(name = 'host', proguard_specs = ['host.cfg'])");

    // None of the attributes we have were specified, none of the attributes we specified exist.
    assertThat(getFilesToBuild(getConfiguredTarget("//test:lib")).toList()).isEmpty();
  }

  @Test
  public void testProguardSpecs_aspectAttributes_detectedByProguardLibrary() throws Exception {
    scratch.file(
        "test/BUILD",
        "aspect_proguard_library(",
        "    name = 'lib',",
        "    deps = [':child'])",
        "sh_library(name = 'child')");

    ConfiguredTarget target = getConfiguredTarget("//test:lib");

    Artifact validatedImplicitTarget =
        getBinArtifact(
            "validated_proguard/implicit_target/test/implicit/implicit_target.cfg_valid",
            getConfiguredTarget("//test/implicit:implicit_target"));

    // The only thing that should be picked up is the implicit target from the aspect.
    assertThat(getFilesToBuild(target).toList()).containsExactly(validatedImplicitTarget);
  }
}
