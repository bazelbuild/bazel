package com.google.devtools.build.lib.rules.cpp;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.baseArtifactNames;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class Cpp20ModuleCompileTest extends BuildViewTestCase {
  private void enableCpp20Module() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfig(
        mockToolsConfig, Crosstool.CcToolchainConfig.builder().withFeatures(
                             CppRuleClasses.CPP20_MODULE));
  }
  // if we use module_interfaces
  // we need to enable cpp20_module feature
  @Test
  public void testCpp20ModuleFeatureNotEnabled1() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file("foo/BUILD",
                 "cc_library(name='foo', module_interfaces=['foo.cppm'])");
    scratch.file("foo/foo.cppm", "foo");
    getConfiguredTarget("//foo:foo");
    assertContainsEvent(
        "to use C++20 Modules, the feature cpp20_module must be enabled");
  }

  // when use C++20 Modules, the compile model changed
  // to keep backward-compatibility
  // the feature cpp20_module is not enabled by default
  // we need to enable the feature in rule attr features
  @Test
  public void testCpp20ModuleFeatureNotEnabled2() throws Exception {
    enableCpp20Module();
    reporter.removeHandler(failFastHandler);
    scratch.file("foo/BUILD",
                 "cc_library(name='foo', module_interfaces=['foo.cppm'])");
    scratch.file("foo/foo.cppm", "foo");
    getConfiguredTarget("//foo:foo");
    assertContainsEvent(
        "to use C++20 Modules, the feature cpp20_module must be enabled");
  }

  // add the feature cpp20_module to toolchain
  // add the feature cpp20_module to rule attr features
  // now we can play with C++20 Modules
  @Test
  public void testCpp20ModuleFeatureEnabled() throws Exception {
    enableCpp20Module();
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "foo/BUILD",
        "cc_library(name='foo', module_interfaces=['foo.cppm'], features=['cpp20_module'])");
    scratch.file("foo/foo.cppm", "foo");
    getConfiguredTarget("//foo:foo");
    assertDoesNotContainEvent(
        "to use C++20 Modules, the feature cpp20_module must be enabled");
  }

  // module_interfaces cannot have two same .cc files
  @Test
  public void testSameCcFileTwice1() throws Exception {
    enableCpp20Module();
    scratch.file(
        "a/BUILD",
        "cc_library(name='a', module_interfaces=['a1', 'a2'], features=['cpp20_module'])",
        "filegroup(name='a1', srcs=['a.cc'])",
        "filegroup(name='a2', srcs=['a.cc'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("Artifact 'a/a.cc' is duplicated");
  }

  // if one file has already in srcs,
  // it can not put to module_interfaces
  // due to module_interfaces is a special srcs that produce bmi files
  @Test
  public void testSameCcFileTwice2() throws Exception {
    enableCpp20Module();
    scratch.file(
        "a/BUILD",
        "cc_library(name='a', srcs=['a1'], module_interfaces=['a2'], features=['cpp20_module'])",
        "filegroup(name='a1', srcs=['a.cc'])",
        "filegroup(name='a2', srcs=['a.cc'])");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//a:a");
    assertContainsEvent("Artifact 'a/a.cc' is duplicated");
  }
  @Test
  public void testObjectFile() throws Exception {
    enableCpp20Module();
    useConfiguration("--cpu=k8");
    ConfiguredTarget archiveInSrcsTest = scratchConfiguredTarget(
        "cc_in_module_interfaces", "cc_in_module_interfaces_test",
        "cc_test(name = 'cc_in_module_interfaces_test',",
        "           module_interfaces = ['foo.cc'],",
        "           features = ['cpp20_module'],", ")");
    List<String> artifactNames =
        baseArtifactNames(getLinkerInputs(archiveInSrcsTest));
    assertThat(artifactNames).contains("foo.o");
  }

  private Iterable<Artifact> getLinkerInputs(ConfiguredTarget target) {
    Artifact executable = getExecutable(target);
    SpawnAction linkAction = (SpawnAction)getGeneratingAction(executable);
    return linkAction.getInputs().toList();
  }

  // use --precompile to compile .cppm to .pcm
  @Test
  public void testModuleCompileOption() throws Exception {
    enableCpp20Module();
    useConfiguration("--cpu=k8");
    scratch.file("a/BUILD", "cc_library(", "name='foo', ",
                 "features = ['cpp20_module'], ",
                 "module_interfaces=['foo.cppm']", ")");
    ConfiguredTarget target = getConfiguredTarget("//a:foo");
    List<CppCompileAction> compilationSteps =
        actionsTestUtil().findTransitivePrerequisitesOf(
            getFilesToBuild(target).toList().get(0), CppCompileAction.class);
    assertThat(compilationSteps.get(1).getArguments()).contains("--precompile");
  }

  @Test
  public void testModuleCompileOnly() throws Exception {
    enableCpp20Module();
    useConfiguration("--cpu=k8");
    scratch.file("a/BUILD", "cc_library(", "name='foo', ",
                 "features = ['cpp20_module'], ",
                 "module_interfaces=['foo.cppm']", ")");
    ConfiguredTarget target = getConfiguredTarget("//a:foo");
    var outputs =
        getOutputGroup(target, OutputGroupInfo.FILES_TO_COMPILE).toList();
    assertThat(outputs).isNotEmpty();

    assertThat(getArtifactByExecPathSuffix(target, "/foo.o")).isNotNull();
    assertThat(getArtifactByExecPathSuffix(target, "/foo.pcm.o")).isNull();
  }

  protected static Artifact getArtifactByExecPathSuffix(ConfiguredTarget target,
                                                        String path) {
    for (Artifact artifact :
         getOutputGroup(target, OutputGroupInfo.FILES_TO_COMPILE).toList()) {
      if (artifact.getExecPathString().endsWith(path)) {
        return artifact;
      }
    }
    return null;
  }
}
