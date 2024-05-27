package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment;
import com.google.devtools.build.lib.packages.util.Crosstool;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class Cpp20ModulesConfiguredTargetTest extends BuildViewTestCase {

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addConfigurationFragment(DummyTestFragment.class);
    return builder.addRuleDefinition(new TestRuleClassProvider.MakeVariableTesterRule()).build();
  }

  @Test
  public void testCpp20ModulesConfigurationNoFlags() throws Exception {
    ImmutableList<String> targetList = ImmutableList.of("//foo:lib", "//foo:bin", "//foo:test");
    scratch.file(
        "foo/BUILD",
        """
                cc_library(
                    name = 'lib',
                    module_interfaces = ["foo.cppm"],
                )
                cc_binary(
                    name = 'bin',
                    module_interfaces = ["foo.cppm"],
                )
                cc_test(
                    name = 'test',
                    module_interfaces = ["foo.cppm"],
                )
                """);
    for(String targetName: targetList) {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget(targetName));
      assertThat(e).hasMessageThat().contains("requires --experimental_cpp20_modules");
    }
  }
  @Test
  public void testCpp20ModulesConfigurationNoFeatures() throws Exception {
    ImmutableList<String> targetList = ImmutableList.of("//foo:lib", "//foo:bin", "//foo:test");
    useConfiguration("--experimental_cpp20_modules");
    scratch.file(
        "foo/BUILD",
        """
                cc_library(
                    name = 'lib',
                    module_interfaces = ["foo.cppm"],
                )
                cc_binary(
                    name = 'bin',
                    module_interfaces = ["foo.cppm"],
                )
                cc_test(
                    name = 'test',
                    module_interfaces = ["foo.cppm"],
                )
                """);
    for(String targetName: targetList) {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget(targetName));
      assertThat(e).hasMessageThat().contains("the feature cpp20_modules must be enabled");
    }
  }
  @Test
  public void testCpp20ModulesConfigurationWithFeatures() throws Exception {
    ImmutableList<String> targetList = ImmutableList.of("//foo:lib", "//foo:bin", "//foo:test");
    AnalysisMock.get()
            .ccSupport()
            .setupCcToolchainConfig(
                    mockToolsConfig,
                    Crosstool.CcToolchainConfig.builder()
                            .withFeatures(
                                    CppRuleClasses.CPP20_MODULES));
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");
    scratch.file(
        "foo/BUILD",
        """
                cc_library(
                    name = 'lib',
                    module_interfaces = ["foo.cppm"],
                )
                cc_binary(
                    name = 'bin',
                    module_interfaces = ["foo.cppm"],
                )
                cc_test(
                    name = 'test',
                    module_interfaces = ["foo.cppm"],
                )
                """);
    for(String targetName: targetList) {
      ImmutableSet<String> features = getRuleContext(getConfiguredTarget(targetName)).getFeatures();
      assertThat(features).contains("cpp20_modules");
    }
  }
}
