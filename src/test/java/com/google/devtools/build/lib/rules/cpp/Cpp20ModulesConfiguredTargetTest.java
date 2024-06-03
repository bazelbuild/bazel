package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.Crosstool;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;

@RunWith(JUnit4.class)
public class Cpp20ModulesConfiguredTargetTest extends BuildViewTestCase {
  @Before
  public void setupBasicRulesWithModules() throws IOException {
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
  }
  @Test
  public void testCpp20ModulesConfigurationNoFlags() {
    {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:lib"));
      assertThat(e).hasMessageThat().contains("requires --experimental_cpp20_modules");
    }
    {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:bin"));
      assertThat(e).hasMessageThat().contains("requires --experimental_cpp20_modules");
    }
    {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:test"));
      assertThat(e).hasMessageThat().contains("requires --experimental_cpp20_modules");
    }
  }
  @Test
  public void testCpp20ModulesConfigurationNoFeatures() throws Exception {
    useConfiguration("--experimental_cpp20_modules");
    {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:lib"));
      assertThat(e).hasMessageThat().contains("the feature cpp20_modules must be enabled");
    }
    {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:bin"));
      assertThat(e).hasMessageThat().contains("the feature cpp20_modules must be enabled");
    }
    {
      AssertionError e = assertThrows(AssertionError.class, () -> getConfiguredTarget("//foo:test"));
      assertThat(e).hasMessageThat().contains("the feature cpp20_modules must be enabled");
    }
  }
  @Test
  public void testCpp20ModulesConfigurationWithFeatures() throws Exception {
    AnalysisMock.get()
            .ccSupport()
            .setupCcToolchainConfig(
                    mockToolsConfig,
                    Crosstool.CcToolchainConfig.builder()
                            .withFeatures(
                                    CppRuleClasses.CPP20_MODULES));
    useConfiguration("--experimental_cpp20_modules", "--features=cpp20_modules");
    {
      ImmutableSet<String> features = getRuleContext(getConfiguredTarget("//foo:lib")).getFeatures();
      assertThat(features).contains("cpp20_modules");
    }
    {
      ImmutableSet<String> features = getRuleContext(getConfiguredTarget("//foo:bin")).getFeatures();
      assertThat(features).contains("cpp20_modules");
    }
    {
      ImmutableSet<String> features = getRuleContext(getConfiguredTarget("//foo:test")).getFeatures();
      assertThat(features).contains("cpp20_modules");
    }
  }
}
