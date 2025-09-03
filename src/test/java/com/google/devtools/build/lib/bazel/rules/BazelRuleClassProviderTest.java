// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider.pathOrDefault;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.ShellConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider.StrictActionEnvOptions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.config.ConfigRules;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Options;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests consistency of {@link BazelRuleClassProvider}. */
@RunWith(JUnit4.class)
public class BazelRuleClassProviderTest extends BuildViewTestCase {

  private static void checkConfigConsistency(ConfiguredRuleClassProvider provider) {
    // Check that every fragment required by a rule is present.
    FragmentClassSet configurationFragments = provider.getFragmentRegistry().getAllFragments();
    for (RuleClass ruleClass : provider.getRuleClassMap().values()) {
      for (Class<?> fragment :
          ruleClass.getConfigurationFragmentPolicy().getRequiredConfigurationFragments()) {
        assertWithMessage(ruleClass.toString()).that(configurationFragments).contains(fragment);
      }
    }

    Set<Class<? extends FragmentOptions>> configOptions =
        provider.getFragmentRegistry().getOptionsClasses();
    for (Class<? extends Fragment> fragmentClass : configurationFragments) {
      // Check that every options class required for fragment creation is provided.
      for (Class<? extends FragmentOptions> options : Fragment.requiredOptions(fragmentClass)) {
        assertThat(configOptions).contains(options);
      }
    }
  }

  private static void checkModule(RuleSet top) {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    builder.setToolsRepository(RepositoryName.BAZEL_TOOLS);
    Set<RuleSet> result = new HashSet<>();
    result.add(BazelRuleClassProvider.BAZEL_SETUP);
    collectTransitiveClosure(result, top);
    for (RuleSet module : result) {
      module.init(builder);
    }
    ConfiguredRuleClassProvider provider = builder.build();
    assertThat(provider).isNotNull();
    checkConfigConsistency(provider);
  }

  private static void collectTransitiveClosure(Set<RuleSet> result, RuleSet module) {
    if (result.add(module)) {
      for (RuleSet dep : module.requires()) {
        collectTransitiveClosure(result, dep);
      }
    }
  }

  @Test
  public void coreConsistency() {
    checkModule(CoreRules.INSTANCE);
  }

  @Test
  public void genericConsistency() {
    checkModule(GenericRules.INSTANCE);
  }

  @Test
  public void configConsistency() {
    checkModule(ConfigRules.INSTANCE);
  }

  @Test
  public void protoConsistency() {
    checkModule(BazelRuleClassProvider.PROTO_RULES);
  }

  @Test
  public void cppConsistency() {
    checkModule(CcRules.INSTANCE);
  }

  @Test
  public void javaConsistency() {
    checkModule(JavaRules.INSTANCE);
  }

  @Test
  public void pythonConsistency() {
    checkModule(BazelRuleClassProvider.PYTHON_RULES);
  }

  @Test
  public void androidConsistency() {
    checkModule(BazelRuleClassProvider.ANDROID_RULES);
  }

  @Test
  public void objcConsistency() {
    checkModule(ObjcRules.INSTANCE);
  }

  @Test
  public void toolchainConsistency() {
    checkModule(ToolchainRules.INSTANCE);
  }

  @Test
  public void strictActionEnv() throws Exception {
    if (OS.getCurrent() == OS.WINDOWS) {
      return;
    }

    BuildOptions options =
        BuildOptions.of(
            ImmutableList.of(
                CoreOptions.class, ShellConfiguration.Options.class, StrictActionEnvOptions.class),
            "--experimental_strict_action_env",
            "--action_env=FOO=bar");

    ActionEnvironment env = BazelRuleClassProvider.SHELL_ACTION_ENV.apply(options);
    assertThat(env.getFixedEnv()).containsEntry("PATH", "/bin:/usr/bin:/usr/local/bin");
    assertThat(env.getFixedEnv()).containsEntry("FOO", "bar");
  }

  @Test
  public void pathOrDefaultOnLinux() {
    assertThat(pathOrDefault(OS.LINUX, null, null)).isEqualTo("/bin:/usr/bin:/usr/local/bin");
    assertThat(pathOrDefault(OS.LINUX, "/not/bin", null)).isEqualTo("/bin:/usr/bin:/usr/local/bin");
  }

  @Test
  public void pathOrDefaultOnWindows() {
    String defaultWindowsPath = "";
    String systemRoot = System.getenv("SYSTEMROOT");
    if (Strings.isNullOrEmpty(systemRoot)) {
      systemRoot = "C:\\Windows";
    }
    defaultWindowsPath += ";" + systemRoot;
    defaultWindowsPath += ";" + systemRoot + "\\System32";
    defaultWindowsPath += ";" + systemRoot + "\\System32\\WindowsPowerShell\\v1.0";
    assertThat(pathOrDefault(OS.WINDOWS, null, null)).isEqualTo(defaultWindowsPath);
    assertThat(pathOrDefault(OS.WINDOWS, "C:/mypath", null))
        .isEqualTo(defaultWindowsPath + ";C:/mypath");
    assertThat(pathOrDefault(OS.WINDOWS, "C:/mypath", PathFragment.create("D:/foo/shell")))
        .isEqualTo("D:\\foo" + defaultWindowsPath + ";C:/mypath");
  }

  @Test
  public void optionsAlsoApplyToHost() throws Exception {
    BuildOptions options = targetConfig.getOptions().clone();
    var strictActionEnvOptions = options.get(StrictActionEnvOptions.class);
    if (strictActionEnvOptions == null) {
      // This Bazel build doesn't include StrictActionEnvOptions. Nothing to test.
      return;
    }
    strictActionEnvOptions.useStrictActionEnv = true;

    StrictActionEnvOptions h =
        AnalysisTestUtil.execOptions(options, skyframeExecutor, reporter)
            .get(StrictActionEnvOptions.class);

    assertThat(h.useStrictActionEnv).isTrue();
  }

  @Test
  public void getShellExecutableUnset() {
    assertThat(determineShellExecutable(OS.LINUX, null))
        .isEqualTo(PathFragment.create("/bin/bash"));
    assertThat(determineShellExecutable(OS.FREEBSD, null))
        .isEqualTo(PathFragment.create("/usr/local/bin/bash"));
    assertThat(determineShellExecutable(OS.OPENBSD, null))
        .isEqualTo(PathFragment.create("/usr/local/bin/bash"));
    assertThat(determineShellExecutable(OS.WINDOWS, null))
        .isEqualTo(PathFragment.create("c:/msys64/usr/bin/bash.exe"));
  }

  @Test
  public void getShellExecutableIfSet() {
    PathFragment binBash = PathFragment.create("/bin/bash");
    assertThat(determineShellExecutable(OS.LINUX, binBash))
        .isEqualTo(PathFragment.create("/bin/bash"));
    assertThat(determineShellExecutable(OS.FREEBSD, binBash))
        .isEqualTo(PathFragment.create("/bin/bash"));
    assertThat(determineShellExecutable(OS.OPENBSD, binBash))
        .isEqualTo(PathFragment.create("/bin/bash"));
    assertThat(determineShellExecutable(OS.WINDOWS, binBash))
        .isEqualTo(PathFragment.create("/bin/bash"));
  }

  private static PathFragment determineShellExecutable(OS os, PathFragment executableOption) {
    ShellConfiguration.Options options = Options.getDefaults(ShellConfiguration.Options.class);
    options.shellExecutable = executableOption;
    return BazelRuleClassProvider.getShellExecutableForOs(os, options);
  }
}
