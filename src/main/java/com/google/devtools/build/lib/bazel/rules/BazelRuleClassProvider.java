// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.ShellConfiguration;
import com.google.devtools.build.lib.analysis.ShellConfiguration.ShellExecutableProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.ActionEnvironmentProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAarImportRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidBinaryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidInstrumentationTestRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidLibraryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidLocalTestRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidSdkRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidSemantics;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppSemantics;
import com.google.devtools.build.lib.bazel.rules.cpp.proto.BazelCcProtoAspect;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaLiteProtoAspect;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaLiteProtoLibraryRule;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaProtoAspect;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaProtoLibraryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyBinaryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyLibraryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyRuleClasses;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyRuntimeRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyTestRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPythonConfiguration;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenServerRule;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.rules.android.AarImportBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidDeviceBrokerInfo;
import com.google.devtools.build.lib.rules.android.AndroidDeviceRule;
import com.google.devtools.build.lib.rules.android.AndroidDeviceScriptFixtureRule;
import com.google.devtools.build.lib.rules.android.AndroidHostServiceFixtureRule;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationInfo;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationTestBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLibraryBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidNativeLibsInfo;
import com.google.devtools.build.lib.rules.android.AndroidNeverlinkAspect;
import com.google.devtools.build.lib.rules.android.AndroidResourcesInfo;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidToolsDefaultsJarRule;
import com.google.devtools.build.lib.rules.android.AndroidSdkBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidSkylarkCommon;
import com.google.devtools.build.lib.rules.android.ApkInfo;
import com.google.devtools.build.lib.rules.android.DexArchiveAspect;
import com.google.devtools.build.lib.rules.config.ConfigRules;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.cpp.proto.CcProtoAspect;
import com.google.devtools.build.lib.rules.cpp.proto.CcProtoLibraryRule;
import com.google.devtools.build.lib.rules.platform.PlatformRules;
import com.google.devtools.build.lib.rules.proto.BazelProtoLibraryRule;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainRule;
import com.google.devtools.build.lib.rules.python.PythonConfigurationLoader;
import com.google.devtools.build.lib.rules.repository.CoreWorkspaceRules;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.test.TestingSupportRules;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidBootstrap;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/** A rule class provider implementing the rules Bazel knows. */
public class BazelRuleClassProvider {
  public static final String TOOLS_REPOSITORY = "@bazel_tools";

  /** Command-line options. */
  public static class StrictActionEnvOptions extends FragmentOptions {
    @Option(
        name = "incompatible_strict_action_env",
        oldName = "experimental_strict_action_env",
        defaultValue = "true",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
        metadataTags = {
          OptionMetadataTag.INCOMPATIBLE_CHANGE,
          OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
        },
        help =
            "If true, Bazel uses an environment with a static value for PATH and does not "
                + "inherit LD_LIBRARY_PATH or TMPDIR. Use --action_env=ENV_VARIABLE if you want to "
                + "inherit specific environment variables from the client, but note that doing so "
                + "can prevent cross-user caching if a shared cache is used.")
    public boolean useStrictActionEnv;

    @Override
    public StrictActionEnvOptions getHost() {
      StrictActionEnvOptions host = (StrictActionEnvOptions) getDefault();
      host.useStrictActionEnv = useStrictActionEnv;
      return host;
    }
  }

  private static final PathFragment FALLBACK_SHELL = PathFragment.create("/bin/bash");

  public static final ShellExecutableProvider SHELL_EXECUTABLE = (BuildOptions options) ->
      ShellConfiguration.Loader.determineShellExecutable(
          OS.getCurrent(),
          options.get(ShellConfiguration.Options.class),
          FALLBACK_SHELL);

  public static final ActionEnvironmentProvider SHELL_ACTION_ENV = (BuildOptions options) -> {
    boolean strictActionEnv = options.get(StrictActionEnvOptions.class).useStrictActionEnv;
    OS os = OS.getCurrent();
    PathFragment shellExecutable = SHELL_EXECUTABLE.getShellExecutable(options);
    TreeMap<String, String> env = new TreeMap<>();

    // All entries in the builder that have a value of null inherit the value from the client
    // environment, which is only known at execution time - we don't want to bake the client env
    // into the configuration since any change to the configuration requires rerunning the full
    // analysis phase.
    if (!strictActionEnv) {
      env.put("LD_LIBRARY_PATH", null);
    }

    if (strictActionEnv) {
      env.put("PATH", pathOrDefault(os, null, shellExecutable));
    } else if (os == OS.WINDOWS) {
      // TODO(ulfjack): We want to add the MSYS root to the PATH, but that prevents us from
      // inheriting PATH from the client environment. For now we use System.getenv even though
      // that is incorrect. We should enable strict_action_env by default and then remove this
      // code, but that change may break Windows users who are relying on the MSYS root being in
      // the PATH.
      env.put("PATH", pathOrDefault(
          os, System.getenv("PATH"), shellExecutable));
    } else {
      // The previous implementation used System.getenv (which uses the server's environment), and
      // fell back to a hard-coded "/bin:/usr/bin" if PATH was not set.
      env.put("PATH", null);
    }

    // Shell environment variables specified via options take precedence over the
    // ones inherited from the fragments. In the long run, these fragments will
    // be replaced by appropriate default rc files anyway.
    for (Map.Entry<String, String> entry :
        options.get(BuildConfiguration.Options.class).actionEnvironment) {
      env.put(entry.getKey(), entry.getValue());
    }

    return ActionEnvironment.split(env);
  };

  /** Used by the build encyclopedia generator. */
  public static ConfiguredRuleClassProvider create() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    builder.setToolsRepository(TOOLS_REPOSITORY);
    setup(builder);
    return builder.build();
  }

  public static void setup(ConfiguredRuleClassProvider.Builder builder) {
    for (RuleSet ruleSet : RULE_SETS) {
      ruleSet.init(builder);
    }
  }

  public static final RuleSet BAZEL_SETUP =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          builder
              .setPrelude("//tools/build_rules:prelude_bazel")
              .setRunfilesPrefix(LabelConstants.DEFAULT_REPOSITORY_DIRECTORY)
              .setPrerequisiteValidator(new BazelPrerequisiteValidator())
              .setActionEnvironmentProvider(SHELL_ACTION_ENV);

          builder.addConfigurationOptions(ShellConfiguration.Options.class);
          builder.addConfigurationFragment(
              new ShellConfiguration.Loader(
                  SHELL_EXECUTABLE,
                  ShellConfiguration.Options.class,
                  StrictActionEnvOptions.class));
          builder.addUniversalConfigurationFragment(ShellConfiguration.class);
          builder.addConfigurationOptions(StrictActionEnvOptions.class);
          builder.addConfigurationOptions(BuildConfiguration.Options.class);
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of();
        }
      };

  public static final RuleSet PROTO_RULES =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          builder.addConfigurationOptions(ProtoConfiguration.Options.class);
          builder.addConfigurationFragment(new ProtoConfiguration.Loader());
          builder.addRuleDefinition(new BazelProtoLibraryRule());
          builder.addRuleDefinition(new ProtoLangToolchainRule());
          builder.addSkylarkAccessibleTopLevels(ProtoInfo.SKYLARK_NAME, ProtoInfo.PROVIDER);
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE);
        }
      };

  public static final RuleSet CPP_PROTO_RULES =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          CcProtoAspect ccProtoAspect = new BazelCcProtoAspect(BazelCppSemantics.INSTANCE, builder);
          builder.addNativeAspectClass(ccProtoAspect);
          builder.addRuleDefinition(new CcProtoLibraryRule(ccProtoAspect));
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE);
        }
      };

  public static final RuleSet JAVA_PROTO_RULES =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          BazelJavaProtoAspect bazelJavaProtoAspect = new BazelJavaProtoAspect(builder);
          BazelJavaLiteProtoAspect bazelJavaLiteProtoAspect = new BazelJavaLiteProtoAspect(builder);
          builder.addNativeAspectClass(bazelJavaProtoAspect);
          builder.addNativeAspectClass(bazelJavaLiteProtoAspect);
          builder.addRuleDefinition(new BazelJavaProtoLibraryRule(bazelJavaProtoAspect));
          builder.addRuleDefinition(new BazelJavaLiteProtoLibraryRule(bazelJavaLiteProtoAspect));
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE, JavaRules.INSTANCE);
        }
      };

  public static final RuleSet ANDROID_RULES =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          String toolsRepository = checkNotNull(builder.getToolsRepository());

          builder.addConfigurationFragment(new AndroidConfiguration.Loader());
          builder.addConfigurationFragment(new AndroidLocalTestConfiguration.Loader());

          AndroidNeverlinkAspect androidNeverlinkAspect = new AndroidNeverlinkAspect();
          DexArchiveAspect dexArchiveAspect = new DexArchiveAspect(toolsRepository);
          builder.addNativeAspectClass(androidNeverlinkAspect);
          builder.addNativeAspectClass(dexArchiveAspect);

          builder.addRuleDefinition(new AndroidSdkBaseRule());
          builder.addRuleDefinition(new BazelAndroidSdkRule());
          builder.addRuleDefinition(new AndroidToolsDefaultsJarRule());
          builder.addRuleDefinition(new AndroidRuleClasses.AndroidBaseRule());
          builder.addRuleDefinition(new AndroidRuleClasses.AndroidResourceSupportRule());
          builder.addRuleDefinition(
              new AndroidRuleClasses.AndroidBinaryBaseRule(
                  androidNeverlinkAspect, dexArchiveAspect));
          builder.addRuleDefinition(new AndroidLibraryBaseRule(androidNeverlinkAspect));
          builder.addRuleDefinition(new BazelAndroidLibraryRule());
          builder.addRuleDefinition(new BazelAndroidBinaryRule());
          builder.addRuleDefinition(new AarImportBaseRule());
          builder.addRuleDefinition(new BazelAarImportRule());
          builder.addRuleDefinition(new AndroidDeviceRule());
          builder.addRuleDefinition(new AndroidLocalTestBaseRule());
          builder.addRuleDefinition(new BazelAndroidLocalTestRule());
          builder.addRuleDefinition(new AndroidInstrumentationTestBaseRule());
          builder.addRuleDefinition(new BazelAndroidInstrumentationTestRule());
          builder.addRuleDefinition(new AndroidDeviceScriptFixtureRule());
          builder.addRuleDefinition(new AndroidHostServiceFixtureRule());

          AndroidBootstrap bootstrap =
              new AndroidBootstrap(
                  new AndroidSkylarkCommon(),
                  ApkInfo.PROVIDER,
                  AndroidInstrumentationInfo.PROVIDER,
                  AndroidDeviceBrokerInfo.PROVIDER,
                  AndroidResourcesInfo.PROVIDER,
                  AndroidNativeLibsInfo.PROVIDER);
          builder.addSkylarkBootstrap(bootstrap);

          try {
            builder.addWorkspaceFilePrefix(
                ResourceFileLoader.loadResource(BazelAndroidSemantics.class, "android.WORKSPACE"));
          } catch (IOException e) {
            throw new IllegalStateException(e);
          }
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE, JavaRules.INSTANCE);
        }
      };

  public static final RuleSet PYTHON_RULES =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          builder.addConfigurationFragment(new PythonConfigurationLoader());
          builder.addConfigurationFragment(new BazelPythonConfiguration.Loader());

          builder.addRuleDefinition(new BazelPyRuleClasses.PyBaseRule());
          builder.addRuleDefinition(new BazelPyRuleClasses.PyBinaryBaseRule());
          builder.addRuleDefinition(new BazelPyLibraryRule());
          builder.addRuleDefinition(new BazelPyBinaryRule());
          builder.addRuleDefinition(new BazelPyTestRule());
          builder.addRuleDefinition(new BazelPyRuntimeRule());
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE, CcRules.INSTANCE);
        }
      };

  public static final RuleSet VARIOUS_WORKSPACE_RULES =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          // TODO(ulfjack): Split this up by conceptual units.
          builder.addRuleDefinition(new MavenJarRule());
          builder.addRuleDefinition(new MavenServerRule());
          builder.addRuleDefinition(new NewLocalRepositoryRule());
          builder.addRuleDefinition(new AndroidSdkRepositoryRule());
          builder.addRuleDefinition(new AndroidNdkRepositoryRule());
          builder.addRuleDefinition(new LocalConfigPlatformRule());

          try {
            builder.addWorkspaceFilePrefix(
                ResourceFileLoader.loadResource(
                    LocalConfigPlatformRule.class, "local_config_platform.WORKSPACE"));
          } catch (IOException e) {
            throw new IllegalStateException(e);
          }
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE, CoreWorkspaceRules.INSTANCE);
        }
      };

  private static final ImmutableSet<RuleSet> RULE_SETS =
      ImmutableSet.of(
          BAZEL_SETUP,
          CoreRules.INSTANCE,
          CoreWorkspaceRules.INSTANCE,
          GenericRules.INSTANCE,
          ConfigRules.INSTANCE,
          PlatformRules.INSTANCE,
          PROTO_RULES,
          ShRules.INSTANCE,
          CcRules.INSTANCE,
          CPP_PROTO_RULES,
          JavaRules.INSTANCE,
          JAVA_PROTO_RULES,
          ANDROID_RULES,
          PYTHON_RULES,
          ObjcRules.INSTANCE,
          J2ObjcRules.INSTANCE,
          TestingSupportRules.INSTANCE,
          VARIOUS_WORKSPACE_RULES,
          // This rule set is a little special: it needs to depend on every configuration fragment
          // that has Make variables, so we put it last.
          ToolchainRules.INSTANCE);

  @VisibleForTesting
  public static String pathOrDefault(OS os, @Nullable String path, @Nullable PathFragment sh) {
    // TODO(ulfjack): The default PATH should be set from the exec platform, which may be different
    // from the local machine. For now, this can be overridden with --action_env=PATH=<value>, so
    // at least there's a workaround.
    if (os != OS.WINDOWS) {
      return "/bin:/usr/bin";
    }

    // Attempt to compute the MSYS root (the real Windows path of "/") from `sh`.
    if (sh != null && sh.getParentDirectory() != null) {
      String newPath = sh.getParentDirectory().getPathString();
      if (sh.getParentDirectory().endsWith(PathFragment.create("usr/bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().getParentDirectory().replaceName("bin").getPathString();
      } else if (sh.getParentDirectory().endsWith(PathFragment.create("bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().replaceName("usr").getRelative("bin").getPathString();
      }
      newPath = newPath.replace('/', '\\');

      if (path != null) {
        newPath += ";" + path;
      }
      return newPath;
    } else {
      return null;
    }
  }
}
