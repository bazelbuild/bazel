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
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.ShellConfiguration;
import com.google.devtools.build.lib.analysis.ShellConfiguration.ShellExecutableProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.ActionEnvironmentProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.bazel.repository.LocalConfigPlatformRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAarImportRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidBinaryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidDevice;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidDeviceScriptFixture;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidHostServiceFixture;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidInstrumentationTestRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidLibraryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidLocalTestRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidSdkRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidSemantics;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidToolsDefaultsJar;
import com.google.devtools.build.lib.bazel.rules.android.BazelSdkToolchainRule;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppSemantics;
import com.google.devtools.build.lib.bazel.rules.cpp.proto.BazelCcProtoAspect;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaLiteProtoAspect;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaLiteProtoLibraryRule;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaProtoAspect;
import com.google.devtools.build.lib.bazel.rules.java.proto.BazelJavaProtoLibraryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyBinaryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyLibraryRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyRuleClasses;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyTestRule;
import com.google.devtools.build.lib.bazel.rules.python.BazelPythonConfiguration;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.packages.RuleClass.Builder.ThirdPartyLicenseExistencePolicy;
import com.google.devtools.build.lib.rules.android.AarImportBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidApplicationResourceInfo;
import com.google.devtools.build.lib.rules.android.AndroidAssetsInfo;
import com.google.devtools.build.lib.rules.android.AndroidBinaryDataInfo;
import com.google.devtools.build.lib.rules.android.AndroidCcLinkParamsProvider;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidDeviceBrokerInfo;
import com.google.devtools.build.lib.rules.android.AndroidDeviceRule;
import com.google.devtools.build.lib.rules.android.AndroidDeviceScriptFixtureRule;
import com.google.devtools.build.lib.rules.android.AndroidFeatureFlagSetProvider;
import com.google.devtools.build.lib.rules.android.AndroidHostServiceFixtureRule;
import com.google.devtools.build.lib.rules.android.AndroidIdeInfoProvider;
import com.google.devtools.build.lib.rules.android.AndroidIdlProvider;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationInfo;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationTestBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLibraryAarInfo;
import com.google.devtools.build.lib.rules.android.AndroidLibraryBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLibraryResourceClassJarProvider;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidManifestInfo;
import com.google.devtools.build.lib.rules.android.AndroidNativeLibsInfo;
import com.google.devtools.build.lib.rules.android.AndroidNeverlinkAspect;
import com.google.devtools.build.lib.rules.android.AndroidPreDexJarProvider;
import com.google.devtools.build.lib.rules.android.AndroidProguardInfo;
import com.google.devtools.build.lib.rules.android.AndroidResourcesInfo;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidToolsDefaultsJarRule;
import com.google.devtools.build.lib.rules.android.AndroidSdkBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidSdkProvider;
import com.google.devtools.build.lib.rules.android.AndroidStarlarkCommon;
import com.google.devtools.build.lib.rules.android.ApkInfo;
import com.google.devtools.build.lib.rules.android.DexArchiveAspect;
import com.google.devtools.build.lib.rules.android.ProguardMappingProvider;
import com.google.devtools.build.lib.rules.android.databinding.DataBindingV2Provider;
import com.google.devtools.build.lib.rules.config.ConfigRules;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.cpp.proto.CcProtoAspect;
import com.google.devtools.build.lib.rules.cpp.proto.CcProtoLibraryRule;
import com.google.devtools.build.lib.rules.platform.PlatformRules;
import com.google.devtools.build.lib.rules.proto.BazelProtoCommon;
import com.google.devtools.build.lib.rules.proto.BazelProtoLibraryRule;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoInfo;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainRule;
import com.google.devtools.build.lib.rules.python.PyInfo;
import com.google.devtools.build.lib.rules.python.PyRuleClasses.PySymlink;
import com.google.devtools.build.lib.rules.python.PyRuntimeInfo;
import com.google.devtools.build.lib.rules.python.PyRuntimeRule;
import com.google.devtools.build.lib.rules.python.PyStarlarkTransitions;
import com.google.devtools.build.lib.rules.python.PythonConfigurationLoader;
import com.google.devtools.build.lib.rules.repository.CoreWorkspaceRules;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.test.TestingSupportRules;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.proto.ProtoBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.python.PyBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.stubs.ProviderStub;
import com.google.devtools.build.lib.starlarkbuildapi.stubs.StarlarkAspectStub;
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
        defaultValue = "false",
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

  public static final ActionEnvironmentProvider SHELL_ACTION_ENV =
      (BuildOptions options) -> {
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
          env.put("PATH", pathOrDefault(os, System.getenv("PATH"), shellExecutable));
        } else {
          // The previous implementation used System.getenv (which uses the server's environment),
          // and fell back to a hard-coded "/bin:/usr/bin" if PATH was not set.
          env.put("PATH", null);
        }

        // Shell environment variables specified via options take precedence over the
        // ones inherited from the fragments. In the long run, these fragments will
        // be replaced by appropriate default rc files anyway.
        for (Map.Entry<String, String> entry : options.get(CoreOptions.class).actionEnvironment) {
          env.put(entry.getKey(), entry.getValue());
        }

        if (!BuildConfiguration.runfilesEnabled(options.get(CoreOptions.class))) {
          // Setting this environment variable is for telling the binary running
          // in a Bazel action when to use runfiles library or runfiles tree.
          // The downside is that it will discard cache for all actions once
          // --enable_runfiles changes, but this also prevents wrong caching result if a binary
          // behaves differently with and without runfiles tree.
          env.put("RUNFILES_MANIFEST_ONLY", "1");
        }

        return ActionEnvironment.split(env);
      };

  /** Used by the build encyclopedia generator. */
  public static ConfiguredRuleClassProvider create() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    builder.setToolsRepository(TOOLS_REPOSITORY);
    builder.setThirdPartyLicenseExistencePolicy(ThirdPartyLicenseExistencePolicy.NEVER_CHECK);
    setup(builder);
    return builder.build();
  }

  public static void setup(ConfiguredRuleClassProvider.Builder builder) {
    for (RuleSet ruleSet : RULE_SETS) {
      ruleSet.init(builder);
    }
    builder.setThirdPartyLicenseExistencePolicy(ThirdPartyLicenseExistencePolicy.NEVER_CHECK);
  }

  public static final RuleSet BAZEL_SETUP =
      new RuleSet() {
        @Override
        public void init(ConfiguredRuleClassProvider.Builder builder) {
          builder
              .setPrelude("//tools/build_rules:prelude_bazel")
              .setRunfilesPrefix(LabelConstants.DEFAULT_REPOSITORY_DIRECTORY)
              .setPrerequisiteValidator(new BazelPrerequisiteValidator())
              .setActionEnvironmentProvider(SHELL_ACTION_ENV)
              .addConfigurationOptions(ShellConfiguration.Options.class)
              .addConfigurationFragment(
                  new ShellConfiguration.Loader(
                      SHELL_EXECUTABLE,
                      ShellConfiguration.Options.class,
                      StrictActionEnvOptions.class))
              .addUniversalConfigurationFragment(ShellConfiguration.class)
              .addUniversalConfigurationFragment(PlatformConfiguration.class)
              .addConfigurationOptions(StrictActionEnvOptions.class)
              .addConfigurationOptions(CoreOptions.class);
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

          ProtoBootstrap bootstrap =
              new ProtoBootstrap(
                  ProtoInfo.PROVIDER,
                  BazelProtoCommon.INSTANCE,
                  new StarlarkAspectStub(),
                  new ProviderStub());
          builder.addStarlarkBootstrap(bootstrap);
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
          builder.addRuleDefinition(
              new AndroidToolsDefaultsJarRule(BazelAndroidToolsDefaultsJar.class));
          builder.addRuleDefinition(new AndroidRuleClasses.AndroidBaseRule());
          builder.addRuleDefinition(new AndroidRuleClasses.AndroidResourceSupportRule());
          builder.addRuleDefinition(
              new AndroidRuleClasses.AndroidBinaryBaseRule(
                  androidNeverlinkAspect, dexArchiveAspect));
          builder.addRuleDefinition(new BazelSdkToolchainRule());
          builder.addRuleDefinition(new AndroidLibraryBaseRule(androidNeverlinkAspect));
          builder.addRuleDefinition(new BazelAndroidLibraryRule());
          builder.addRuleDefinition(new BazelAndroidBinaryRule());
          builder.addRuleDefinition(new AarImportBaseRule());
          builder.addRuleDefinition(new BazelAarImportRule());
          builder.addRuleDefinition(new AndroidDeviceRule(BazelAndroidDevice.class));
          builder.addRuleDefinition(new AndroidLocalTestBaseRule());
          builder.addRuleDefinition(new BazelAndroidLocalTestRule());
          builder.addRuleDefinition(new AndroidInstrumentationTestBaseRule());
          builder.addRuleDefinition(new BazelAndroidInstrumentationTestRule());
          builder.addRuleDefinition(
              new AndroidDeviceScriptFixtureRule(BazelAndroidDeviceScriptFixture.class));
          builder.addRuleDefinition(
              new AndroidHostServiceFixtureRule(BazelAndroidHostServiceFixture.class));

          AndroidBootstrap bootstrap =
              new AndroidBootstrap(
                  new AndroidStarlarkCommon(),
                  ApkInfo.PROVIDER,
                  AndroidInstrumentationInfo.PROVIDER,
                  AndroidDeviceBrokerInfo.PROVIDER,
                  AndroidResourcesInfo.PROVIDER,
                  AndroidNativeLibsInfo.PROVIDER,
                  AndroidApplicationResourceInfo.PROVIDER,
                  AndroidSdkProvider.PROVIDER,
                  AndroidManifestInfo.PROVIDER,
                  AndroidAssetsInfo.PROVIDER,
                  AndroidLibraryAarInfo.PROVIDER,
                  AndroidProguardInfo.PROVIDER,
                  AndroidIdlProvider.PROVIDER,
                  AndroidIdeInfoProvider.PROVIDER,
                  AndroidPreDexJarProvider.PROVIDER,
                  AndroidCcLinkParamsProvider.PROVIDER,
                  DataBindingV2Provider.PROVIDER,
                  AndroidLibraryResourceClassJarProvider.PROVIDER,
                  AndroidFeatureFlagSetProvider.PROVIDER,
                  ProguardMappingProvider.PROVIDER,
                  AndroidBinaryDataInfo.PROVIDER);
          builder.addStarlarkBootstrap(bootstrap);

          try {
            builder.addWorkspaceFilePrefix(
                ResourceFileLoader.loadResource(BazelAndroidSemantics.class, "android.WORKSPACE"));
            builder.addWorkspaceFileSuffix(
                ResourceFileLoader.loadResource(
                    BazelAndroidSemantics.class, "android_remote_tools.WORKSPACE"));
            builder.addWorkspaceFileSuffix(
                ResourceFileLoader.loadResource(JavaRules.class, "coverage.WORKSPACE"));
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
          builder.addRuleDefinition(new PyRuntimeRule());

          builder.addStarlarkBootstrap(
              new PyBootstrap(
                  PyInfo.PROVIDER, PyRuntimeInfo.PROVIDER, PyStarlarkTransitions.INSTANCE));

          builder.addSymlinkDefinition(PySymlink.PY2);
          builder.addSymlinkDefinition(PySymlink.PY3);

          try {
            builder.addWorkspaceFileSuffix(
                ResourceFileLoader.loadResource(BazelPyBinaryRule.class, "python.WORKSPACE"));
          } catch (IOException e) {
            throw new IllegalStateException(e);
          }
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
          builder.addRuleDefinition(new NewLocalRepositoryRule());
          builder.addRuleDefinition(new AndroidSdkRepositoryRule());
          builder.addRuleDefinition(new AndroidNdkRepositoryRule());
          builder.addRuleDefinition(new LocalConfigPlatformRule());

          try {
            builder.addWorkspaceFileSuffix(
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
      // The default used to be "/bin:/usr/bin". However, on Mac the Python 3 interpreter, if it is
      // installed at all, tends to be under /usr/local/bin. The autodetecting Python toolchain
      // searches PATH for "python3", so if we don't include this directory then we can't run PY3
      // targets with this toolchain if strict action environment is on.
      //
      // Note that --action_env does not propagate to the host config, so it is not a viable
      // workaround when a genrule is itself built in the host config (e.g. nested genrules). See
      // #8536.
      return "/bin:/usr/bin:/usr/local/bin";
    }

    String newPath = "";
    // Attempt to compute the MSYS root (the real Windows path of "/") from `sh`.
    if (sh != null && sh.getParentDirectory() != null) {
      newPath = sh.getParentDirectory().getPathString();
      if (sh.getParentDirectory().endsWith(PathFragment.create("usr/bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().getParentDirectory().replaceName("bin").getPathString();
      } else if (sh.getParentDirectory().endsWith(PathFragment.create("bin"))) {
        newPath +=
            ";" + sh.getParentDirectory().replaceName("usr").getRelative("bin").getPathString();
      }
      newPath = newPath.replace('/', '\\');
    }
    // On Windows, the following dirs should always be available in PATH:
    //   C:\Windows
    //   C:\Windows\System32
    //   C:\Windows\System32\WindowsPowerShell\v1.0
    // They are similar to /bin:/usr/bin, which makes the basic tools on the platform available.
    String systemRoot = System.getenv("SYSTEMROOT");
    if (Strings.isNullOrEmpty(systemRoot)) {
      systemRoot = "C:\\Windows";
    }
    newPath += ";" + systemRoot;
    newPath += ";" + systemRoot + "\\System32";
    newPath += ";" + systemRoot + "\\System32\\WindowsPowerShell\\v1.0";
    if (path != null) {
      newPath += ";" + path;
    }
    return newPath;
  }
}
