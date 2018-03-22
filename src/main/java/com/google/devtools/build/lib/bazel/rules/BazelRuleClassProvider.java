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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.Builder;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.RuleSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.bazel.rules.android.AndroidNdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.AndroidSdkRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAarImportRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidBinaryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidLibraryRule;
import com.google.devtools.build.lib.bazel.rules.android.BazelAndroidLocalTestRule;
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
import com.google.devtools.build.lib.bazel.rules.workspace.GitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpArchiveRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpFileRule;
import com.google.devtools.build.lib.bazel.rules.workspace.HttpJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenJarRule;
import com.google.devtools.build.lib.bazel.rules.workspace.MavenServerRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewGitRepositoryRule;
import com.google.devtools.build.lib.bazel.rules.workspace.NewHttpArchiveRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.rules.android.AarImportBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidDeviceBrokerInfo;
import com.google.devtools.build.lib.rules.android.AndroidDeviceRule;
import com.google.devtools.build.lib.rules.android.AndroidDeviceScriptFixtureRule;
import com.google.devtools.build.lib.rules.android.AndroidHostServiceFixtureRule;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationInfo;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationTestRule;
import com.google.devtools.build.lib.rules.android.AndroidLibraryBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidLocalTestConfiguration;
import com.google.devtools.build.lib.rules.android.AndroidManifestInfo;
import com.google.devtools.build.lib.rules.android.AndroidNativeLibsInfo;
import com.google.devtools.build.lib.rules.android.AndroidNeverlinkAspect;
import com.google.devtools.build.lib.rules.android.AndroidResourcesInfo;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidToolsDefaultsJarRule;
import com.google.devtools.build.lib.rules.android.AndroidSkylarkCommon;
import com.google.devtools.build.lib.rules.android.ApkInfo;
import com.google.devtools.build.lib.rules.android.DexArchiveAspect;
import com.google.devtools.build.lib.rules.config.ConfigRules;
import com.google.devtools.build.lib.rules.core.CoreRules;
import com.google.devtools.build.lib.rules.cpp.proto.CcProtoAspect;
import com.google.devtools.build.lib.rules.cpp.proto.CcProtoLibraryRule;
import com.google.devtools.build.lib.rules.cpp.transitions.LipoDataTransitionRuleSet;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.platform.PlatformRules;
import com.google.devtools.build.lib.rules.proto.BazelProtoLibraryRule;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.proto.ProtoLangToolchainRule;
import com.google.devtools.build.lib.rules.python.PythonConfigurationLoader;
import com.google.devtools.build.lib.rules.python.PythonOptions;
import com.google.devtools.build.lib.rules.repository.CoreWorkspaceRules;
import com.google.devtools.build.lib.rules.repository.NewLocalRepositoryRule;
import com.google.devtools.build.lib.rules.test.TestingSupportRules;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import java.io.IOException;

/** A rule class provider implementing the rules Bazel knows. */
public class BazelRuleClassProvider {
  public static final String TOOLS_REPOSITORY = "@bazel_tools";

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
        public void init(Builder builder) {
          builder
              .setPrelude("//tools/build_rules:prelude_bazel")
              .setNativeLauncherLabel("//tools/launcher:launcher")
              .setRunfilesPrefix(Label.DEFAULT_REPOSITORY_DIRECTORY)
              .setPrerequisiteValidator(new BazelPrerequisiteValidator());

          builder.setUniversalConfigurationFragment(BazelConfiguration.class);
          builder.addConfigurationFragment(new BazelConfiguration.Loader());
          builder.addConfigurationOptions(BazelConfiguration.Options.class);
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
        public void init(Builder builder) {
          builder.addConfigurationOptions(ProtoConfiguration.Options.class);
          builder.addConfigurationFragment(new ProtoConfiguration.Loader());
          builder.addRuleDefinition(new BazelProtoLibraryRule());
          builder.addRuleDefinition(new ProtoLangToolchainRule());
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE);
        }
      };

  public static final RuleSet CPP_PROTO_RULES =
      new RuleSet() {
        @Override
        public void init(Builder builder) {
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
        public void init(Builder builder) {
          LabelLateBoundDefault<?> hostJdkAttribute = JavaSemantics.hostJdkAttribute(builder);
          BazelJavaProtoAspect bazelJavaProtoAspect = new BazelJavaProtoAspect(hostJdkAttribute);
          BazelJavaLiteProtoAspect bazelJavaLiteProtoAspect =
              new BazelJavaLiteProtoAspect(hostJdkAttribute);
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
        public void init(Builder builder) {
          String toolsRepository = checkNotNull(builder.getToolsRepository());

          builder.addConfig(AndroidConfiguration.Options.class, new AndroidConfiguration.Loader());
          builder.addConfig(
              AndroidLocalTestConfiguration.Options.class,
              new AndroidLocalTestConfiguration.Loader());

          AndroidNeverlinkAspect androidNeverlinkAspect = new AndroidNeverlinkAspect();
          DexArchiveAspect dexArchiveAspect = new DexArchiveAspect(toolsRepository);
          builder.addNativeAspectClass(androidNeverlinkAspect);
          builder.addNativeAspectClass(dexArchiveAspect);

          builder.addRuleDefinition(new AndroidRuleClasses.AndroidSdkRule());
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
          builder.addRuleDefinition(new AndroidInstrumentationTestRule());
          builder.addRuleDefinition(new AndroidDeviceScriptFixtureRule());
          builder.addRuleDefinition(new AndroidHostServiceFixtureRule());

          builder.addSkylarkAccessibleTopLevels("android_common", new AndroidSkylarkCommon());
          builder.addSkylarkAccessibleTopLevels(ApkInfo.PROVIDER.getName(), ApkInfo.PROVIDER);
          builder.addSkylarkAccessibleTopLevels(
              AndroidInstrumentationInfo.PROVIDER.getName(), AndroidInstrumentationInfo.PROVIDER);
          builder.addSkylarkAccessibleTopLevels(
              AndroidDeviceBrokerInfo.PROVIDER.getName(), AndroidDeviceBrokerInfo.PROVIDER);
          builder.addSkylarkAccessibleTopLevels(
              AndroidResourcesInfo.PROVIDER.getName(), AndroidResourcesInfo.PROVIDER);
          builder.addSkylarkAccessibleTopLevels(
              AndroidNativeLibsInfo.PROVIDER.getName(), AndroidNativeLibsInfo.PROVIDER);
          builder.addSkylarkAccessibleTopLevels(
              AndroidManifestInfo.PROVIDER.getName(), AndroidManifestInfo.PROVIDER);

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
        public void init(Builder builder) {
          builder.addConfig(PythonOptions.class, new PythonConfigurationLoader());
          builder.addConfig(
              BazelPythonConfiguration.Options.class, new BazelPythonConfiguration.Loader());

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
        public void init(Builder builder) {
          // TODO(ulfjack): Split this up by conceptual units.
          builder.addRuleDefinition(new GitRepositoryRule());
          builder.addRuleDefinition(new HttpArchiveRule());
          builder.addRuleDefinition(new HttpJarRule());
          builder.addRuleDefinition(new HttpFileRule());
          builder.addRuleDefinition(new MavenJarRule());
          builder.addRuleDefinition(new MavenServerRule());
          builder.addRuleDefinition(new NewHttpArchiveRule());
          builder.addRuleDefinition(new NewGitRepositoryRule());
          builder.addRuleDefinition(new NewLocalRepositoryRule());
          builder.addRuleDefinition(new AndroidSdkRepositoryRule());
          builder.addRuleDefinition(new AndroidNdkRepositoryRule());
        }

        @Override
        public ImmutableList<RuleSet> requires() {
          return ImmutableList.of(CoreRules.INSTANCE, CoreWorkspaceRules.INSTANCE);
        }
      };

  private static final ImmutableSet<RuleSet> RULE_SETS =
      ImmutableSet.of(
          // Rules defined before LipoDataTransitionRuleSet will fail when trying to declare a data
          // transition.
          // TODO(b/73071922): remove this when LIPO support is phased out
          LipoDataTransitionRuleSet.INSTANCE,
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
}
