// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.config.ConfigBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.platform.PlatformBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.proto.ProtoBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.python.PyBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.stubs.ProviderStub;
import com.google.devtools.build.lib.starlarkbuildapi.stubs.StarlarkAspectStub;
import com.google.devtools.build.lib.starlarkbuildapi.test.TestingBootstrap;
import com.google.devtools.build.skydoc.fakebuildapi.FakeOutputGroupInfo.FakeOutputGroupInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi.FakeStructProviderApi;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidApplicationResourceInfo.FakeAndroidApplicationResourceInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidAssetsInfo;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidBinaryDataInfo;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidCcLinkParamsProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidDeviceBrokerInfo.FakeAndroidDeviceBrokerInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidFeatureFlagSetProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidIdeInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidIdlProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidInstrumentationInfo.FakeAndroidInstrumentationInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidLibraryAarInfo;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidLibraryResourceClassJarProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidManifestInfo;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidNativeLibsInfo.FakeAndroidNativeLibsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidPreDexJarProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidProguardInfo;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidResourcesInfo.FakeAndroidResourcesInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidSdkProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidStarlarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeApkInfo.FakeApkInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeDataBindingV2Provider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeProguardMappingProvider;
import com.google.devtools.build.skydoc.fakebuildapi.apple.FakeAppleCommon;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigGlobalLibrary;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigStarlarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcInfo;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcModule;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcToolchainConfigInfo;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeDebugPackageInfo;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeGoWrapCcHelper;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakePyCcLinkParamsProvider;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakePyWrapCcHelper;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakePyWrapCcInfo;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaCcLinkParamsProvider;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaCommon;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaInfo.FakeJavaInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaProtoCommon;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeProguardSpecProvider;
import com.google.devtools.build.skydoc.fakebuildapi.platform.FakePlatformCommon;
import com.google.devtools.build.skydoc.fakebuildapi.proto.FakeProtoCommon;
import com.google.devtools.build.skydoc.fakebuildapi.proto.FakeProtoInfo.FakeProtoInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyInfo.FakePyInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyRuntimeInfo.FakePyRuntimeInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyStarlarkTransitions;
import com.google.devtools.build.skydoc.fakebuildapi.repository.FakeRepositoryModule;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeAnalysisFailureInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeAnalysisTestResultInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeCoverageCommon;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeInstrumentedFilesInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeTestingModule;
import com.google.devtools.build.skydoc.rendering.AspectInfoWrapper;
import com.google.devtools.build.skydoc.rendering.ProviderInfoWrapper;
import com.google.devtools.build.skydoc.rendering.RuleInfoWrapper;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import java.util.List;
import net.starlark.java.eval.Starlark;

/** Defines the fake .bzl environment. */
public final class FakeApi {

  private FakeApi() {} // uninstantiable

  /**
   * Adds the predeclared environment containing the fake build API.
   *
   * @param rules the list of {@link RuleInfo} objects, to which 'rule' and 'repository_rule'
   *     invocation information will be added
   * @param providers the list of {@link ProviderInfo} objects, to which 'provider' invocation
   *     information will be added
   * @param aspects the list of {@link AspectInfo} objects, to which 'aspect' invocation information
   *     will be added
   */
  public static void addPredeclared(
      ImmutableMap.Builder<String, Object> env,
      /* out parameters: */
      List<RuleInfoWrapper> rules,
      List<ProviderInfoWrapper> providers,
      List<AspectInfoWrapper> aspects) {

    Starlark.addMethods(env, new FakeBuildApiGlobals()); // e.g. configuration_field func
    Starlark.addMethods(
        env, new FakeStarlarkRuleFunctionsApi(rules, providers, aspects)); // e.g. rule func
    env.put("attr", new FakeStarlarkAttrModuleApi());
    env.put("cmd_helper", new FakeStarlarkCommandLineApi());
    env.put("native", new FakeStarlarkNativeModuleApi());
    env.put("struct", new FakeStructProviderApi());
    env.put("OutputGroupInfo", new FakeOutputGroupInfoProvider());
    env.put("Actions", new FakeActionsInfoProvider());
    env.put("DefaultInfo", new FakeDefaultInfoProvider());

    new AndroidBootstrap(
            new FakeAndroidStarlarkCommon(),
            new FakeApkInfoProvider(),
            new FakeAndroidInstrumentationInfoProvider(),
            new FakeAndroidDeviceBrokerInfoProvider(),
            new FakeAndroidResourcesInfoProvider(),
            new FakeAndroidNativeLibsInfoProvider(),
            new FakeAndroidApplicationResourceInfoProvider(),
            new FakeAndroidSdkProvider.FakeProvider(),
            new FakeAndroidManifestInfo.FakeProvider(),
            new FakeAndroidAssetsInfo.FakeProvider(),
            new FakeAndroidLibraryAarInfo.FakeProvider(),
            new FakeAndroidProguardInfo.FakeProvider(),
            new FakeAndroidIdlProvider.FakeProvider(),
            new FakeAndroidIdeInfoProvider.FakeProvider(),
            new FakeAndroidPreDexJarProvider.FakeProvider(),
            new FakeAndroidCcLinkParamsProvider.FakeProvider(),
            new FakeDataBindingV2Provider.FakeProvider(),
            new FakeAndroidLibraryResourceClassJarProvider.FakeProvider(),
            new FakeAndroidFeatureFlagSetProvider.FakeProvider(),
            new FakeProguardMappingProvider.FakeProvider(),
            new FakeAndroidBinaryDataInfo.FakeProvider())
        .addBindingsToBuilder(env);
    new AppleBootstrap(new FakeAppleCommon()).addBindingsToBuilder(env);
    new ConfigBootstrap(
            new FakeConfigStarlarkCommon(), //
            new FakeConfigApi(),
            new FakeConfigGlobalLibrary())
        .addBindingsToBuilder(env);
    new CcBootstrap(
            new FakeCcModule(),
            new FakeCcInfo.Provider(),
            new FakeDebugPackageInfo.Provider(),
            new FakeCcToolchainConfigInfo.Provider(),
            new FakePyWrapCcHelper(),
            new FakeGoWrapCcHelper(),
            new FakePyWrapCcInfo.Provider(),
            new FakePyCcLinkParamsProvider.Provider())
        .addBindingsToBuilder(env);
    new JavaBootstrap(
            new FakeJavaCommon(),
            new FakeJavaInfoProvider(),
            new FakeJavaProtoCommon(),
            new FakeJavaCcLinkParamsProvider.Provider(),
            new FakeProguardSpecProvider.FakeProvider())
        .addBindingsToBuilder(env);
    new PlatformBootstrap(new FakePlatformCommon()).addBindingsToBuilder(env);
    new ProtoBootstrap(
            new FakeProtoInfoProvider(),
            new FakeProtoCommon(),
            new StarlarkAspectStub(),
            new ProviderStub())
        .addBindingsToBuilder(env);
    new PyBootstrap(
            new FakePyInfoProvider(),
            new FakePyRuntimeInfoProvider(),
            new FakePyStarlarkTransitions())
        .addBindingsToBuilder(env);
    new RepositoryBootstrap(new FakeRepositoryModule(rules)).addBindingsToBuilder(env);
    new TestingBootstrap(
            new FakeTestingModule(),
            new FakeCoverageCommon(),
            new FakeInstrumentedFilesInfoProvider(),
            new FakeAnalysisFailureInfoProvider(),
            new FakeAnalysisTestResultInfoProvider())
        .addBindingsToBuilder(env);
  }
}
