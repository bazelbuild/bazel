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
package com.google.devtools.build.docgen;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.docgen.starlark.StarlarkBuiltinDoc;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.starlarkbuildapi.TopLevelBootstrap;
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
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import com.google.devtools.build.skydoc.fakebuildapi.FakeActionsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeBuildApiGlobals;
import com.google.devtools.build.skydoc.fakebuildapi.FakeConfigApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDefaultInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeOutputGroupInfo.FakeOutputGroupInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStarlarkAttrModuleApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStarlarkCommandLineApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStarlarkNativeModuleApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStarlarkRuleFunctionsApi;
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
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;

/**
 * A helper class that collects Starlark Api symbols including top level modules, native rules and
 * builtin types.
 */
public class SymbolFamilies {
  private final ImmutableList<RuleDocumentation> nativeRules;
  private final ImmutableMap<String, StarlarkBuiltinDoc> types;

  // Mappings between Starlark names and Starlark entities generated from the fakebuildapi.
  private final ImmutableMap<String, Object> globals;
  private final ImmutableMap<String, Object> bzlGlobals;

  public SymbolFamilies(
      String productName, String provider, List<String> inputDirs, String blackList)
      throws NoSuchMethodException, ClassPathException, InvocationTargetException,
          IllegalAccessException, BuildEncyclopediaDocException, ClassNotFoundException,
          IOException {
    this.nativeRules =
        ImmutableList.copyOf(collectNativeRules(productName, provider, inputDirs, blackList));
    this.globals = Starlark.UNIVERSE;
    this.bzlGlobals = ImmutableMap.copyOf(collectBzlGlobals());
    this.types = StarlarkDocumentationCollector.getAllModules();
  }

  /*
   * Returns a list of native rules.
   */
  public List<RuleDocumentation> getNativeRules() {
    return nativeRules;
  }

  /*
   * Returns a mapping between Starlark names and Starkark entities that are available both in BZL
   * and BUILD files.
   */
  public Map<String, Object> getGlobals() {
    return globals;
  }

  /*
   * Returns a mapping between Starlark names and Starkark entities that are available only in BZL
   * files.
   */
  public Map<String, Object> getBzlGlobals() {
    return bzlGlobals;
  }

  // Returns a mapping between type names and module/type documentation.
  public Map<String, StarlarkBuiltinDoc> getTypes() {
    return types;
  }

  /*
   * Collects a list of native rules that are available in BUILD files as top level functions
   * and in BZL files as methods of the native package.
   */
  private List<RuleDocumentation> collectNativeRules(
      String productName, String provider, List<String> inputDirs, String blackList)
      throws NoSuchMethodException, InvocationTargetException, IllegalAccessException,
          BuildEncyclopediaDocException, ClassNotFoundException, IOException {
    ProtoFileBuildEncyclopediaProcessor processor =
        new ProtoFileBuildEncyclopediaProcessor(productName, createRuleClassProvider(provider));
    processor.generateDocumentation(inputDirs, "", blackList);
    return processor.getNativeRules();
  }

  /*
   * Collects a mapping between names and Starlark entities that are available only in BZL files
   */
  private Map<String, Object> collectBzlGlobals() {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();
    TopLevelBootstrap topLevelBootstrap =
        new TopLevelBootstrap(
            new FakeBuildApiGlobals(),
            new FakeStarlarkAttrModuleApi(),
            new FakeStarlarkCommandLineApi(),
            new FakeStarlarkNativeModuleApi(),
            new FakeStarlarkRuleFunctionsApi(
                Lists.newArrayList(), Lists.newArrayList(), Lists.newArrayList()),
            new FakeStructProviderApi(),
            new FakeOutputGroupInfoProvider(),
            new FakeActionsInfoProvider(),
            new FakeDefaultInfoProvider());
    AndroidBootstrap androidBootstrap =
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
            new FakeAndroidBinaryDataInfo.FakeProvider());
    AppleBootstrap appleBootstrap = new AppleBootstrap(new FakeAppleCommon());
    ConfigBootstrap configBootstrap =
        new ConfigBootstrap(
            new FakeConfigStarlarkCommon(), new FakeConfigApi(), new FakeConfigGlobalLibrary());
    CcBootstrap ccBootstrap =
        new CcBootstrap(
            new FakeCcModule(),
            new FakeCcInfo.Provider(),
            new FakeCcToolchainConfigInfo.Provider(),
            new FakePyWrapCcHelper(),
            new FakeGoWrapCcHelper(),
            new FakePyWrapCcInfo.Provider(),
            new FakePyCcLinkParamsProvider.Provider());
    JavaBootstrap javaBootstrap =
        new JavaBootstrap(
            new FakeJavaCommon(),
            new FakeJavaInfoProvider(),
            new FakeJavaProtoCommon(),
            new FakeJavaCcLinkParamsProvider.Provider(),
            new FakeProguardSpecProvider.FakeProvider());
    PlatformBootstrap platformBootstrap = new PlatformBootstrap(new FakePlatformCommon());
    ProtoBootstrap protoBootstrap =
        new ProtoBootstrap(
            new FakeProtoInfoProvider(),
            new FakeProtoCommon(),
            new StarlarkAspectStub(),
            new ProviderStub());
    PyBootstrap pyBootstrap =
        new PyBootstrap(
            new FakePyInfoProvider(),
            new FakePyRuntimeInfoProvider(),
            new FakePyStarlarkTransitions());
    RepositoryBootstrap repositoryBootstrap =
        new RepositoryBootstrap(new FakeRepositoryModule(Lists.newArrayList()));
    TestingBootstrap testingBootstrap =
        new TestingBootstrap(
            new FakeTestingModule(),
            new FakeCoverageCommon(),
            new FakeInstrumentedFilesInfoProvider(),
            new FakeAnalysisFailureInfoProvider(),
            new FakeAnalysisTestResultInfoProvider());

    topLevelBootstrap.addBindingsToBuilder(envBuilder);
    androidBootstrap.addBindingsToBuilder(envBuilder);
    appleBootstrap.addBindingsToBuilder(envBuilder);
    ccBootstrap.addBindingsToBuilder(envBuilder);
    configBootstrap.addBindingsToBuilder(envBuilder);
    javaBootstrap.addBindingsToBuilder(envBuilder);
    platformBootstrap.addBindingsToBuilder(envBuilder);
    protoBootstrap.addBindingsToBuilder(envBuilder);
    pyBootstrap.addBindingsToBuilder(envBuilder);
    repositoryBootstrap.addBindingsToBuilder(envBuilder);
    testingBootstrap.addBindingsToBuilder(envBuilder);

    return envBuilder.build();
  }

  private ConfiguredRuleClassProvider createRuleClassProvider(String classProvider)
      throws NoSuchMethodException, InvocationTargetException, IllegalAccessException,
          ClassNotFoundException {
    Class<?> providerClass = Class.forName(classProvider);
    Method createMethod = providerClass.getMethod("create");
    return (ConfiguredRuleClassProvider) createMethod.invoke(null);
  }
}
