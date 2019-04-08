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
import com.google.devtools.build.docgen.skylark.SkylarkModuleDoc;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.skylarkbuildapi.TopLevelBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.android.AndroidBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.config.ConfigBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.platform.PlatformBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.python.PyBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.repository.RepositoryBootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.test.TestingBootstrap;
import com.google.devtools.build.lib.syntax.MethodLibrary;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.util.Classpath;
import com.google.devtools.build.lib.util.Classpath.ClassPathException;
import com.google.devtools.build.skydoc.fakebuildapi.FakeActionsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeBuildApiGlobals;
import com.google.devtools.build.skydoc.fakebuildapi.FakeConfigApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeDefaultInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeOutputGroupInfo.FakeOutputGroupInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkAttrApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkCommandLineApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkNativeModuleApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkRuleFunctionsApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi.FakeStructProviderApi;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidDeviceBrokerInfo.FakeAndroidDeviceBrokerInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidInstrumentationInfo.FakeAndroidInstrumentationInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidNativeLibsInfo.FakeAndroidNativeLibsInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidResourcesInfo.FakeAndroidResourcesInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeAndroidSkylarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.android.FakeApkInfo.FakeApkInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.apple.FakeAppleCommon;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigGlobalLibrary;
import com.google.devtools.build.skydoc.fakebuildapi.config.FakeConfigSkylarkCommon;
import com.google.devtools.build.skydoc.fakebuildapi.cpp.FakeCcModule;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaCcLinkParamsProvider;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaCommon;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaInfo.FakeJavaInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.java.FakeJavaProtoCommon;
import com.google.devtools.build.skydoc.fakebuildapi.platform.FakePlatformCommon;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyInfo.FakePyInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.python.FakePyRuntimeInfo.FakePyRuntimeInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.repository.FakeRepositoryModule;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeAnalysisFailureInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeAnalysisTestResultInfoProvider;
import com.google.devtools.build.skydoc.fakebuildapi.test.FakeCoverageCommon;
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
  // Common prefix of packages that may contain Skylark modules.
  private static final String MODULES_PACKAGE_PREFIX = "com/google/devtools/build";

  private final ImmutableList<RuleDocumentation> nativeRules;
  private final ImmutableMap<String, SkylarkModuleDoc> types;

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
    this.globals = ImmutableMap.copyOf(collectGlobals());
    this.bzlGlobals = ImmutableMap.copyOf(collectBzlGlobals());
    this.types = ImmutableMap.copyOf(collectTypes());
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
  public Map<String, SkylarkModuleDoc> getTypes() {
    return types;
  }

  /*
   * Collects a mapping between type names and module/type documentation that are available both
   * in BZL and BUILD files.
   */
  private Map<String, SkylarkModuleDoc> collectTypes() throws ClassPathException {
    return SkylarkDocumentationCollector.collectModules(
        Classpath.findClasses(MODULES_PACKAGE_PREFIX));
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
   * Collects a mapping between names and Starlark entities that are available both in BZL and
   * BUILD files.
   */
  private Map<String, Object> collectGlobals() {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();
    MethodLibrary.addBindingsToBuilder(envBuilder);
    Runtime.addConstantsToBuilder(envBuilder);
    return envBuilder.build();
  }

  /*
   * Collects a mapping between names and Starlark entities that are available only in BZL files
   */
  private Map<String, Object> collectBzlGlobals() {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();
    TopLevelBootstrap topLevelBootstrap =
        new TopLevelBootstrap(
            new FakeBuildApiGlobals(),
            new FakeSkylarkAttrApi(),
            new FakeSkylarkCommandLineApi(),
            new FakeSkylarkNativeModuleApi(),
            new FakeSkylarkRuleFunctionsApi(Lists.newArrayList(), Lists.newArrayList()),
            new FakeStructProviderApi(),
            new FakeOutputGroupInfoProvider(),
            new FakeActionsInfoProvider(),
            new FakeDefaultInfoProvider());
    AndroidBootstrap androidBootstrap =
        new AndroidBootstrap(
            new FakeAndroidSkylarkCommon(),
            new FakeApkInfoProvider(),
            new FakeAndroidInstrumentationInfoProvider(),
            new FakeAndroidDeviceBrokerInfoProvider(),
            new FakeAndroidResourcesInfoProvider(),
            new FakeAndroidNativeLibsInfoProvider());
    AppleBootstrap appleBootstrap = new AppleBootstrap(new FakeAppleCommon());
    ConfigBootstrap configBootstrap =
        new ConfigBootstrap(new FakeConfigSkylarkCommon(), new FakeConfigApi(),
            new FakeConfigGlobalLibrary());
    CcBootstrap ccBootstrap = new CcBootstrap(new FakeCcModule());
    JavaBootstrap javaBootstrap =
        new JavaBootstrap(
            new FakeJavaCommon(),
            new FakeJavaInfoProvider(),
            new FakeJavaProtoCommon(),
            new FakeJavaCcLinkParamsProvider.Provider());
    PlatformBootstrap platformBootstrap = new PlatformBootstrap(new FakePlatformCommon());
    PyBootstrap pyBootstrap =
        new PyBootstrap(new FakePyInfoProvider(), new FakePyRuntimeInfoProvider());
    RepositoryBootstrap repositoryBootstrap =
        new RepositoryBootstrap(new FakeRepositoryModule(Lists.newArrayList()));
    TestingBootstrap testingBootstrap =
        new TestingBootstrap(
            new FakeTestingModule(),
            new FakeCoverageCommon(),
            new FakeAnalysisFailureInfoProvider(),
            new FakeAnalysisTestResultInfoProvider());

    topLevelBootstrap.addBindingsToBuilder(envBuilder);
    androidBootstrap.addBindingsToBuilder(envBuilder);
    appleBootstrap.addBindingsToBuilder(envBuilder);
    ccBootstrap.addBindingsToBuilder(envBuilder);
    configBootstrap.addBindingsToBuilder(envBuilder);
    javaBootstrap.addBindingsToBuilder(envBuilder);
    platformBootstrap.addBindingsToBuilder(envBuilder);
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
