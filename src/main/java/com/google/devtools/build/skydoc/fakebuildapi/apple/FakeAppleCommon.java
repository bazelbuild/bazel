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

package com.google.devtools.build.skydoc.fakebuildapi.apple;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkAspectApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.SplitTransitionProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleCommonApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleDynamicFrameworkInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ApplePlatformApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleStaticLibraryInfoApi.AppleStaticLibraryInfoProvider;
import com.google.devtools.build.lib.skylarkbuildapi.apple.AppleToolchainApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.DottedVersionApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.ObjcProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.apple.XcodeConfigProviderApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSkylarkAspect;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSplitTransitionProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi;
import com.google.devtools.build.skydoc.fakebuildapi.apple.FakeAppleStaticLibraryInfo.FakeAppleStaticLibraryInfoProvider;

/**
 * Fake implementation of {@link AppleCommonApi}.
 */
public class FakeAppleCommon implements AppleCommonApi<
    FileApi,
    ObjcProviderApi<?>,
    XcodeConfigProviderApi<?, ?>,
    ApplePlatformApi> {

  @Override
  public AppleToolchainApi<?> getAppleToolchain() {
    return new FakeAppleToolchain();
  }

  @Override
  public StructApi getPlatformTypeStruct() {
    return new FakeStructApi();
  }

  @Override
  public StructApi getPlatformStruct() {
    return new FakeStructApi();
  }

  @Override
  public ProviderApi getXcodeVersionPropertiesConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public ProviderApi getXcodeVersionConfigConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public ProviderApi getObjcProviderConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public ProviderApi getAppleDynamicFrameworkConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public ProviderApi getAppleDylibBinaryConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public ProviderApi getAppleExecutableBinaryConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public AppleStaticLibraryInfoProvider<?, ?> getAppleStaticLibraryProvider() {
    return new FakeAppleStaticLibraryInfoProvider();
  }

  @Override
  public ProviderApi getAppleDebugOutputsConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public ProviderApi getAppleLoadableBundleBinaryConstructor() {
    return new FakeProviderApi();
  }

  @Override
  public SplitTransitionProviderApi getMultiArchSplitProvider() {
    return new FakeSplitTransitionProvider();
  }

  @Override
  public StructApi linkMultiArchBinary(
      SkylarkRuleContextApi skylarkRuleContext,
      SkylarkList<String> extraLinkopts,
      SkylarkList<? extends FileApi> extraLinkInputs,
      Environment environment) {
    return new FakeStructApi();
  }

  @Override
  public DottedVersionApi<?> dottedVersion(String version) {
    return new FakeDottedVersion();
  }

  @Override
  public SkylarkAspectApi getObjcProtoAspect() {
    return new FakeSkylarkAspect();
  }

  @Override
  public AppleDynamicFrameworkInfoApi<?, ?> newDynamicFrameworkProvider(Object dylibBinary,
      ObjcProviderApi<?> depsObjcProvider, Object dynamicFrameworkDirs,
      Object dynamicFrameworkFiles) {
    return new FakeAppleDynamicFrameworkInfo();
  }

  @Override
  public ObjcProviderApi<?> newObjcProvider(Boolean usesSwift, SkylarkDict<?, ?> kwargs,
      Environment environment) {
    return new FakeObjcProvider();
  }

  @Override
  public ImmutableMap<String, String> getTargetAppleEnvironment(
      XcodeConfigProviderApi<?, ?> xcodeConfig,
      ApplePlatformApi platform) {
    return ImmutableMap.of();
  }

  @Override
  public ImmutableMap<String, String> getAppleHostSystemEnv(
      XcodeConfigProviderApi<?, ?> xcodeConfig) {
    return ImmutableMap.of();
  }
}
