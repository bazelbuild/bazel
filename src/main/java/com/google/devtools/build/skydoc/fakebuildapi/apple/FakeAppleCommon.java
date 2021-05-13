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
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.SplitTransitionProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleCommonApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleDynamicFrameworkInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleExecutableBinaryApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ApplePlatformApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleStaticLibraryInfoApi.AppleStaticLibraryInfoProvider;
import com.google.devtools.build.lib.starlarkbuildapi.apple.AppleToolchainApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.DottedVersionApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.ObjcProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.apple.XcodeConfigInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeSplitTransitionProvider;
import com.google.devtools.build.skydoc.fakebuildapi.FakeStructApi;
import com.google.devtools.build.skydoc.fakebuildapi.apple.FakeAppleStaticLibraryInfo.FakeAppleStaticLibraryInfoProvider;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkThread;

/** Fake implementation of {@link AppleCommonApi}. */
public class FakeAppleCommon
    implements AppleCommonApi<
        FileApi,
        ConstraintValueInfoApi,
        StarlarkRuleContextApi<ConstraintValueInfoApi>,
        ObjcProviderApi<?>,
        XcodeConfigInfoApi<?, ?>,
        ApplePlatformApi> {

  @Override
  public AppleToolchainApi<?> getAppleToolchain() {
    return new FakeAppleToolchain();
  }

  @Override
  public StructApi getPlatformTypeStruct() {
    return new FakeStructApi(
        ImmutableMap.of(
            "ios", "ios",
            "macos", "macos",
            "tvos", "tvos",
            "watchos", "watchos"));
  }

  @Override
  public StructApi getPlatformStruct() {
    return new FakeStructApi(
        new ImmutableMap.Builder<String, Object>()
            .put("ios_device", "ios_device")
            .put("ios_simulator", "ios_simulator")
            .put("macos", "macos")
            .put("tvos_device", "tvos_device")
            .put("tvos_simulator", "tvos_simulator")
            .put("watchos_device", "watchos_device")
            .put("watchos_simulator", "watchos_simulator")
            .build());
  }

  @Override
  public ProviderApi getXcodeVersionPropertiesConstructor() {
    return new FakeProviderApi("XcodeProperties");
  }

  @Override
  public ProviderApi getXcodeVersionConfigConstructor() {
    return new FakeProviderApi("XcodeVersionConfig");
  }

  @Override
  public ProviderApi getObjcProviderConstructor() {
    return new FakeProviderApi("objc");
  }

  @Override
  public ProviderApi getAppleDynamicFrameworkConstructor() {
    return new FakeProviderApi("AppleDynamicFramework");
  }

  @Override
  public ProviderApi getAppleDylibBinaryConstructor() {
    return new FakeProviderApi("AppleDylibBinary");
  }

  @Override
  public ProviderApi getAppleExecutableBinaryConstructor() {
    return new FakeProviderApi("AppleExecutableBinary");
  }

  @Override
  public AppleStaticLibraryInfoProvider<?, ?> getAppleStaticLibraryProvider() {
    return new FakeAppleStaticLibraryInfoProvider();
  }

  @Override
  public ProviderApi getAppleDebugOutputsConstructor() {
    return new FakeProviderApi("AppleDebugOutputs");
  }

  @Override
  public ProviderApi getAppleLoadableBundleBinaryConstructor() {
    return new FakeProviderApi("AppleLoadableBundleBinary");
  }

  @Override
  public SplitTransitionProviderApi getMultiArchSplitProvider() {
    return new FakeSplitTransitionProvider();
  }

  @Override
  public StructApi linkMultiArchBinary(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      Sequence<?> extraLinkopts,
      Sequence<?> extraLinkInputs,
      StarlarkInt stamp,
      Boolean shouldLipo,
      StarlarkThread thread) {
    return new FakeStructApi();
  }

  @Override
  public DottedVersionApi<?> dottedVersion(String version) {
    return new FakeDottedVersion();
  }

  @Override
  public AppleDynamicFrameworkInfoApi<?> newDynamicFrameworkProvider(
      Object dylibBinary,
      ObjcProviderApi<?> depsObjcProvider,
      Object dynamicFrameworkDirs,
      Object dynamicFrameworkFiles) {
    return new FakeAppleDynamicFrameworkInfo();
  }

  @Override
  public AppleExecutableBinaryApi newExecutableBinaryProvider(
      Object executableBinary, ObjcProviderApi<?> depsObjcProvider) {
    return new FakeAppleExecutableBinaryInfo();
  }

  @Override
  public ObjcProviderApi<?> newObjcProvider(
      Boolean usesSwift, Dict<String, Object> kwargs, StarlarkThread thread) {
    return new FakeObjcProvider();
  }

  @Override
  public ImmutableMap<String, String> getTargetAppleEnvironment(
      XcodeConfigInfoApi<?, ?> xcodeConfig, ApplePlatformApi platform) {
    return ImmutableMap.of();
  }

  @Override
  public ImmutableMap<String, String> getAppleHostSystemEnv(XcodeConfigInfoApi<?, ?> xcodeConfig) {
    return ImmutableMap.of();
  }
}
