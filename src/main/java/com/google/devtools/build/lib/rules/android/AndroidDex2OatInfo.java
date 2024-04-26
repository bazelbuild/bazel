// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.starlarkbuildapi.android.AndroidDex2OatInfoApi;
import javax.annotation.Nullable;

/**
 * Supplies the pregenerate_oat_files_for_tests attribute of type boolean provided by android_device
 * rule.
 */
@Immutable
public final class AndroidDex2OatInfo extends NativeInfo
    implements AndroidDex2OatInfoApi<Artifact, FilesToRunProvider> {

  public static final BuiltinProvider<AndroidDex2OatInfo> PROVIDER = new Provider();

  /** Provider for {@link AndroidDex2OatInfo} objects. */
  private static class Provider extends BuiltinProvider<AndroidDex2OatInfo>
      implements AndroidDex2OatInfoApi.Provider<Artifact, FilesToRunProvider> {
    public Provider() {
      super(NAME, AndroidDex2OatInfo.class);
    }

    @Override
    public AndroidDex2OatInfo androidDex2OatInfo(
        Boolean enabled,
        Boolean executeDex2OatOnHost,
        Object sandboxForPregeneratingOatFilesForTests,
        Object framework,
        Object dalvikCache,
        Object deviceProps) {
      return new AndroidDex2OatInfo(
          enabled,
          executeDex2OatOnHost,
          sandboxForPregeneratingOatFilesForTests instanceof FilesToRunProvider filesToRunProvider
              ? filesToRunProvider
              : null,
          framework instanceof Artifact ? (Artifact) framework : null,
          dalvikCache instanceof Artifact ? (Artifact) dalvikCache : null,
          deviceProps instanceof Artifact ? (Artifact) deviceProps : null);
    }
  }

  private final boolean dex2OatEnabled;
  private final boolean executeDex2OatOnExecPlatform;
  private final FilesToRunProvider sandboxForPregeneratingOatFilesForTests;
  private final Artifact framework;
  private final Artifact dalvikCache;
  private final Artifact deviceProps;

  public AndroidDex2OatInfo(
      boolean dex2OatEnabled,
      boolean executeDex2OatOnHost,
      FilesToRunProvider sandboxForPregeneratingOatFilesForTests,
      Artifact framework,
      Artifact dalvikCache,
      Artifact deviceProps) {
    this.dex2OatEnabled = dex2OatEnabled;
    this.executeDex2OatOnExecPlatform = executeDex2OatOnHost;
    this.sandboxForPregeneratingOatFilesForTests = sandboxForPregeneratingOatFilesForTests;
    this.framework = framework;
    this.dalvikCache = dalvikCache;
    this.deviceProps = deviceProps;
  }

  @Override
  public BuiltinProvider<AndroidDex2OatInfo> getProvider() {
    return PROVIDER;
  }

  /** Returns if the device should run cloud dex2oat. */
  public boolean isDex2OatEnabled() {
    return dex2OatEnabled;
  }

  /** Returns whether dex2oat should be executed on the exec platform. */
  public boolean executeDex2OatOnExecPlatform() {
    return executeDex2OatOnExecPlatform;
  }

  /** Returns the virtual device executable to run dex2oat on the exec platform */
  @Nullable
  public FilesToRunProvider getSandboxForPregeneratingOatFilesForTests() {
    return sandboxForPregeneratingOatFilesForTests;
  }

  @Nullable
  public Artifact getFramework() {
    return framework;
  }

  @Nullable
  public Artifact getDalvikCache() {
    return dalvikCache;
  }

  @Nullable
  public Artifact getDeviceProps() {
    return deviceProps;
  }
}
