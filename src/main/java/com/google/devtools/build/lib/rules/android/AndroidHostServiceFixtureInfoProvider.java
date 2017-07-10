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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;

/**
 * Information about an {@code android_host_service_fixture} to run as part of an {@code
 * android_instrumentation_test}.
 */
@Immutable
public class AndroidHostServiceFixtureInfoProvider extends SkylarkClassObject
    implements TransitiveInfoProvider {

  private static final String SKYLARK_NAME = "HostServiceFixtureInfo";
  static final NativeClassObjectConstructor<AndroidHostServiceFixtureInfoProvider>
      ANDROID_HOST_SERVICE_FIXTURE_INFO =
          new NativeClassObjectConstructor<AndroidHostServiceFixtureInfoProvider>(
              AndroidHostServiceFixtureInfoProvider.class, SKYLARK_NAME) {};

  private final Artifact executable;
  private final ImmutableList<String> serviceNames;
  private final NestedSet<Artifact> supportApks;
  private final boolean providesTestArgs;
  private final boolean daemon;

  AndroidHostServiceFixtureInfoProvider(
      Artifact executable,
      ImmutableList<String> serviceNames,
      NestedSet<Artifact> supportApks,
      boolean providesTestArgs,
      boolean isDaemon) {
    super(ANDROID_HOST_SERVICE_FIXTURE_INFO, ImmutableMap.<String, Object>of());
    this.executable = executable;
    this.serviceNames = serviceNames;
    this.supportApks = supportApks;
    this.providesTestArgs = providesTestArgs;
    this.daemon = isDaemon;
  }

  public Artifact getExecutable() {
    return executable;
  }

  public ImmutableList<String> getServiceNames() {
    return serviceNames;
  }

  public NestedSet<Artifact> getSupportApks() {
    return supportApks;
  }

  public boolean getProvidesTestArgs() {
    return providesTestArgs;
  }

  public boolean getDaemon() {
    return daemon;
  }
}
