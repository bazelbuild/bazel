// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;

/**
 * Provider of transitively available ZIPs of native libs that should be directly copied into the
 * APK.
 */
@SkylarkModule(name = "AndroidNativeLibsInfo", doc = "", documented = false)
@Immutable
public final class AndroidNativeLibsInfo extends NativeInfo {

  private static final String SKYLARK_NAME = "AndroidNativeLibsInfo";
  private static final FunctionSignature.WithValues<Object, SkylarkType> SIGNATURE =
      FunctionSignature.WithValues.create(
          FunctionSignature.of(
              /*numMandatoryPositionals=*/ 0,
              /*numOptionalPositionals=*/ 0,
              /*numMandatoryNamedOnly=*/ 1,
              /*starArg=*/ false,
              /*kwArg=*/ false,
              "native_libs"),
          /*defaultValues=*/ null,
          /*types=*/ ImmutableList.of(SkylarkType.of(SkylarkNestedSet.class)));
  public static final NativeProvider<AndroidNativeLibsInfo> PROVIDER =
      new NativeProvider<AndroidNativeLibsInfo>(
          AndroidNativeLibsInfo.class, SKYLARK_NAME, SIGNATURE) {
        @Override
        protected AndroidNativeLibsInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) {
          return new AndroidNativeLibsInfo(
              /*nativeLibs=*/ ((SkylarkNestedSet) args[0]).getSet(Artifact.class));
        }
      };

  private final NestedSet<Artifact> nativeLibs;

  public AndroidNativeLibsInfo(NestedSet<Artifact> nativeLibs) {
    super(PROVIDER);
    this.nativeLibs = nativeLibs;
  }

  @SkylarkCallable(
    name = "native_libs",
    doc = "Returns the native libraries produced by the rule.",
    structField = true
  )
  public NestedSet<Artifact> getNativeLibs() {
    return nativeLibs;
  }
}
