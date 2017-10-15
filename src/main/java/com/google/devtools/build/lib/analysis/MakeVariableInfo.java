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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/** Provides access to make variables from the current fragments. */
@SkylarkModule(name = "MakeVariables", doc = "Make variables exposed by the current target.")
@Immutable
public final class MakeVariableInfo extends NativeInfo {
  public static final String SKYLARK_NAME = "MakeVariableInfo";

  public static final NativeProvider<MakeVariableInfo> PROVIDER =
      new NativeProvider<MakeVariableInfo>(MakeVariableInfo.class, SKYLARK_NAME) {};

  private final ImmutableMap<String, String> makeVariables;

  public MakeVariableInfo(ImmutableMap<String, String> makeVariables) {
    super(PROVIDER);
    this.makeVariables = makeVariables;
  }

  @SkylarkCallable(
    name = "make_variables",
    doc = "Returns the make variables defined by this target.",
    structField = true
  )
  public ImmutableMap<String, String> getMakeVariables() {
    return makeVariables;
  }

  @Override
  public boolean equals(Object other) {
    return other == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
