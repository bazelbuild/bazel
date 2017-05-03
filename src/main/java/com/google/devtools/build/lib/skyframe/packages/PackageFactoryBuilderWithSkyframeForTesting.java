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
package com.google.devtools.build.lib.skyframe.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;

/**
 * A {@link PackageFactory.BuilderForTesting} that also allows specification of some skyframe
 * details.
 */
public abstract class PackageFactoryBuilderWithSkyframeForTesting
    extends PackageFactory.BuilderForTesting {
  protected ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions = ImmutableMap.of();
  protected ImmutableList<PrecomputedValue.Injected> extraPrecomputedValues = ImmutableList.of();

  public PackageFactoryBuilderWithSkyframeForTesting setExtraSkyFunctions(
      ImmutableMap<SkyFunctionName, SkyFunction> extraSkyFunctions) {
    this.extraSkyFunctions = extraSkyFunctions;
    return this;
  }

  public PackageFactoryBuilderWithSkyframeForTesting setExtraPrecomputeValues(
      Iterable<PrecomputedValue.Injected> extraPrecomputedValues) {
    this.extraPrecomputedValues = ImmutableList.copyOf(extraPrecomputedValues);
    return this;
  }
}
