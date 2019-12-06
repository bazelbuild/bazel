// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.configuredtargets;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.TargetContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * Dummy ConfiguredTarget for environment groups. Contains no functionality, since environment
 * groups are not really first-class Targets.
 */
@AutoCodec
@Immutable // (and Starlark-hashable)
public final class EnvironmentGroupConfiguredTarget extends AbstractConfiguredTarget {
  @AutoCodec.Instantiator
  @AutoCodec.VisibleForSerialization
  EnvironmentGroupConfiguredTarget(Label label) {
    super(label, null);
  }

  public EnvironmentGroupConfiguredTarget(TargetContext targetContext) {
    this(targetContext.getLabel());
    Preconditions.checkState(targetContext.getConfiguration() == null, targetContext);
  }

  @Override
  protected Info rawGetSkylarkProvider(Provider.Key providerKey) {
    return null;
  }

  @Override
  protected Object rawGetSkylarkProvider(String providerKey) {
    return null;
  }
}
