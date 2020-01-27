// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.configuredtargets.AbstractConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import javax.annotation.Nullable;

/** A configured target that is empty. */
@Immutable // (and Starlark-hashable)
public class EmptyConfiguredTarget extends AbstractConfiguredTarget {
  public EmptyConfiguredTarget(Label label, BuildConfigurationValue.Key configurationKey) {
    super(label, configurationKey, NestedSetBuilder.emptySet(Order.STABLE_ORDER));
  }

  @Nullable
  @Override
  protected Info rawGetSkylarkProvider(Provider.Key providerKey) {
    return null;
  }

  @Override
  protected Object rawGetSkylarkProvider(String providerKey) {
    return null;
  }

  @Override
  public String toString() {
    return "Empty" + super.toString();
  }
}
