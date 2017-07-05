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

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * Dummy ConfiguredTarget for environment groups. Contains no functionality, since
 * environment groups are not really first-class Targets.
 */
public final class EnvironmentGroupConfiguredTarget extends AbstractConfiguredTarget {
  EnvironmentGroupConfiguredTarget(TargetContext targetContext, EnvironmentGroup envGroup) {
    super(targetContext);
    Preconditions.checkArgument(targetContext.getConfiguration() == null);
  }

  @Override
  public EnvironmentGroup getTarget() {
    return (EnvironmentGroup) super.getTarget();
  }

  @Override
  protected SkylarkClassObject rawGetSkylarkProvider(ClassObjectConstructor.Key providerKey) {
    return null;
  }

  @Override
  protected Object rawGetSkylarkProvider(String providerKey) {
    return null;
  }
}
