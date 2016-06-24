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

package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Target;

import java.util.Map;

/**
 * Standard {@link SupportedEnvironmentsProvider} implementation.
 */
public class SupportedEnvironments implements SupportedEnvironmentsProvider {
  private final EnvironmentCollection staticEnvironments;
  private final EnvironmentCollection refinedEnvironments;
  private final ImmutableMap<Label, Target> removedEnvironmentCulprits;

  public SupportedEnvironments(EnvironmentCollection staticEnvironments,
      EnvironmentCollection refinedEnvironments, Map<Label, Target> removedEnvironmentCulprits) {
    this.staticEnvironments = staticEnvironments;
    this.refinedEnvironments = refinedEnvironments;
    this.removedEnvironmentCulprits = ImmutableMap.copyOf(removedEnvironmentCulprits);
  }

  @Override
  public EnvironmentCollection getStaticEnvironments() {
    return staticEnvironments;
  }

  @Override
  public EnvironmentCollection getRefinedEnvironments() {
    return refinedEnvironments;
  }

  @Override
  public Target getRemovedEnvironmentCulprit(Label environment) {
    return removedEnvironmentCulprits.get(environment);
  }
}
