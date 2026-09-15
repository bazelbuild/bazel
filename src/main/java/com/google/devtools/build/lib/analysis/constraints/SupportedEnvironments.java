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
import java.util.Map;

/** Standard {@link SupportedEnvironmentsProvider} implementation. */
public final class SupportedEnvironments implements SupportedEnvironmentsProvider {

  static final SupportedEnvironments EMPTY =
      new SupportedEnvironments(
          EnvironmentCollection.EMPTY, EnvironmentCollection.EMPTY, ImmutableMap.of());

  public static SupportedEnvironments create(
      EnvironmentCollection staticEnvironments,
      EnvironmentCollection refinedEnvironments,
      Map<Label, RemovedEnvironmentCulprit> removedEnvironmentCulprits) {
    if (staticEnvironments.isEmpty()
        && refinedEnvironments.isEmpty()
        && removedEnvironmentCulprits.isEmpty()) {
      return EMPTY;
    }
    return new SupportedEnvironments(
        staticEnvironments, refinedEnvironments, ImmutableMap.copyOf(removedEnvironmentCulprits));
  }

  private final EnvironmentCollection staticEnvironments;
  private final EnvironmentCollection refinedEnvironments;
  private final ImmutableMap<Label, RemovedEnvironmentCulprit> removedEnvironmentCulprits;

  private SupportedEnvironments(
      EnvironmentCollection staticEnvironments,
      EnvironmentCollection refinedEnvironments,
      ImmutableMap<Label, RemovedEnvironmentCulprit> removedEnvironmentCulprits) {
    this.staticEnvironments = staticEnvironments;
    this.refinedEnvironments = refinedEnvironments;
    this.removedEnvironmentCulprits = removedEnvironmentCulprits;
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
  public RemovedEnvironmentCulprit getRemovedEnvironmentCulprit(Label environment) {
    return removedEnvironmentCulprits.get(environment);
  }
}
