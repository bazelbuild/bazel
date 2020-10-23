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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** Result of {@link ProcessPackageDirectory#getPackageExistenceAndSubdirDeps}. */
public class ProcessPackageDirectoryResult {
  static final ProcessPackageDirectoryResult EMPTY_RESULT =
      new ProcessPackageDirectoryResult(false, ImmutableList.<SkyKey>of(), ImmutableMap.of());
  private final boolean packageExists;
  private final Iterable<SkyKey> childDeps;
  private final ImmutableMap<SkyKey, SkyValue> additionalValuesToAggregate;

  /** {@code childDeps} and {@code additionalValuesToAggregate} must be disjoint. */
  public ProcessPackageDirectoryResult(
      boolean packageExists,
      Iterable<SkyKey> childDeps,
      ImmutableMap<SkyKey, SkyValue> additionalValuesToAggregate) {
    this.packageExists = packageExists;
    this.childDeps = childDeps;
    this.additionalValuesToAggregate = additionalValuesToAggregate;
  }

  public boolean packageExists() {
    return packageExists;
  }

  public Iterable<SkyKey> getChildDeps() {
    return childDeps;
  }

  public ImmutableMap<SkyKey, SkyValue> getAdditionalValuesToAggregate() {
    return additionalValuesToAggregate;
  }
}
