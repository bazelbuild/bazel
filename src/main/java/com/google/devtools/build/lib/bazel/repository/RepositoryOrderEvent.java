// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * Event indicating a canonical (execution-independent) order of external repositories.
 *
 * <p>This order can be used to report about the execution of external repositories in a
 * reproducible way. In particular, in this way meaningful diffs can be obtained in a resolved file.
 */
public class RepositoryOrderEvent implements Postable {

  private final ImmutableList<String> orderedNames;

  public RepositoryOrderEvent(ImmutableList<String> orderedNames) {
    this.orderedNames = orderedNames;
  }

  public ImmutableList<String> getOrderedNames() {
    return orderedNames;
  }
}
