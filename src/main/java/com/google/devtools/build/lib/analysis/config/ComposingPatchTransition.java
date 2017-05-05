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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.Iterables;

/**
 * A {@link ComposingSplitTransition} that only supports {@link PatchTransition}s
 *
 * <p>Calling code that doesn't want to have to handle splits should prefer this version.
 */
public class ComposingPatchTransition implements PatchTransition {
  private final ComposingSplitTransition delegate;

  public ComposingPatchTransition(PatchTransition transition1, PatchTransition transition2) {
    this.delegate = new ComposingSplitTransition(transition1, transition2);
  }

  @Override
  public boolean defaultsToSelf() {
    throw new UnsupportedOperationException(
        "dynamic configurations don't use global transition tables");
  }

  @Override
  public BuildOptions apply(BuildOptions options) {
    return Iterables.getOnlyElement(delegate.split(options));
  }
}

