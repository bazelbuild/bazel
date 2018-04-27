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

package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A {@link ComposingSplitTransition} that only supports {@link PatchTransition}s
 *
 * <p>Calling code that doesn't want to have to handle splits should prefer this version.
 */
@AutoCodec
public class ComposingPatchTransition implements PatchTransition {
  private final ComposingSplitTransition delegate;

  /**
   * Creates a {@link ComposingPatchTransition} that applies the sequence: {@code fromOptions ->
   * transition1 -> transition2 -> toOptions }.
   *
   * <p>Note that it's possible to create silly transitions with this constructor (e.g., if one or
   * both of the transitions is NoTransition). Use composePatchTransitions instead, which checks for
   * these states and avoids instantiation appropriately.
   *
   * @see TransitionResolver#composePatchTransitions
   */
  public ComposingPatchTransition(PatchTransition transition1, PatchTransition transition2) {
    this(new ComposingSplitTransition(transition1, transition2));
  }

  @AutoCodec.Instantiator
  ComposingPatchTransition(ComposingSplitTransition delegate) {
    Preconditions.checkArgument(delegate.isPatchOnly());
    this.delegate = delegate;
  }

  @Override
  public BuildOptions apply(BuildOptions options) {
    return Iterables.getOnlyElement(delegate.split(options));
  }

  @Override
  public String getName() {
    return delegate.getName();
  }

  @Override
  public int hashCode() {
    return delegate.hashCode();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof ComposingPatchTransition
        && ((ComposingPatchTransition) other).delegate.equals(this.delegate);
  }
}

