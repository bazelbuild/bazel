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

package com.google.devtools.build.lib.actions;


import com.google.common.collect.ImmutableList;
import java.util.Collection;

/** A {@link RunfilesSupplier} implementation for composing multiple instances. */
public final class CompositeRunfilesSupplier implements RunfilesSupplier {
  private final ImmutableList<RunfilesTree> runfilesTrees;

  /**
   * Create a runfiles supplier with an explicit list of runfiles trees.
   *
   * <p>No clever de-duplication is done aside from returning a singleton instance if there are no
   * runfiles trees. The expectation is that the return value of this method is ephemeral.
   */
  public static RunfilesSupplier fromRunfilesTrees(Collection<RunfilesTree> runfilesTrees) {
    if (runfilesTrees.isEmpty()) {
      return EmptyRunfilesSupplier.INSTANCE;
    }

    return new CompositeRunfilesSupplier(ImmutableList.copyOf(runfilesTrees));
  }

  /**
   * Create an instance combining all of {@code suppliers}, with earlier elements taking precedence.
   */
  private CompositeRunfilesSupplier(ImmutableList<RunfilesTree> runfilesTrees) {
    this.runfilesTrees = runfilesTrees;
  }

  @Override
  public ImmutableList<RunfilesTree> getRunfilesTrees() {
    return runfilesTrees;
  }

}
