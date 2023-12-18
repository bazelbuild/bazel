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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** A {@link RunfilesSupplier} implementation for composing multiple instances. */
public final class CompositeRunfilesSupplier implements RunfilesSupplier {
  private final ImmutableList<RunfilesTree> runfilesTrees;

  /**
   * Create a composite {@link RunfilesSupplier} from a collection of suppliers. Suppliers earlier
   * in the collection take precedence over later suppliers.
   */
  public static RunfilesSupplier fromSuppliers(Collection<RunfilesSupplier> suppliers) {
    ImmutableList<RunfilesSupplier> nonEmptySuppliers =
        suppliers.stream()
            .filter((s) -> s != EmptyRunfilesSupplier.INSTANCE)
            .collect(toImmutableList());

    if (nonEmptySuppliers.isEmpty()) {
      return EmptyRunfilesSupplier.INSTANCE;
    }

    if (nonEmptySuppliers.size() == 1) {
      return Iterables.getOnlyElement(nonEmptySuppliers);
    }

    Set<PathFragment> execPaths = new HashSet<>();
    List<RunfilesTree> trees = new ArrayList<>();

    for (RunfilesSupplier supplier : nonEmptySuppliers) {
      for (RunfilesTree tree : supplier.getRunfilesTrees()) {
        if (execPaths.add(tree.getExecPath())) {
          trees.add(tree);
        }
      }
    }

    return new CompositeRunfilesSupplier(ImmutableList.copyOf(trees));
  }

  /**
   * Convenience method for creating a composite {@link RunfilesSupplier} from two other suppliers.
   */
  public static RunfilesSupplier of(RunfilesSupplier supplier1, RunfilesSupplier supplier2) {
    return fromSuppliers(ImmutableList.of(supplier1, supplier2));
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
