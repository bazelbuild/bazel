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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.Map;

/** A {@link RunfilesSupplier} implementation for composing multiple instances. */
@AutoCodec
public class CompositeRunfilesSupplier implements RunfilesSupplier {

  private final ImmutableList<RunfilesSupplier> suppliers;

  /**
   * Create a composite {@link RunfilesSupplier} from a collection of suppliers. Suppliers earlier
   * in the collection take precedence over later suppliers.
   */
  public static RunfilesSupplier fromSuppliers(Collection<RunfilesSupplier> suppliers) {
    ImmutableList<RunfilesSupplier> finalSuppliers =
        suppliers.stream()
            .filter((s) -> s != EmptyRunfilesSupplier.INSTANCE)
            .collect(ImmutableList.toImmutableList());
    if (finalSuppliers.isEmpty()) {
      return EmptyRunfilesSupplier.INSTANCE;
    }
    if (finalSuppliers.size() == 1) {
      return finalSuppliers.get(0);
    }
    return new CompositeRunfilesSupplier(finalSuppliers);
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
  @AutoCodec.Instantiator
  @AutoCodec.VisibleForSerialization
  CompositeRunfilesSupplier(ImmutableList<RunfilesSupplier> suppliers) {
    this.suppliers = suppliers;
  }

  @Override
  public NestedSet<Artifact> getArtifacts() {
    NestedSetBuilder<Artifact> result = NestedSetBuilder.stableOrder();
    for (RunfilesSupplier supplier : suppliers) {
      result.addTransitive(supplier.getArtifacts());
    }
    return result.build();
  }

  @Override
  public ImmutableSet<PathFragment> getRunfilesDirs() {
    ImmutableSet.Builder<PathFragment> result = ImmutableSet.builder();
    for (RunfilesSupplier supplier : suppliers) {
      result.addAll(supplier.getRunfilesDirs());
    }
    return result.build();
  }

  @Override
  public ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings() {
    Map<PathFragment, Map<PathFragment, Artifact>> result = Maps.newHashMap();
    for (RunfilesSupplier supplier : suppliers) {
      Map<PathFragment, Map<PathFragment, Artifact>> mappings = supplier.getMappings();
      for (Map.Entry<PathFragment, Map<PathFragment, Artifact>> entry : mappings.entrySet()) {
        result.putIfAbsent(entry.getKey(), entry.getValue());
      }
    }
    return ImmutableMap.copyOf(result);
  }

  @Override
  public ImmutableList<Artifact> getManifests() {
    ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (RunfilesSupplier supplier : suppliers) {
      result.addAll(supplier.getManifests());
    }
    return result.build();
  }

  @Override
  public boolean isBuildRunfileLinks(PathFragment runfilesDir) {
    for (RunfilesSupplier supplier : suppliers) {
      if (supplier.isBuildRunfileLinks(runfilesDir)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean isRunfileLinksEnabled(PathFragment runfilesDir) {
    for (RunfilesSupplier supplier : suppliers) {
      if (supplier.isRunfileLinksEnabled(runfilesDir)) {
        return true;
      }
    }
    return false;
  }
}
