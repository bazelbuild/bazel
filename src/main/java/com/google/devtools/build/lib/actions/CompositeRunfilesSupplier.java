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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Map;

/** A {@link RunfilesSupplier} implementation for composing multiple instances. */
// TODO(michajlo): Move creation to factory constructor and optimize structure of the returned
// supplier, ie: [] -> EmptyRunfilesSupplier, [singleSupplier] -> supplier,
// [Empty, single] -> single, and so on.
@AutoCodec
public class CompositeRunfilesSupplier implements RunfilesSupplier {

  private final ImmutableList<RunfilesSupplier> suppliers;

  /** Create an instance with {@code first} taking precedence over {@code second}. */
  public CompositeRunfilesSupplier(RunfilesSupplier first, RunfilesSupplier second) {
    this(ImmutableList.of(first, second));
  }

  /**
   * Create an instance combining all of {@code suppliers}, with earlier elements taking precedence.
   */
  @AutoCodec.Instantiator
  public CompositeRunfilesSupplier(Iterable<RunfilesSupplier> suppliers) {
    this.suppliers = ImmutableList.copyOf(suppliers);
  }

  @Override
  public Iterable<Artifact> getArtifacts() {
    ImmutableSet.Builder<Artifact> result = ImmutableSet.builder();
    for (RunfilesSupplier supplier : suppliers) {
      result.addAll(supplier.getArtifacts());
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
  public ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings() throws IOException {
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
}
