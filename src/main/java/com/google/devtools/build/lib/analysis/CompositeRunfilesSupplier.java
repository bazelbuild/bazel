// Copyright 2015 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Map;

/** RunfilesSupplier implementation composing instances. */
public class CompositeRunfilesSupplier implements RunfilesSupplier {

  private final RunfilesSupplier first;
  private final RunfilesSupplier second;

  /** Create an instance with {@code first} taking precedence over {@code second}. */
  public CompositeRunfilesSupplier(RunfilesSupplier first, RunfilesSupplier second) {
    this.first = Preconditions.checkNotNull(first);
    this.second = Preconditions.checkNotNull(second);
  }

  @Override
  public Iterable<Artifact> getArtifacts() {
    ImmutableSet.Builder<Artifact> result = ImmutableSet.builder();
    result.addAll(first.getArtifacts());
    result.addAll(second.getArtifacts());
    return result.build();
  }

  @Override
  public ImmutableSet<PathFragment> getRunfilesDirs() {
    ImmutableSet.Builder<PathFragment> result = ImmutableSet.builder();
    result.addAll(first.getRunfilesDirs());
    result.addAll(second.getRunfilesDirs());
    return result.build();
  }

  @Override
  public ImmutableMap<PathFragment, Map<PathFragment, Artifact>> getMappings() throws IOException {
    Map<PathFragment, Map<PathFragment, Artifact>> result = Maps.newHashMap();
    result.putAll(second.getMappings());
    result.putAll(first.getMappings());
    return ImmutableMap.copyOf(result);
  }
}
