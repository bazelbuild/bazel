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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/** Filters containers of android resources. */
public class ResourceFilter {
  private final ImmutableSet<Artifact> acceptedResources;
  private final Consumer<Artifact> filteredDependencyConsumer;
  private final boolean isEmpty;

  static final ResourceFilter empty() {
    return new ResourceFilter(ImmutableSet.of(), (artifact -> {}), /* isEmpty= */ true);
  }

  static final ResourceFilter of(
      ImmutableSet<Artifact> acceptedResources, Consumer<Artifact> filteredDependencyConsumer) {
    return new ResourceFilter(acceptedResources, filteredDependencyConsumer, /* isEmpty= */ false);
  }

  private ResourceFilter(
      ImmutableSet<Artifact> acceptedResources,
      Consumer<Artifact> filteredDependencyConsumer,
      boolean isEmpty) {
    this.acceptedResources = acceptedResources;
    this.filteredDependencyConsumer = filteredDependencyConsumer;
    this.isEmpty = isEmpty;
  }

  public Optional<NestedSet<Artifact>> maybeFilterDependencies(NestedSet<Artifact> artifacts) {
    if (isEmpty) {
      return Optional.empty();
    }

    List<Artifact> asList = artifacts.toList();
    List<Artifact> filtered =
        asList.stream().filter(acceptedResources::contains).collect(Collectors.toList());
    if (filtered.size() == asList.size()) {
      // No filtering needs to be done
      return Optional.empty();
    }

    return Optional.of(NestedSetBuilder.wrap(artifacts.getOrder(), filtered));
  }

  public NestedSet<ResourceContainer> filterDependencyContainers(
      NestedSet<ResourceContainer> resourceContainers) {
    if (isEmpty) {
      return resourceContainers;
    }

    NestedSetBuilder<ResourceContainer> builder =
        new NestedSetBuilder<>(resourceContainers.getOrder());

    for (ResourceContainer container : resourceContainers) {
      builder.add(container.filter(this, /* isDependency = */ true));
    }

    return builder.build();
  }

  Optional<ImmutableList<Artifact>> maybeFilter(
      ImmutableList<Artifact> artifacts, boolean isDependency) {
    if (isEmpty) {
      return Optional.empty();
    }

    boolean removedAny = false;
    ImmutableList.Builder<Artifact> filtered = ImmutableList.builder();

    for (Artifact artifact : artifacts) {
      if (acceptedResources.contains(artifact)) {
        filtered.add(artifact);
      } else {
        removedAny = true;
        if (isDependency) {
          filteredDependencyConsumer.accept(artifact);
        }
      }
    }

    if (!removedAny) {
      // No filtering was done, return the original
      return Optional.empty();
    }

    return Optional.of(filtered.build());
  }
}
