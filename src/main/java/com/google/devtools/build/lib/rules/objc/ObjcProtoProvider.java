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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;

/**
 * A provider that provides all protos and portable proto filters information in the transitive
 * closure of its dependencies that are needed for generating and compiling only one version of
 * proto files.
 */
public class ObjcProtoProvider implements TransitiveInfoProvider {

  private final NestedSet<Artifact> protoSources;
  private final NestedSet<Artifact> portableProtoFilters;

  private ObjcProtoProvider(
      NestedSet<Artifact> protoSources, NestedSet<Artifact> portableProtoFilters) {
    this.protoSources = Preconditions.checkNotNull(protoSources);
    this.portableProtoFilters = Preconditions.checkNotNull(portableProtoFilters);
  }

  /**
   * Returns the set of all the protos that the dependencies of this provider has seen.
   */
  public NestedSet<Artifact> getProtoSources() {
    return protoSources;
  }

  /**
   * Returns the set of all the associated filters to the collected protos.
   */
  public NestedSet<Artifact> getPortableProtoFilters() {
    return portableProtoFilters;
  }

  /**
   * A builder for this context with an API that is optimized for collecting information from
   * several transitive dependencies.
   */
  public static final class Builder {
    private final NestedSetBuilder<Artifact> protoSources = NestedSetBuilder.linkOrder();
    private final NestedSetBuilder<Artifact> portableProtoFilters = NestedSetBuilder.linkOrder();

    /**
     * Adds all the protos to the set of dependencies.
     */
    public Builder addProtoSources(NestedSet<Artifact> protoSources) {
      this.protoSources.addTransitive(protoSources);
      return this;
    }

    /**
     * Adds all the proto filters to the set of dependencies.
     */
    public Builder addPortableProtoFilters(Iterable<Artifact> protoFilters) {
      this.portableProtoFilters.addAll(protoFilters);
      return this;
    }

    /**
     * Add all protos and filters from providers, and propagate them to any (transitive) dependers
     * on this ObjcProtoProvider.
     */
    public Builder addTransitive(Iterable<ObjcProtoProvider> providers) {
      for (ObjcProtoProvider provider : providers) {
        this.protoSources.addTransitive(provider.getProtoSources());
        this.portableProtoFilters.addTransitive(provider.getPortableProtoFilters());
      }
      return this;
    }

    /**
     * Whether this provider has any protos or filters.
     */
    public boolean isEmpty() {
      return protoSources.isEmpty() && portableProtoFilters.isEmpty();
    }

    public ObjcProtoProvider build() {
      return new ObjcProtoProvider(protoSources.build(), portableProtoFilters.build());
    }
  }
}
