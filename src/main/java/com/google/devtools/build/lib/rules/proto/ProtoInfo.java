// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.ProtoInfoApi;
import javax.annotation.Nullable;

/**
 * Configured target classes that implement this class can contribute .proto files to the
 * compilation of proto_library rules.
 */
@AutoValue
@Immutable
@AutoCodec
public abstract class ProtoInfo implements TransitiveInfoProvider, ProtoInfoApi<Artifact> {
  /** The name of the field in Skylark used to access this class. */
  public static final String SKYLARK_NAME = "proto";

  @AutoCodec.Instantiator
  public static ProtoInfo create(
      ImmutableList<Artifact> directProtoSources,
      String directProtoSourceRoot,
      NestedSet<Artifact> transitiveProtoSources,
      NestedSet<String> transitiveProtoSourceRoots,
      NestedSet<Artifact> strictImportableProtoSourcesForDependents,
      NestedSet<Artifact> strictImportableProtoSources,
      NestedSet<String> strictImportableProtoSourceRoots,
      NestedSet<Artifact> exportedProtoSources,
      NestedSet<String> exportedProtoSourceRoots,
      Artifact directDescriptorSet,
      NestedSet<Artifact> transitiveDescriptorSets) {
    return new AutoValue_ProtoInfo(
        directProtoSources,
        directProtoSourceRoot,
        transitiveProtoSources,
        transitiveProtoSourceRoots,
        strictImportableProtoSourcesForDependents,
        strictImportableProtoSources,
        strictImportableProtoSourceRoots,
        exportedProtoSources,
        exportedProtoSourceRoots,
        directDescriptorSet,
        transitiveDescriptorSets);
  }

  /** The proto sources of the {@code proto_library} declaring this provider. */
  @Override
  public abstract ImmutableList<Artifact> getDirectProtoSources();

  /** The source root of the current library. */
  @Override
  public abstract String getDirectProtoSourceRoot();

  /** The proto sources in the transitive closure of this rule. */
  @Override
  public abstract NestedSet<Artifact> getTransitiveProtoSources();

  /**
   * The proto source roots of the transitive closure of this rule. These flags will be passed to
   * {@code protoc} in the specified order, via the {@code --proto_path} flag.
   */
  @Override
  public abstract NestedSet<String> getTransitiveProtoSourceRoots();

  @Deprecated
  @Override
  public NestedSet<Artifact> getTransitiveImports() {
    return getTransitiveProtoSources();
  }

  /**
   * Returns the set of source files importable by rules directly depending on the rule declaring
   * this provider if strict dependency checking is in effect.
   *
   * <p>(strict dependency checking: when a target can only include / import source files from its
   * direct dependencies, but not from transitive ones)
   */
  @Override
  public abstract NestedSet<Artifact> getStrictImportableProtoSourcesForDependents();

  /**
   * Returns the set of source files importable by the rule declaring this provider if strict
   * dependency checking is in effect.
   *
   * <p>(strict dependency checking: when a target can only include / import source files from its
   * direct dependencies, but not from transitive ones)
   */
  public abstract NestedSet<Artifact> getStrictImportableProtoSources();

  /**
   * Returns the proto source roots of the dependencies whose sources can be imported if strict
   * dependency checking is in effect.
   *
   * <p>(strict dependency checking: when a target can only include / import source files from its
   * direct dependencies, but not from transitive ones)
   */
  public abstract NestedSet<String> getStrictImportableProtoSourceRoots();

  /**
   * Returns the .proto files that are the direct srcs of the exported dependencies of this rule.
   */
  @Nullable
  public abstract NestedSet<Artifact> getExportedProtoSources();

  public abstract NestedSet<String> getExportedProtoSourceRoots();

  /**
   * Be careful while using this artifact - it is the parsing of the transitive set of .proto files.
   * It's possible to cause a O(n^2) behavior, where n is the length of a proto chain-graph.
   * (remember that proto-compiler reads all transitive .proto files, even when producing the
   * direct-srcs descriptor set)
   */
  @Override
  public abstract Artifact getDirectDescriptorSet();

  /**
   * Be careful while using this artifact - it is the parsing of the transitive set of .proto files.
   * It's possible to cause a O(n^2) behavior, where n is the length of a proto chain-graph.
   * (remember that proto-compiler reads all transitive .proto files, even when producing the
   * direct-srcs descriptor set)
   */
  @Override
  public abstract NestedSet<Artifact> getTransitiveDescriptorSets();

  ProtoInfo() {}
}
