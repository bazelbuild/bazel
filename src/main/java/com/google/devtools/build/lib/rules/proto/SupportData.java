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

package com.google.devtools.build.lib.rules.proto;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

/** A helper class for the *Support classes containing some data from ProtoLibrary. */
@AutoCodec
@AutoValue
@Immutable
public abstract class SupportData {
  @AutoCodec.Instantiator
  public static SupportData create(
      ImmutableList<Artifact> directProtoSources,
      NestedSet<Artifact> protosInDirectDeps,
      NestedSet<Artifact> transitiveImports,
      NestedSet<String> transitiveProtoPathFlags,
      String protoSourceRoot,
      NestedSet<String> directProtoSourceRoots,
      boolean hasProtoSources,
      @Nullable NestedSet<Artifact> protosInExports,
      @Nullable NestedSet<String> exportedProtoSourceRoots) {
    return new AutoValue_SupportData(
        directProtoSources,
        transitiveImports,
        protosInDirectDeps,
        transitiveProtoPathFlags,
        protoSourceRoot,
        directProtoSourceRoots,
        hasProtoSources,
        protosInExports,
        exportedProtoSourceRoots);
  }

  public abstract ImmutableList<Artifact> getDirectProtoSources();

  public abstract NestedSet<Artifact> getTransitiveImports();

  /**
   * .proto files in the direct dependencies of this proto_library. Used for strict deps checking.
   */
  public abstract NestedSet<Artifact> getProtosInDirectDeps();

  /**
   * Directories of .proto sources collected from the transitive closure. These flags will be passed
   * to {@code protoc} in the specified order, via the {@code --proto_path} flag.
   */
  public abstract NestedSet<String> getTransitiveProtoPathFlags();

  /** The {@code proto_source_root} of the current library. */
  public abstract String getProtoSourceRoot();

  /**
   * The {@code proto_source_root}'s collected from the current library and the direct dependencies.
   */
  public abstract NestedSet<String> getDirectProtoSourceRoots();

  public abstract boolean hasProtoSources();

  /** .proto files in the exported dependencies of this proto_library. */
  public abstract @Nullable NestedSet<Artifact> getProtosInExports();

  /**
   * The {@code proto_source_root}'s collected from the current library and the exported
   * dependencies.
   */
  public abstract @Nullable NestedSet<String> getExportedProtoSourceRoots();

  SupportData() {}
}
