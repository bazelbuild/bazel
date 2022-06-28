// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.proto.ProtoCommonApi;
import com.google.devtools.build.lib.vfs.PathFragment;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** Protocol buffers support for Starlark. */
public class BazelProtoCommon implements ProtoCommonApi {
  public static final BazelProtoCommon INSTANCE = new BazelProtoCommon();

  protected BazelProtoCommon() {}

  @StarlarkMethod(
      name = "ProtoSource",
      documented = false,
      parameters = {
        @Param(name = "source_file", doc = "The proto file."),
        @Param(name = "original_source_file", doc = "Original proto file."),
        @Param(name = "proto_path", doc = "Path to proto file."),
      },
      useStarlarkThread = true)
  public ProtoSource protoSource(
      Artifact sourceFile, Artifact originalSourceFile, String sourceRoot, StarlarkThread thread)
      throws EvalException {
    ProtoCommon.checkPrivateStarlarkificationAllowlist(thread);
    return new ProtoSource(sourceFile, originalSourceFile, PathFragment.create(sourceRoot));
  }

  @StarlarkMethod(
      name = "ProtoInfo",
      documented = false,
      parameters = {
        @Param(name = "direct_sources", doc = "Direct sources."),
        @Param(name = "proto_path", doc = "Proto path."),
        @Param(name = "transitive_sources", doc = "Transitive sources."),
        @Param(name = "transitive_proto_sources", doc = "Transitive proto sources."),
        @Param(name = "transitive_proto_path", doc = "Transitive proto path."),
        @Param(name = "check_deps_sources", doc = "Check deps sources."),
        @Param(name = "direct_descriptor_set", doc = "Direct descriptor set."),
        @Param(name = "transitive_descriptor_set", doc = "Transitive descriptor sets."),
        @Param(name = "exported_sources", doc = "Exported sources"),
      },
      useStarlarkThread = true)
  @SuppressWarnings("unchecked")
  public ProtoInfo protoInfo(
      StarlarkList<? extends ProtoSource> directSources,
      String directProtoSourceRoot,
      Depset transitiveProtoSources,
      Depset transitiveSources,
      Depset transitiveProtoSourceRoots,
      Depset strictImportableProtoSourcesForDependents,
      Artifact directDescriptorSet,
      Depset transitiveDescriptorSets,
      Depset exportedSources,
      StarlarkThread thread)
      throws EvalException {
    ProtoCommon.checkPrivateStarlarkificationAllowlist(thread);
    return new ProtoInfo(
        ((StarlarkList<ProtoSource>) directSources).getImmutableList(),
        PathFragment.create(directProtoSourceRoot),
        Depset.cast(transitiveSources, ProtoSource.class, "transitive_sources"),
        Depset.cast(transitiveProtoSources, Artifact.class, "transitive_proto_sources"),
        Depset.cast(transitiveProtoSourceRoots, String.class, "transitive_proto_path"),
        Depset.cast(
            strictImportableProtoSourcesForDependents, Artifact.class, "check_deps_sources"),
        directDescriptorSet,
        Depset.cast(transitiveDescriptorSets, Artifact.class, "transitive_descriptor_set"),
        Depset.cast(exportedSources, ProtoSource.class, "exported_sources"));
  }
}
