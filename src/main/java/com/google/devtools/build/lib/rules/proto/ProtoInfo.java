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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.ProtoInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.proto.ProtoBootstrap;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Configured target classes that implement this class can contribute .proto files to the
 * compilation of proto_library rules.
 */
@Immutable
@AutoCodec
public final class ProtoInfo extends NativeInfo implements ProtoInfoApi<Artifact> {
  /** Provider class for {@link ProtoInfo} objects. */
  public static class Provider extends BuiltinProvider<ProtoInfo> implements ProtoInfoApi.Provider {
    public Provider() {
      super(ProtoBootstrap.PROTO_INFO_STARLARK_NAME, ProtoInfo.class);
    }
  }

  public static final Provider PROVIDER = new Provider();

  private final ImmutableList<Artifact> directProtoSources;
  private final ImmutableList<Artifact> originalDirectProtoSources;
  private final String directProtoSourceRoot;
  private final NestedSet<Artifact> transitiveProtoSources;
  private final NestedSet<Artifact> originalTransitiveProtoSources;
  private final NestedSet<String> transitiveProtoSourceRoots;
  private final NestedSet<Artifact> strictImportableProtoSourcesForDependents;
  private final NestedSet<Pair<Artifact, String>> strictImportableProtoSourcesImportPaths;
  private final NestedSet<Pair<Artifact, String>>
      strictImportableProtoSourcesImportPathsForDependents;
  private final Artifact directDescriptorSet;
  private final NestedSet<Artifact> transitiveDescriptorSets;
  private final NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> importableProtos;
  private final NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> exportedProtos;
  private final NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> transitiveProtos;

  @AutoCodec.Instantiator
  public ProtoInfo(
      ImmutableList<Artifact> directProtoSources,
      ImmutableList<Artifact> originalDirectProtoSources,
      String directProtoSourceRoot,
      NestedSet<Artifact> transitiveProtoSources,
      NestedSet<Artifact> originalTransitiveProtoSources,
      NestedSet<String> transitiveProtoSourceRoots,
      NestedSet<Artifact> strictImportableProtoSourcesForDependents,
      NestedSet<Pair<Artifact, String>> strictImportableProtoSourcesImportPaths,
      NestedSet<Pair<Artifact, String>> strictImportableProtoSourcesImportPathsForDependents,
      Artifact directDescriptorSet,
      NestedSet<Artifact> transitiveDescriptorSets,
      NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> importableProtos,
      NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> exportedProtos,
      NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> transitiveProtos,
      Location location) {
    super(PROVIDER, location);
    this.directProtoSources = directProtoSources;
    this.originalDirectProtoSources = originalDirectProtoSources;
    this.directProtoSourceRoot = directProtoSourceRoot;
    this.transitiveProtoSources = transitiveProtoSources;
    this.originalTransitiveProtoSources = originalTransitiveProtoSources;
    this.transitiveProtoSourceRoots = transitiveProtoSourceRoots;
    this.strictImportableProtoSourcesForDependents = strictImportableProtoSourcesForDependents;
    this.strictImportableProtoSourcesImportPaths = strictImportableProtoSourcesImportPaths;
    this.strictImportableProtoSourcesImportPathsForDependents =
        strictImportableProtoSourcesImportPathsForDependents;
    this.directDescriptorSet = directDescriptorSet;
    this.transitiveDescriptorSets = transitiveDescriptorSets;
    this.importableProtos = importableProtos;
    this.exportedProtos = exportedProtos;
    this.transitiveProtos = transitiveProtos;
  }

  /**
   * The proto source files that are used in compiling this {@code proto_library}.
   */
  @Override
  public ImmutableList<Artifact> getDirectProtoSources() {
    return directProtoSources;
  }

  /**
   * The non-virtual proto sources of the {@code proto_library} declaring this provider.
   *
   * <p>Different from {@link #getDirectProtoSources()} if a transitive dependency has {@code
   * import_prefix} or the like.
   */
  public ImmutableList<Artifact> getOriginalDirectProtoSources() {
    return originalDirectProtoSources;
  }

  /** The source root of the current library. */
  @Override
  public String getDirectProtoSourceRoot() {
    return directProtoSourceRoot;
  }

  /** The proto sources in the transitive closure of this rule. */
  @Override
  public Depset /*<Artifact>*/ getTransitiveProtoSourcesForStarlark() {
    return Depset.of(Artifact.TYPE, getTransitiveProtoSources());
  }

  public NestedSet<Artifact> getTransitiveProtoSources() {
    return transitiveProtoSources;
  }

  /**
   * The non-virtual transitive proto source files.
   *
   * <p>Different from {@link #getTransitiveProtoSources()} if a transitive dependency has {@code
   * import_prefix} or the like.
   */
  public NestedSet<Artifact> getOriginalTransitiveProtoSources() {
    return originalTransitiveProtoSources;
  }

  /**
   * The proto source roots of the transitive closure of this rule. These flags will be passed to
   * {@code protoc} in the specified order, via the {@code --proto_path} flag.
   */
  @Override
  public Depset /*<String>*/ getTransitiveProtoSourceRootsForStarlark() {
    return Depset.of(SkylarkType.STRING, transitiveProtoSourceRoots);
  }

  public NestedSet<String> getTransitiveProtoSourceRoots() {
    return transitiveProtoSourceRoots;
  }

  @Deprecated
  @Override
  public Depset /*<Artifact>*/ getTransitiveImports() {
    return getTransitiveProtoSourcesForStarlark();
  }

  /**
   * Returns the set of source files importable by rules directly depending on the rule declaring
   * this provider if strict dependency checking is in effect.
   *
   * <p>(strict dependency checking: when a target can only include / import source files from its
   * direct dependencies, but not from transitive ones)
   */
  @Override
  public Depset /*<Artifact>*/ getStrictImportableProtoSourcesForDependentsForStarlark() {
    return Depset.of(Artifact.TYPE, strictImportableProtoSourcesForDependents);
  }

  public NestedSet<Artifact> getStrictImportableProtoSourcesForDependents() {
    return strictImportableProtoSourcesForDependents;
  }

  /**
   * Returns the set of source files and import paths importable by rules directly depending on the
   * rule declaring this provider if strict dependency checking is in effect.
   *
   * <p>(strict dependency checking: when a target can only include / import source files from its
   * direct dependencies, but not from transitive ones)
   */
  public NestedSet<Pair<Artifact, String>>
      getStrictImportableProtoSourcesImportPathsForDependents() {
    return strictImportableProtoSourcesImportPathsForDependents;
  }

  /**
   * Returns the set of source files importable by the rule declaring this provider if strict
   * dependency checking is in effect.
   *
   * <p>(strict dependency checking: when a target can only include / import source files from its
   * direct dependencies, but not from transitive ones)
   */
  public NestedSet<Pair<Artifact, String>> getStrictImportableProtoSourcesImportPaths() {
    return strictImportableProtoSourcesImportPaths;
  }

  /**
   * Be careful while using this artifact - it is the parsing of the transitive set of .proto files.
   * It's possible to cause a O(n^2) behavior, where n is the length of a proto chain-graph.
   * (remember that proto-compiler reads all transitive .proto files, even when producing the
   * direct-srcs descriptor set)
   */
  @Override
  public Artifact getDirectDescriptorSet() {
    return directDescriptorSet;
  }

  /**
   * Be careful while using this artifact - it is the parsing of the transitive set of .proto files.
   * It's possible to cause a O(n^2) behavior, where n is the length of a proto chain-graph.
   * (remember that proto-compiler reads all transitive .proto files, even when producing the
   * direct-srcs descriptor set)
   */
  @Override
  public Depset /*<Artifact>*/ getTransitiveDescriptorSetsForStarlark() {
    return Depset.of(Artifact.TYPE, transitiveDescriptorSets);
  }

  public NestedSet<Artifact> getTransitiveDescriptorSets() {
    return transitiveDescriptorSets;
  }

  /**
   * The {@code .proto} that are importable by this {@link ProtoInfo}. Usually, this contains files
   * from {@code srcs} and exported files from {@code deps}, but not from {@code exports} (unless
   * they are also in {@code deps}).
   *
   * <p>For efficiency, these are pairs of {@code proto_source_root -> [sources]}. There may be
   * multiple occasions of the same {@code proto_source_root}.
   */
  NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> getImportableProtos() {
    return importableProtos;
  }

  /**
   * The {@code .proto} files that are exported by this {@link ProtoInfo}. Usually, this contains
   * files from {@code srcs} and exported files from {@code exports}.
   *
   * <p>For efficiency, these are pairs of {@code proto_source_root -> [sources]}. There may be
   * multiple occasions of the same {@code proto_source_root}.
   */
  NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> getExportedProtos() {
    return exportedProtos;
  }

  /**
   * The transitive closure of all {@code .proto} files reachable from this {@link ProtoInfo}.
   *
   * <p>For efficiency, these are pairs of {@code proto_source_root -> [sources]}. There may be
   * multiple occasions of the same {@code proto_source_root}.
   */
  NestedSet<Pair<PathFragment, ImmutableList<Artifact>>> getTransitiveProtos() {
    return transitiveProtos;
  }
}
