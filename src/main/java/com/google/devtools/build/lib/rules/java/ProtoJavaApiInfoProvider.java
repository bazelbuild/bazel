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
package com.google.devtools.build.lib.rules.java;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import javax.annotation.Nullable;

/** An object that provides information about API versions used by a proto library. */
@AutoCodec
@Immutable
@AutoValue
public abstract class ProtoJavaApiInfoProvider implements TransitiveInfoProvider {
  @AutoCodec.Instantiator
  public static ProtoJavaApiInfoProvider create(
      JavaCompilationArgs javaCompilationContext,
      JavaCompilationArgs transitiveJavaCompilationArgs,
      JavaCompilationArgs transitiveJavaRpcLibs,
      JavaCompilationArgs transitiveJavaCompilationArgs1,
      JavaCompilationArgs transitiveJavaCompilationArgsMutable,
      JavaCompilationArgs transitiveJavaCompilationArgsImmutable,
      JavaCompilationArtifacts javaCompilationArtifacts1,
      JavaCompilationArtifacts javaCompilationArtifactsMutable,
      JavaCompilationArtifacts javaCompilationArtifactsImmutable,
      Artifact sourceJar1,
      Artifact sourceJarMutable,
      Artifact sourceJarImmutable,
      @Nullable ImmutableList<TransitiveInfoCollection> protoRuntime1,
      @Nullable ImmutableList<TransitiveInfoCollection> protoRuntimeMutable,
      @Nullable ImmutableList<TransitiveInfoCollection> protoRuntimeImmutable,
      JavaCompilationArgs transitiveProtoRuntime1,
      JavaCompilationArgs transitiveProtoRuntimeMutable,
      JavaCompilationArgs transitiveProtoRuntimeImmutable,
      boolean hasMixedApiVersions,
      int apiVersion,
      boolean supportsProto1,
      boolean supportsProto2Mutable,
      boolean hasProto1OnlyDependency) {
    return new AutoValue_ProtoJavaApiInfoProvider(
        javaCompilationContext,
        transitiveJavaCompilationArgs,
        transitiveJavaRpcLibs,
        transitiveJavaCompilationArgs1,
        transitiveJavaCompilationArgsMutable,
        transitiveJavaCompilationArgsImmutable,
        javaCompilationArtifacts1,
        javaCompilationArtifactsMutable,
        javaCompilationArtifactsImmutable,
        sourceJar1,
        sourceJarMutable,
        sourceJarImmutable,
        protoRuntime1,
        protoRuntimeMutable,
        protoRuntimeImmutable,
        transitiveProtoRuntime1,
        transitiveProtoRuntimeMutable,
        transitiveProtoRuntimeImmutable,
        hasMixedApiVersions,
        apiVersion,
        supportsProto1,
        supportsProto2Mutable,
        hasProto1OnlyDependency);
  }

  /**
   * Returns the Java artifacts created for this target. This method should only be called on
   * recursive visitations if {@code hasProtoLibraryShellInDeps()} returns {@code false}.
   */
  // TODO(bazel-team): this is mostly used by the tests
  public abstract JavaCompilationArgs getJavaCompilationContext();

  /**
   * Returns the the transitive Java artifacts created for this target.
   */
  // TODO(bazel-team): this is mostly used by the tests
  public abstract JavaCompilationArgs getTransitiveJavaCompilationArgs();

  /**
   * Returns the Java RPC library if any dependencies need it, null otherwise.
   */
  public abstract JavaCompilationArgs getTransitiveJavaRpcLibs();

  /**
   * Returns the artifacts for java compilation (API version 1) from the transitive
   * closure (excluding this target).
   */
  public abstract JavaCompilationArgs getTransitiveJavaCompilationArgs1();

  /**
   * Returns the artifacts for java compilation (API version 2, code for mutable API)
   * from the transitive closure (excluding this target).
   */
  public abstract JavaCompilationArgs getTransitiveJavaCompilationArgsMutable();

  /**
   * Returns the artifacts for java compilation (API version 2, code for immutable API)
   * from the transitive closure (excluding this target).
   */
  public abstract JavaCompilationArgs getTransitiveJavaCompilationArgsImmutable();

  /** Returns the artifacts for java compilation (API version 1) for only this target. */
  public abstract JavaCompilationArtifacts getJavaCompilationArtifacts1();

  /**
   * Returns the artifacts for java compilation (API version 2, code for mutable API) for only this
   * target.
   */
  public abstract JavaCompilationArtifacts getJavaCompilationArtifactsMutable();

  /**
   * Returns the artifacts for java compilation (API version 2, code for immutable API) for only
   * this target.
   */
  public abstract JavaCompilationArtifacts getJavaCompilationArtifactsImmutable();

  // The following 3 fields are the -src.jar artifact created by proto_library. If a certain
  // proto_library does not produce some artifact, it'll be null. This can happen for example when
  // there are no srcs, or when a certain combination of attributes results in "mutable" not being
  // produced.
  @Nullable
  public abstract Artifact sourceJar1();

  @Nullable
  public abstract Artifact sourceJarMutable();

  @Nullable
  public abstract Artifact sourceJarImmutable();

  // The following 3 fields are the targets that proto_library got from the proto runtime, including
  // Stubby. Different flavors can have different runtimes. If a certain proto_library does not
  // produce some artifact, it'll be null. This can happen for example when a certain combination of
  // attributes results in "mutable" not being produced.
  @Nullable
  public abstract ImmutableList<TransitiveInfoCollection> getProtoRuntime1();

  @Nullable
  public abstract ImmutableList<TransitiveInfoCollection> getProtoRuntimeMutable();

  @Nullable
  public abstract ImmutableList<TransitiveInfoCollection> getProtoRuntimeImmutable();

  // The following 3 fields are the jars that proto_library got from the proto runtime, and their
  // transitive dependencies.
  public abstract JavaCompilationArgs getTransitiveProtoRuntime1();

  public abstract JavaCompilationArgs getTransitiveProtoRuntimeMutable();

  public abstract JavaCompilationArgs getTransitiveProtoRuntimeImmutable();

  /**
   * Returns true if the transitive closure contains libraries with API versions other than the one
   * specified in this target. Building in mixed mode will add implicit deps for all the api_version
   * and might generate adapter code that has some runtime overhead.
   */
  public abstract boolean hasMixedApiVersions();

  /** Returns the API version. */
  public abstract int getApiVersion();

  /**
   * Returns true if this target support proto1 API.
   */
  public abstract boolean supportsProto1();

  /**
   * Returns true if this target support proto2 mutable API.
   */
  public abstract boolean supportsProto2Mutable();

  /**
   * Returns true if this target has a dependency (can be recursively) that only
   * supports proto1 API but not proto2 mutable API.
   */
  public abstract boolean hasProto1OnlyDependency();

}
