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
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A provider that provides all protos and portable proto filters information in the transitive
 * closure of its dependencies that are needed for generating and compiling only one version of
 * proto files.
 *
 * <p>This provider also propagates the headers and search path for the protobuf runtime library.
 * This solves the issue that the proto bundling behavior (gather all the protos in the top target
 * and generate, compile and link only one version in the final binary) needs this data at the
 * linking target but the dependency on the runtime library is defined on the objc_proto_library.
 *
 * <p>Ideally we should make objc_binary (and other linking targets such as ios_extension_binary)
 * depend on the runtime library's ObjcProvider. Unfortunately this runs into a bug where Xcode
 * project generation cannot handle the dependency if it points to a label in an external workspace
 * (such as {@code @bazel_tools}). To avoid breaking Xcode project generation for all binary targets
 * all the time (whether protos are used or not), the dependency is specified on objc_proto_library
 * instead.
 */
public class ObjcProtoProvider extends NativeInfo {

  /** Starlark name for the ObjcProtoProvider. */
  public static final String STARLARK_NAME = "ObjcProto";

  /** Starlark constructor and identifier for AppleExecutableBinaryInfo. */
  public static final BuiltinProvider<ObjcProtoProvider> STARLARK_CONSTRUCTOR =
      new BuiltinProvider<ObjcProtoProvider>(STARLARK_NAME, ObjcProtoProvider.class) {};

  private final NestedSet<Artifact> protoFiles;
  private final NestedSet<Artifact> protobufHeaders;
  private final NestedSet<PathFragment> protobufHeaderSearchPaths;
  private final NestedSet<Artifact> portableProtoFilters;

  private ObjcProtoProvider(
      NestedSet<Artifact> protoFiles,
      NestedSet<Artifact> portableProtoFilters,
      NestedSet<Artifact> protobufHeaders,
      NestedSet<PathFragment> protobufHeaderSearchPaths) {
    this.protoFiles = Preconditions.checkNotNull(protoFiles);
    this.portableProtoFilters = Preconditions.checkNotNull(portableProtoFilters);
    this.protobufHeaders = Preconditions.checkNotNull(protobufHeaders);
    this.protobufHeaderSearchPaths = Preconditions.checkNotNull(protobufHeaderSearchPaths);
  }

  @Override
  public BuiltinProvider<ObjcProtoProvider> getProvider() {
    return STARLARK_CONSTRUCTOR;
  }

  /** Returns the set of all proto files that the dependencies of this provider has seen. */
  public NestedSet<Artifact> getProtoFiles() {
    return protoFiles;
  }

  /** Returns the header artifacts provided by the Protobuf library. */
  public NestedSet<Artifact> getProtobufHeaders() {
    return protobufHeaders;
  }

  /** Returns the header search paths provided by the Protobuf library. */
  public NestedSet<PathFragment> getProtobufHeaderSearchPaths() {
    return protobufHeaderSearchPaths;
  }

  /** Returns the set of all the associated filters to the collected protos. */
  public NestedSet<Artifact> getPortableProtoFilters() {
    return portableProtoFilters;
  }

  /**
   * A builder for this context with an API that is optimized for collecting information from
   * several transitive dependencies.
   */
  public static final class Builder {
    private final NestedSetBuilder<Artifact> protoFiles = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> portableProtoFilters = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> protobufHeaders = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<PathFragment> protobufHeaderSearchPaths =
        NestedSetBuilder.linkOrder();

    /** Adds proto files to be propagated. */
    public Builder addProtoFiles(NestedSet<Artifact> protoFiles) {
      this.protoFiles.addTransitive(protoFiles);
      return this;
    }

    /** Adds the header artifacts provided by the Protobuf library. */
    public Builder addProtobufHeaders(NestedSet<Artifact> protobufHeaders) {
      this.protobufHeaders.addTransitive(protobufHeaders);
      return this;
    }

    /** Adds the header search paths provided by the Protobuf library. */
    public Builder addProtobufHeaderSearchPaths(NestedSet<PathFragment> protobufHeaderSearchPaths) {
      this.protobufHeaderSearchPaths.addTransitive(protobufHeaderSearchPaths);
      return this;
    }

    /** Adds all the proto filters to the set of dependencies. */
    public Builder addPortableProtoFilters(NestedSet<Artifact> protoFilters) {
      this.portableProtoFilters.addTransitive(protoFilters);
      return this;
    }

    /**
     * Add all protos and filters from providers, and propagate them to any (transitive) dependers
     * on this ObjcProtoProvider.
     */
    public Builder addTransitive(Iterable<ObjcProtoProvider> providers) {
      for (ObjcProtoProvider provider : providers) {
        this.protoFiles.addTransitive(provider.getProtoFiles());
        this.portableProtoFilters.addTransitive(provider.getPortableProtoFilters());
        this.protobufHeaders.addTransitive(provider.getProtobufHeaders());
        this.protobufHeaderSearchPaths.addTransitive(provider.getProtobufHeaderSearchPaths());
      }
      return this;
    }

    /** Whether this provider has any protos or filters. */
    public boolean isEmpty() {
      return protoFiles.isEmpty() && portableProtoFilters.isEmpty();
    }

    public ObjcProtoProvider build() {
      return new ObjcProtoProvider(
          protoFiles.build(),
          portableProtoFilters.build(),
          protobufHeaders.build(),
          protobufHeaderSearchPaths.build());
    }
  }
}
