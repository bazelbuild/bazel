// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * Info object propagating information about protocol buffer sources.
 */
@SkylarkModule(name = "ProtoSourcesProvider", doc = "")
public interface ProtoSourcesProviderApi<FileT extends FileApi> {

  @SkylarkCallable(
    name = "transitive_imports",
    doc = "Transitive imports including weak dependencies.",
    structField = true
  )
  public NestedSet<FileT> getTransitiveImports();

  @SkylarkCallable(
      name = "transitive_sources",
      doc = "Proto sources for this rule and all its dependent protocol buffer rules.",
      structField = true)
  // TODO(bazel-team): The difference between transitive imports and transitive proto sources
  // should never be used by Skylark or by an Aspect. One of these two should be removed,
  // preferably soon, before Skylark users start depending on them.
  public NestedSet<FileT> getTransitiveProtoSources();

  @SkylarkCallable(
      name = "direct_sources",
      doc = "Proto sources from the 'srcs' attribute.",
      structField = true)
  public ImmutableList<FileT> getDirectProtoSources();

  @SkylarkCallable(
      name = "check_deps_sources",
      doc =
          "Proto sources from the 'srcs' attribute. If the library is a proxy library "
              + "that has no sources, it contains the check_deps_sources "
              + "from this library's direct deps.",
      structField = true)
  public NestedSet<FileT> getStrictImportableProtoSourcesForDependents();

  @SkylarkCallable(
      name = "direct_descriptor_set",
      doc =
          "The <a href=\""
              + "https://github.com/google/protobuf/search?q=%22message+FileDescriptorSet%22+path%3A%2Fsrc"
              + "\">FileDescriptorSet</a> of the direct sources. "
              + "If no srcs, contains an empty file.",
      structField = true)
  public FileT getDirectDescriptorSet();

  @SkylarkCallable(
      name = "transitive_descriptor_sets",
      doc =
          "A set of <a href=\""
              + "https://github.com/google/protobuf/search?q=%22message+FileDescriptorSet%22+path%3A%2Fsrc"
              + "\">FileDescriptorSet</a> files of all dependent proto_library rules, "
              + "and this one's. "
              + "This is not the same as passing --include_imports to proto-compiler. "
              + "Will be empty if no dependencies.",
      structField = true)
  public NestedSet<FileT> getTransitiveDescriptorSets();

  @SkylarkCallable(
      name = "transitive_proto_path",
      doc = "A set of proto source roots collected from the transitive closure of this rule.",
      structField = true)
  public NestedSet<String> getTransitiveProtoSourceRoots();

  @SkylarkCallable(
      name = "proto_source_root",
      doc =
          "The directory relative to which the .proto files defined in the proto_library are "
              + "defined. For example, if this is 'a/b' and the rule has the file 'a/b/c/d.proto'"
              + " as a source, that source file would be imported as 'import c/d.proto'",
      structField = true)
  String getDirectProtoSourceRoot();
}
