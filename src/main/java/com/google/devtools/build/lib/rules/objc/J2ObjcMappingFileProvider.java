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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * This provider is exported by java_library rules and proto_library rules with attribute
 * "j2objc_api_version=1" to support distributed J2ObjC translation and proto compilation.
 */
@Immutable
public final class J2ObjcMappingFileProvider implements TransitiveInfoProvider {

  private final NestedSet<Artifact> headerMappingFiles;
  private final NestedSet<Artifact> classMappingFiles;
  private final NestedSet<Artifact> dependencyMappingFiles;
  private final NestedSet<Artifact> archiveSourceMappingFiles;

  /**
   * Constructs a {@link J2ObjcMappingFileProvider} with mapping files to export mappings required
   * by J2ObjC translation and proto compilation.
   *
   * @param headerMappingFiles a nested set of header mapping files which map Java classes to
   *     their associated translated ObjC header. Used by J2ObjC to output correct import directives
   *     during translation.
   * @param classMappingFiles a nested set of class mapping files which map Java class names to
   *     their associated ObjC class names. Used to support J2ObjC package prefixes.
   * @param dependencyMappingFiles a nested set of dependency mapping files which map translated
   *     ObjC files to their translated direct dependency files. Used to support J2ObjC dead code
   *     analysis and removal.
   * @param archiveSourceMappingFiles a nested set of files containing mappings between J2ObjC
   *     static library archives and their associated J2ObjC-translated source files.
   */
  public J2ObjcMappingFileProvider(NestedSet<Artifact> headerMappingFiles,
      NestedSet<Artifact> classMappingFiles, NestedSet<Artifact> dependencyMappingFiles,
      NestedSet<Artifact> archiveSourceMappingFiles) {
    this.headerMappingFiles = headerMappingFiles;
    this.classMappingFiles = classMappingFiles;
    this.dependencyMappingFiles = dependencyMappingFiles;
    this.archiveSourceMappingFiles = archiveSourceMappingFiles;
  }

  /**
   * Returns the ObjC header to Java type mapping files for J2ObjC translation. J2ObjC needs these
   * mapping files to be able to output translated files with correct header import paths in the
   * same directories of the Java source files.
   */
  public NestedSet<Artifact> getHeaderMappingFiles() {
    return headerMappingFiles;
  }

  /**
   * Returns the Java class name to ObjC class name mapping files. J2ObjC transpiler and J2ObjC
   * proto plugin needs this mapping files to support "objc_class_prefix" proto option, which sets
   * the ObjC class prefix on generated protos.
   */
  public NestedSet<Artifact> getClassMappingFiles() {
    return classMappingFiles;
  }

  /**
   * Returns the mapping files containing file dependency information among the translated ObjC
   * source files. When flag --j2objc_dead_code_removal is specified, they are used to strip unused
   * object files inside J2ObjC static libraries before the linking action at binary level.
   */
  public NestedSet<Artifact> getDependencyMappingFiles() {
    return dependencyMappingFiles;
  }

  /**
   * Returns the files containing mappings between J2ObjC static library archives and their
   * associated J2ObjC-translated source files. When flag --j2objc_dead_code_removal is specified,
   * they are used to strip unused object files inside J2ObjC static libraries before the linking
   * action at binary level.
   */
  public NestedSet<Artifact> getArchiveSourceMappingFiles() {
    return archiveSourceMappingFiles;
  }

  /**
   * A builder for this provider that is optimized for collection information from transitive
   * dependencies.
   */
  public static final class Builder {
    private final NestedSetBuilder<Artifact> headerMappingFiles = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> classMappingFiles = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> depEntryFiles = NestedSetBuilder.stableOrder();
    private final NestedSetBuilder<Artifact> archiveSourceMappingFiles =
        NestedSetBuilder.stableOrder();

    public Builder addTransitive(J2ObjcMappingFileProvider provider) {
      headerMappingFiles.addTransitive(provider.getHeaderMappingFiles());
      classMappingFiles.addTransitive(provider.getClassMappingFiles());
      depEntryFiles.addTransitive(provider.getDependencyMappingFiles());
      archiveSourceMappingFiles.addTransitive(provider.getArchiveSourceMappingFiles());

      return this;
    }

    public J2ObjcMappingFileProvider build() {
      return new J2ObjcMappingFileProvider(
          headerMappingFiles.build(),
          classMappingFiles.build(),
          depEntryFiles.build(),
          archiveSourceMappingFiles.build());
    }
  }
}
