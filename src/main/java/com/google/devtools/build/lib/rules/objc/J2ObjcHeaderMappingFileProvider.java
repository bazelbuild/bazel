// Copyright 2014 Google Inc. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * This provider is exported by java_library rules to supply ObjC header to Java type mapping files
 * for J2ObjC translation. J2ObjC needs the mapping files to be able to output translated files with
 * correct header import paths in the same directories of the Java source files.
 */
@Immutable
public final class J2ObjcHeaderMappingFileProvider implements TransitiveInfoProvider {
  private final NestedSet<Artifact> mappingFiles;

  public J2ObjcHeaderMappingFileProvider(NestedSet<Artifact> mappingFiles) {
    this.mappingFiles = mappingFiles;
  }

  public NestedSet<Artifact> getMappingFiles() {
    return mappingFiles;
  }
}
