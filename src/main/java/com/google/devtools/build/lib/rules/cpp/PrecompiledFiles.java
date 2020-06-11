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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitionMode;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.util.FileType;

/**
 * A helper class that filters a given list of source files into different buckets - shared
 * libraries, static libraries, pic object files, and non-pic object files.
 */
public final class PrecompiledFiles {
  /** The Artifacts from srcs. */
  private final ImmutableList<Artifact> files;

  /**
   * Initializes this object with the artifacts obtained from the "srcs" attribute of the given rule
   * (this is the most common usage for this class).
   */
  public PrecompiledFiles(RuleContext ruleContext) {
    if (ruleContext.attributes().has("srcs", BuildType.LABEL_LIST)) {
      this.files = ruleContext.getPrerequisiteArtifacts("srcs", TransitionMode.TARGET).list();
    } else {
      this.files = ImmutableList.<Artifact>of();
    }
  }

  public Iterable<Artifact> getLibraries() {
    return FileType.filter(files, CppFileTypes.ARCHIVE, CppFileTypes.PIC_ARCHIVE,
        CppFileTypes.ALWAYS_LINK_LIBRARY, CppFileTypes.ALWAYS_LINK_PIC_LIBRARY,
        CppFileTypes.SHARED_LIBRARY,
        CppFileTypes.VERSIONED_SHARED_LIBRARY);
  }

  public Iterable<Artifact> getSharedLibraries() {
    return getSharedLibrariesFrom(files);
  }

  static Iterable<Artifact> getSharedLibrariesFrom(Iterable<Artifact> collection) {
    return FileType.filter(collection, CppFileTypes.SHARED_LIBRARY,
        CppFileTypes.VERSIONED_SHARED_LIBRARY);
  }

  public Iterable<Artifact> getStaticLibraries() {
    return FileType.filter(files, CppFileTypes.ARCHIVE);
  }

  public Iterable<Artifact> getAlwayslinkStaticLibraries() {
    return FileType.filter(files, CppFileTypes.ALWAYS_LINK_LIBRARY);
  }

  public Iterable<Artifact> getPicStaticLibraries() {
    return FileType.filter(files, CppFileTypes.PIC_ARCHIVE);
  }

  public Iterable<Artifact> getPicAlwayslinkLibraries() {
    return FileType.filter(files, CppFileTypes.ALWAYS_LINK_PIC_LIBRARY);
  }

  public Iterable<Artifact> getObjectFiles(final boolean usePic) {
    if (usePic) {
      return Iterables.filter(
          files,
          artifact -> {
            String filename = artifact.getExecPathString();

            // For compatibility with existing BUILD files, any ".o" files listed
            // in srcs are assumed to be position-independent code, or
            // at least suitable for inclusion in shared libraries, unless they
            // end with ".nopic.o". (The ".nopic.o" extension is an undocumented
            // feature to give users at least some control over this.) Note that
            // some target platforms do not require shared library code to be PIC.
            return CppFileTypes.PIC_OBJECT_FILE.matches(filename)
                || (CppFileTypes.OBJECT_FILE.matches(filename) && !filename.endsWith(".nopic.o"));
          });
    } else {
      return FileType.filter(files, CppFileTypes.OBJECT_FILE);
    }
  }
}

