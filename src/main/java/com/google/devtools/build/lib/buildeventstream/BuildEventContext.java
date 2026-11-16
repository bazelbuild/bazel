// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream;

/**
 * Interface for providing {@link BuildEvent}s with the convertes needed for computing the protobuf
 * representation.
 */
public interface BuildEventContext {
  /**
   * Return the {@link PathConverter} to be used in order to obtain URIs for the file paths to be
   * reported in the event.
   */
  PathConverter pathConverter();

  /**
   * Return the {@link ArtifactGroupNamer} that can be used to refer to a {@code
   * NestedSet<Artifact>} declared via the {@link
   * com.google.devtools.build.lib.actions.EventReportingArtifacts} interface.
   */
  ArtifactGroupNamer artifactGroupNamer();

  /**
   * Returns the options for the build event stream.
   */
  BuildEventProtocolOptions getOptions();

  /** Mode in which files are reported in an output group: recursive filesets, inline, or both. */
  enum OutputGroupFileMode {
    NAMED_SET_OF_FILES_ONLY,
    INLINE_ONLY,
    BOTH;

    /**
     * Whether the event of a target or aspect refers to the files of an output group in this mode
     * through named sets of files, whose own events name and declare the files.
     */
    public boolean reportsNamedSetsOfFiles() {
      return this == NAMED_SET_OF_FILES_ONLY || this == BOTH;
    }

    /**
     * Whether the event of a target or aspect names the files of an output group in this mode
     * itself. The event has to declare the files it names in {@link
     * BuildEvent#referencedLocalFiles(BuildEventProtocolOptions)}: both ask this method, so that
     * they agree.
     */
    public boolean reportsFilesInline() {
      return this == INLINE_ONLY || this == BOTH;
    }
  }

  /** Return the {@link OutputGroupFileMode} to use for a given output group. */
  default OutputGroupFileMode getFileModeForOutputGroup(String outputGroup) {
    return OutputGroupFileMode.NAMED_SET_OF_FILES_ONLY;
  }
}
