// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildeventstream;

/** Interface for conversion of paths to URIs. */
// TODO(lpino): This interface shouldn't exist since there's only trivial implementation of it.
// However, it's really hard to move this class to the right package because of package boundaries.
public interface ArtifactGroupNamer {
  /**
   * Return the name of a declared group of artifacts, identified by the identifier of their {@link
   * NestedSetView}. A {@link BuildEvent} should only assume that this function is defined if the
   * corresponding {@link NestedSet<Artifact>} is declared via the {@link EventReportingArtifacts}
   * interface. On undefined positions, the value null is returned.
   */
  BuildEventStreamProtos.BuildEventId.NamedSetOfFilesId apply(Object id);
}
