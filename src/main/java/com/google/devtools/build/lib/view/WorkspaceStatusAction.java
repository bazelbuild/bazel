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

package com.google.devtools.build.lib.view;

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactOwner;

import java.util.Map;
import java.util.UUID;

/**
 * An action writing the workspace status files.
 *
 * <p>These files represent information about the environment the build was run in. They are used
 * by language-specific build info factories to make the data in them available for individual
 * languages (e.g. by turning them into .h files for C++)
 *
 * <p>The format of these files a list of key-value pairs, one for each line. The key and the value
 * are separated by a space.
 *
 * <p>There are two of these files: volatile and stable. Changes in the volatile file do not
 * cause rebuilds if no other file is changed. This is useful for frequently-changing information
 * that does not significantly affect the build, e.g. the current time.
 */
public abstract class WorkspaceStatusAction extends AbstractAction {
  /**
   * Factory for {@link WorkspaceStatusAction}.
   */
  public interface Factory {
    /**
     * Creates the workspace status action.
     *
     * <p>If the objects returned for two builds are equals, the workspace status action can be
     * be reused between them. Note that this only applies to the action object itself (the action
     * will be unconditionally re-executed on every build)
     */
    WorkspaceStatusAction createWorkspaceStatusAction(
        ArtifactFactory artifactFactory, ArtifactOwner artifactOwner, Supplier<UUID> buildId);

    /**
     * Creates a dummy workspace status map. Used in cases where the build failed, so that part of
     * the workspace status is nevertheless available.
     */
    Map<String, String> createDummyWorkspaceStatus();
  }

  protected WorkspaceStatusAction(ActionOwner owner,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs) {
    super(owner, inputs, outputs);
  }

  /**
   * The volatile status artifact containing items that may change even if nothing changed
   * between the two builds, e.g. current time.
   */
  public abstract Artifact getVolatileStatus();

  /**
   * The stable status artifact containing items that change only if information relevant to the
   * build changes, e.g. the name of the user running the build or the hostname.
   */
  public abstract Artifact getStableStatus();
}
