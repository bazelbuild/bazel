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

package com.google.devtools.build.lib.vfs;

import java.util.Objects;

/**
 * Represents a workspace versioned by a revision number, such as a changelist.
 */
public abstract class RevisionedWorkspace extends AbstractWorkspace {
  private final long workspaceRevision;

  /**
   * Creates a new Workspace representation with the given user, name, and revision.
   */
  public RevisionedWorkspace(String workspaceUser, String workspaceName, long workspaceRevision) {
    super(workspaceUser, workspaceName);
    this.workspaceRevision = workspaceRevision;
  }

  public long getWorkspaceRevision() {
    return workspaceRevision;
  }

  @Override
  public boolean equals(Object other) {
    return super.equals(other)
        && other instanceof RevisionedWorkspace
        && ((RevisionedWorkspace) other).workspaceRevision == workspaceRevision;
  }

  @Override
  public int hashCode() {
    return Objects.hash(getWorkspaceUser(), getWorkspaceName(),
                        getWorkspaceRevision(), getWorkspaceAsPathSegment());
  }
}
