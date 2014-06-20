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

import com.google.common.base.Preconditions;

import java.util.Objects;

/**
 * Class for storing and representing information common to all workspaces.
 * Derived classes can add additional information and should update the path segment.
 */
public abstract class AbstractWorkspace {
  private final String workspaceUser;
  private final String workspaceName;

  /**
   * Creates a new Workspace representation with the given user and name.
   */
  public AbstractWorkspace(String workspaceUser, String workspaceName) {
    this.workspaceUser = Preconditions.checkNotNull(workspaceUser);
    this.workspaceName = Preconditions.checkNotNull(workspaceName);
  }

  public String getWorkspaceUser() {
    return workspaceUser;
  }

  public String getWorkspaceName() {
    return workspaceName;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof AbstractWorkspace)) {
      return false;
    }
    AbstractWorkspace otherWorkspace = (AbstractWorkspace) other;
    return workspaceUser.equals(otherWorkspace.workspaceUser)
        && workspaceName.equals(otherWorkspace.workspaceName)
        && getWorkspaceAsPathSegment().equals(otherWorkspace.getWorkspaceAsPathSegment());
  }

  @Override
  public int hashCode() {
    return Objects.hash(workspaceUser, workspaceName, getWorkspaceAsPathSegment());
  }

  /**
   * Creates a relative path representation of the workspace which can be
   * inserted into a Path or PathFragment.
   */
  public abstract PathFragment getWorkspaceAsPathSegment();
}
