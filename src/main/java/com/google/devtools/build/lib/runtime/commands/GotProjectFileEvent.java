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
package com.google.devtools.build.lib.runtime.commands;

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * An event describing a project file which has been parsed.
 */
public class GotProjectFileEvent implements Postable {
  private final String projectFile;

  /**
   * Construct the event.
   * @param projectFile The workspace-relative path of the project file.
   */
  public GotProjectFileEvent(String projectFile) {
    this.projectFile = projectFile;
  }

  /** Returns the project file that was parsed. */
  public String getProjectFile() {
    return projectFile;
  }
}
