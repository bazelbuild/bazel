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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.packages.FileTarget;
import com.google.devtools.build.lib.util.FileType;

/**
 * A ConfiguredTarget for a source FileTarget.  (Generated files use a
 * subclass, OutputFileConfiguredTarget.)
 */
public abstract class FileConfiguredTarget extends AbstractConfiguredTarget
    implements FileType.HasFilename {

  FileConfiguredTarget(TargetContext targetContext) {
    super(targetContext);
  }

  @Override
  public FileTarget getTarget() {
    return (FileTarget) super.getTarget();
  }

  public abstract Artifact getArtifact();

  /**
   *  Returns the file type of this file target.
   */
  @Override
  public String getFilename() {
    return getTarget().getFilename();
  }

  @Override
  public Artifact getExecutable() {
    // We optimistically return the Artifact here as we have no means of actually knowing for sure
    // if the Artifact is executable.
    return getArtifact();
  }
}
