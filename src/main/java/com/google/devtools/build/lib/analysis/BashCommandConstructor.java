// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.vfs.PathFragment;

/** The class for constructing command line for Bash. */
public final class BashCommandConstructor implements CommandConstructor {

  private final PathFragment shellPath;
  private final String scriptNameSuffix;

  BashCommandConstructor(PathFragment shellPath, String scriptNameSuffix) {
    this.shellPath = shellPath;
    this.scriptNameSuffix = scriptNameSuffix;
  }

  @Override
  public ImmutableList<String> asExecArgv(Artifact scriptFileArtifact) {
    return ImmutableList.of(shellPath.getPathString(), scriptFileArtifact.getExecPathString());
  }

  @Override
  public ImmutableList<String> asExecArgv(String command) {
    return ImmutableList.of(shellPath.getPathString(), "-c", command);
  }

  @Override
  public Artifact commandAsScript(RuleContext ruleContext, String command) {
    String scriptFileName = ruleContext.getTarget().getName() + scriptNameSuffix;
    String scriptFileContents = "#!/bin/bash\n" + command;
    return FileWriteAction.createFile(
        ruleContext, scriptFileName, scriptFileContents, /*executable=*/ true);
  }
}
