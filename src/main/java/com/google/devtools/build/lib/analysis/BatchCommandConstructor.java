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

/**
 * The class for constructing command line for Batch.
 */
public class BatchCommandConstructor implements CommandConstructor {

  // `cmd.exe` exists at C:\Windows\System32, which is in the default PATH on Windows.
  private static final String BATCH_BIN = "cmd.exe";
  private String scriptPostFix;

  BatchCommandConstructor(String scriptPostFix) {
    this.scriptPostFix = scriptPostFix;
  }

  @Override
  public ImmutableList<String> buildCommandLineSimpleArgv(String command) {
    return ImmutableList.of(BATCH_BIN, "/c", command);
  }

  @Override
  public Artifact buildCommandLineArtifact(RuleContext ruleContext, String command) {
    String scriptFileName = ruleContext.getTarget().getName() + this.scriptPostFix;
    String scriptFileContents = "@echo off\n" + command;
    return FileWriteAction.createFile(
        ruleContext, scriptFileName, scriptFileContents, /*executable=*/true);
  }

  @Override
  public ImmutableList<String> buildCommandLineArgvWithArtifact(Artifact scriptFileArtifact) {
    return buildCommandLineSimpleArgv(scriptFileArtifact.getExecPathString().replace('/', '\\'));
  }
}
