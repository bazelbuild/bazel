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
 * The class for constructing command line for Powershell.
 */
public class PowershellCommandConstructor implements CommandConstructor {

  // `powershell.exe` exists at C:\Windows\System32\WindowsPowerShell\v1.0, which is in the default
  // PATH on Windows.
  private static final String POWERSHELL_BIN = "powershell.exe";
  private static final String[] POWERSHELL_SETUP_COMMANDS = {
      // 1. Use Set-ExecutionPolicy to allow users to run scripts unsigned.
      "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned",
      // 2. Set $errorActionPreference to Stop so that we exit immediately if a command fails.
      //    This ensures the command doesn't succeed with wrong result.
      "$errorActionPreference='Stop'",
      // 3. Change the default encoding to utf-8, by default it was utf-16.
      //    https://stackoverflow.com/questions/40098771
      "$PSDefaultParameterValues['*:Encoding'] = 'utf8'",
  };
  private String scriptPostFix;

  PowershellCommandConstructor(String scriptPostFix) {
    this.scriptPostFix = scriptPostFix;
  }

  @Override
  public ImmutableList<String> buildCommandLineSimpleArgv(String command) {
    return ImmutableList.of(POWERSHELL_BIN, "/c", String.join("; ", POWERSHELL_SETUP_COMMANDS) + "; " + command);
  }

  @Override
  public Artifact buildCommandLineArtifact(RuleContext ruleContext, String command) {
    String scriptFileName = ruleContext.getTarget().getName() + scriptPostFix;
    return FileWriteAction.createFile(
        ruleContext, scriptFileName, command, /*executable=*/true);
  }

  @Override
  public ImmutableList<String> buildCommandLineArgvWithArtifact(Artifact scriptFileArtifact) {
    // Powershell doesn't search for the current directory by default, we need to always add
    // ".\" prefix to a relative path.
    return buildCommandLineSimpleArgv(".\\" + scriptFileArtifact.getExecPathString().replace('/', '\\'));
  }
}
