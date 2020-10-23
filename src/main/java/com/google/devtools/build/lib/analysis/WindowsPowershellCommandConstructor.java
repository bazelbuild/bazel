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

/** The class for constructing command line for Powershell on Windows. */
public final class WindowsPowershellCommandConstructor implements CommandConstructor {
  private static final String POWERSHELL_SETUP_COMMANDS =
      // 1. Use Set-ExecutionPolicy to allow users to run scripts unsigned.
      "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned; "
          // 2. Set $errorActionPreference to Stop so that we exit immediately if a CmdLet
          // fails,
          //    but when an external command fails, it will still continue.
          + "$errorActionPreference='Stop'; "
          // 3. Change the default encoding to utf-8, by default it was utf-16.
          //    https://stackoverflow.com/questions/40098771
          + "$PSDefaultParameterValues['*:Encoding'] = 'utf8'; ";
  private final String scriptNameSuffix;

  WindowsPowershellCommandConstructor(String scriptNameSuffix) {
    this.scriptNameSuffix = scriptNameSuffix;
  }

  @Override
  public ImmutableList<String> asExecArgv(String command) {
    // `powershell.exe` exists at C:\Windows\System32\WindowsPowerShell\v1.0,
    // which is in the default PATH on Windows.
    // We currently don't support Powershell on Linux/macOS, although Powershell is available on
    // those platforms.
    return ImmutableList.of("powershell.exe", "/c", POWERSHELL_SETUP_COMMANDS + command);
  }

  @Override
  public ImmutableList<String> asExecArgv(Artifact scriptFileArtifact) {
    // Powershell doesn't search for the current directory by default, we need to always add
    // ".\" prefix to a relative path.
    return this.asExecArgv(".\\" + scriptFileArtifact.getExecPathString().replace('/', '\\'));
  }

  @Override
  public Artifact commandAsScript(RuleContext ruleContext, String command) {
    String scriptFileName = ruleContext.getTarget().getName() + scriptNameSuffix;
    return FileWriteAction.createFile(ruleContext, scriptFileName, command, /*executable=*/ true);
  }
}
