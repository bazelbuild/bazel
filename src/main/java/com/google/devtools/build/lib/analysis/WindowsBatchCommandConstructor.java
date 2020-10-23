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

/** The class for constructing command line for Batch on Windows. */
public final class WindowsBatchCommandConstructor implements CommandConstructor {
  private final String scriptNameSuffix;

  WindowsBatchCommandConstructor(String scriptNameSuffix) {
    this.scriptNameSuffix = scriptNameSuffix;
  }

  @Override
  public ImmutableList<String> asExecArgv(String command) {
    // `cmd.exe` exists at C:\Windows\System32, which is in the default PATH on Windows.
    return ImmutableList.of(
        "cmd.exe", "/S", // strip first and last quotes and execute everything else as is.
        "/E:ON", // enable extended command set.
        "/V:ON", // enable delayed variable expansion
        "/D", // ignore AutoRun registry entries.
        "/c", // execute command. This must be the last option before the command itself.
        command);
  }

  @Override
  public ImmutableList<String> asExecArgv(Artifact scriptFileArtifact) {
    return this.asExecArgv(scriptFileArtifact.getExecPathString().replace('/', '\\'));
  }

  @Override
  public Artifact commandAsScript(RuleContext ruleContext, String command) {
    String scriptFileName = ruleContext.getTarget().getName() + this.scriptNameSuffix;
    String scriptFileContents = "@echo off\n" + command;
    return FileWriteAction.createFile(
        ruleContext, scriptFileName, scriptFileContents, /*executable=*/ true);
  }
}
