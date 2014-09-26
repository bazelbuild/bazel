/*
 * Copyright 2014-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple.xcode.xcodeproj;

import com.dd.plist.NSArray;
import com.dd.plist.NSString;
import com.facebook.buck.apple.xcode.XcodeprojSerializer;
import com.google.common.collect.Lists;

import java.util.List;

import javax.annotation.Nullable;

/**
 * Build phase which represents running a shell script.
 */
public class PBXShellScriptBuildPhase extends PBXBuildPhase {
  private List<String> inputPaths;
  private List<String> outputPaths;
  @Nullable private String shellPath;
  @Nullable private String shellScript;

  private static final NSString DEFAULT_SHELL_PATH = new NSString("/bin/sh");
  private static final NSString DEFAULT_SHELL_SCRIPT = new NSString("");

  public PBXShellScriptBuildPhase() {
    this.inputPaths = Lists.newArrayList();
    this.outputPaths = Lists.newArrayList();
  }

  @Override
  public String isa() {
    return "PBXShellScriptBuildPhase";
  }

  /**
   * Returns the list (possibly empty) of files passed as input to the shell script.
   * May not be actual paths, because they can have variable interpolations.
   */
  public List<String> getInputPaths() {
    return inputPaths;
  }

  /**
   * Returns the list (possibly empty) of files created as output of the shell script.
   * May not be actual paths, because they can have variable interpolations.
   */
  public List<String> getOutputPaths() {
    return outputPaths;
  }

  /**
   * Returns the path to the shell under which the script is to be executed.
   * Defaults to "/bin/sh".
   */
  @Nullable
  public String getShellPath() {
    return shellPath;
  }

  /**
   * Sets the path to the shell under which the script is to be executed.
   */
  public void setShellPath(String shellPath) {
    this.shellPath = shellPath;
  }

  /**
   * Gets the contents of the shell script to execute under the shell
   * returned by {@link #getShellPath()}.
   */
  @Nullable
  public String getShellScript() {
    return shellScript;
  }

  /**
   * Sets the contents of the script to execute.
   */
  public void setShellScript(String shellScript) {
    this.shellScript = shellScript;
  }

  @Override
  public void serializeInto(XcodeprojSerializer s) {
    super.serializeInto(s);

    NSArray inputPathsArray = new NSArray(inputPaths.size());
    for (int i = 0; i < inputPaths.size(); i++) {
      inputPathsArray.setValue(i, new NSString(inputPaths.get(i)));
    }
    s.addField("inputPaths", inputPathsArray);

    NSArray outputPathsArray = new NSArray(outputPaths.size());
    for (int i = 0; i < outputPaths.size(); i++) {
      outputPathsArray.setValue(i, new NSString(outputPaths.get(i)));
    }
    s.addField("outputPaths", outputPathsArray);

    NSString shellPathString;
    if (shellPath == null) {
      shellPathString = DEFAULT_SHELL_PATH;
    } else {
      shellPathString = new NSString(shellPath);
    }
    s.addField("shellPath", shellPathString);

    NSString shellScriptString;
    if (shellScript == null) {
      shellScriptString = DEFAULT_SHELL_SCRIPT;
    } else {
      shellScriptString = new NSString(shellScript);
    }
    s.addField("shellScript", shellScriptString);
  }
}
