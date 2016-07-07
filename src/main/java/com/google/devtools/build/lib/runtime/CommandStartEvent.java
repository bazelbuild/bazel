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

package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.vfs.Path;

import java.util.Map;
import java.util.UUID;

/**
 * This event is fired when the Blaze command is started (clean, build, test,
 * etc.).
 */
public class CommandStartEvent extends CommandEvent {
  private final String commandName;
  private final UUID commandId;
  private final Map<String, String> clientEnv;
  private final Path workingDirectory;
  private final long waitTimeInMs;
  private final BlazeDirectories directories;

  /**
   * @param commandName the name of the command
   */
  public CommandStartEvent(String commandName, UUID commandId, Map<String, String> clientEnv,
      Path workingDirectory, BlazeDirectories directories, long waitTimeInMs) {
    this.commandName = commandName;
    this.commandId = commandId;
    this.clientEnv = clientEnv;
    this.workingDirectory = workingDirectory;
    this.directories = directories;
    this.waitTimeInMs = waitTimeInMs;
  }

  public String getCommandName() {
    return commandName;
  }

  public UUID getCommandId() {
    return commandId;
  }

  public Map<String, String> getClientEnv() {
    return clientEnv;
  }

  public Path getWorkingDirectory() {
    return workingDirectory;
  }

  public BlazeDirectories getBlazeDirectories() {
    return directories;
  }

  public long getWaitTimeInMs() {
    return waitTimeInMs;
  }
}
