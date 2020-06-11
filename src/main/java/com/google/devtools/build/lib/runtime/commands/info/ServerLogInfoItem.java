// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands.info;

import com.google.common.base.Supplier;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DebugLoggerConfigurator;
import java.io.IOException;

/** Info item for server_log path. */
public class ServerLogInfoItem extends InfoItem {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Constructs an info item for the server log path.
   *
   * @param productName name of the tool whose server log path will be queried
   */
  public ServerLogInfoItem(String productName) {
    super("server_log", productName + " server log path", false);
  }

  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env)
      throws AbruptExitException {
    try {
      return print(DebugLoggerConfigurator.getServerLogPath().orElse(""));
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Failed to determine server log location");
      return print("UNKNOWN LOG LOCATION");
    }
  }
}
