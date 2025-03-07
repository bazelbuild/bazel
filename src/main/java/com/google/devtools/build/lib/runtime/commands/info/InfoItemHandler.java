// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.util.AbruptExitException;
import java.io.IOException;

/** Handles how {@link InfoItem}s are outputted. */
public interface InfoItemHandler extends AutoCloseable {
  /** Stores {@link InfoItem} information. */
  void addInfoItem(String key, byte[] value)
      throws AbruptExitException, InterruptedException, IOException;

  /** Flushes any internal state. */
  @Override
  void close() throws IOException;

  /** Defines the way to output {@link InfoItem} information. */
  enum InfoItemOutputType {
    /**
     * Info information is directly printed to the console. {@link StdoutInfoItemHandler} is created
     * if this type is passed in.
     */
    STDOUT,

    /**
     * Info information is packed in response extensions for downstream service processing. {@link
     * RemoteRequestedInfoItemHandler} is created if this type is passed in.
     */
    RESPONSE_PROTO
  }

  /** Contains method to create the correct type of {@link InfoItemHandler}. */
  interface InfoItemHandlerFactory {
    InfoItemHandler create(
        CommandEnvironment env, InfoItemOutputType infoItemOutputType, boolean printKeys);
  }

  /**
   * Implementation of {@link InfoItemHandlerFactory} that creates {@link InfoItemHandler} instances
   * based on the provided {@link InfoItemOutputType}.
   */
  class InfoItemHandlerFactoryImpl implements InfoItemHandlerFactory {
    @Override
    public InfoItemHandler create(
        CommandEnvironment env, InfoItemOutputType infoItemOutputType, boolean printKeys) {
      return switch (infoItemOutputType) {
        case STDOUT -> new StdoutInfoItemHandler(env.getReporterOutErr(), printKeys);
        case RESPONSE_PROTO -> new RemoteRequestedInfoItemHandler(env, printKeys);
      };
    }
  }
}
