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
package com.google.devtools.build.lib.buildeventservice;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.runtime.BlazeModule.ModuleEnvironment;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import javax.annotation.Nullable;

/**
 * Function used by a {@link BuildEventTransport} to report errors using the provided {@link
 * EventHandler} and maybe exit abruptly.
 */
@FunctionalInterface
public interface ExitFunction {
  void accept(String message, Throwable cause, ExitCode code);

  // TODO(lpino): The error handling should in the BES module but we can't do that yet because
  // we use the exit function inside the {@link BuildEventServiceUploader}.
  static ExitFunction standardExitFunction(
      EventHandler commandLineReporter,
      ModuleEnvironment moduleEnvironment,
      @Nullable String besResultsUrl,
      boolean errorsFailBuild) {
    return (String message, Throwable cause, ExitCode exitCode) -> {
      if (exitCode == ExitCode.SUCCESS) {
        Preconditions.checkState(cause == null, cause);
        commandLineReporter.handle(Event.info("Build Event Protocol upload finished successfully"));
        if (besResultsUrl != null) {
          commandLineReporter.handle(
              Event.info("Build Event Protocol results available at " + besResultsUrl));
        }
      } else {
        Preconditions.checkState(cause != null, cause);
        if (errorsFailBuild) {
          commandLineReporter.handle(Event.error(message));
          moduleEnvironment.exit(new AbruptExitException(exitCode, cause));
        } else {
          commandLineReporter.handle(Event.warn(message));
        }
        if (besResultsUrl != null) {
          if (!Strings.isNullOrEmpty(besResultsUrl)) {
            commandLineReporter.handle(
                Event.info(
                    "Partial Build Event Protocol results may be available at " + besResultsUrl));
          }
        }
      }
    };
  }
}
