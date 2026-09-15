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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import net.starlark.java.syntax.Location;

/**
 * This exception is thrown during a build when an input file is missing, but the file is not the
 * input to any action being executed.
 *
 * <p>If a missing input file is an input to an action, an {@link ActionExecutionException} is
 * thrown instead.
 */
public class MissingInputFileException extends Exception implements DetailedException {
  private final DetailedExitCode detailedExitCode;
  private final Location location;

  public MissingInputFileException(FailureDetail failureDetail, Location location) {
    super(failureDetail.getMessage());
    this.detailedExitCode = DetailedExitCode.of(failureDetail);
    this.location = location;
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  /**
   * Return a location where this input file is referenced. If there are multiple such locations,
   * one is chosen arbitrarily. If there are none, return null.
   */
  public Location getLocation() {
    return location;
  }
}
