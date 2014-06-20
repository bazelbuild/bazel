// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.util.ExitCausingException;
import com.google.devtools.build.lib.util.ExitCode;

/**
 * An exception thrown by {@link BuildTool} when a target is determined to be
 * invalid.
 */
public class InvalidTargetException extends ExitCausingException {

  public InvalidTargetException(String message) {
    super(message, ExitCode.COMMAND_LINE_ERROR);
  }
}
