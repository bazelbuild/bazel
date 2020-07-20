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
package com.google.devtools.build.lib.buildtool;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import java.io.IOException;

/** For exceptions that arise from the post-execution action graph dump. */
public class PostExecutionActionGraphDumpException extends Exception {
  private final Exception rootCause;

  PostExecutionActionGraphDumpException(Exception rootCause) {
    Preconditions.checkArgument(
        rootCause instanceof CommandLineExpansionException || rootCause instanceof IOException,
        "Unexpected exception type: %s",
        rootCause);
    this.rootCause = rootCause;
  }

  public Exception getRootCause() {
    return rootCause;
  }

  @Override
  public String getMessage() {
    return rootCause.getMessage();
  }
}
