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

package com.google.devtools.build.lib.rules.cpp;

import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Executor.ActionContext;
import com.google.devtools.build.lib.rules.cpp.IncludeParser.Inclusion;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.util.Collection;

/** Parses a single file for its (direct) includes, possibly using a remote service. */
public interface RemoteIncludeExtractor extends ActionContext {
  /** Result of checking if this object should be used to parse a given file. */
  interface RemoteParseData {
    boolean shouldParseRemotely();
  }

  /**
   * Returns whether to use this object to parse the given file for includes. The returned data
   * should be passed to {@link #extractInclusions} to direct its behavior.
   */
  RemoteParseData shouldParseRemotely(Path file);

  /**
   * Extracts all inclusions from a given source file, possibly using a remote service.
   *
   * @param file the file from which to parse and extract inclusions.
   * @param actionExecutionContext services in the scope of the action. Like the Err/Out stream
   *                               outputs.
   * @param remoteParseData the returned value of {@link #shouldParseRemotely}.
   * @return a collection of inclusions, normalized to the cache
   */
  public Collection<Inclusion> extractInclusions(Artifact file,
      ActionExecutionContext actionExecutionContext, RemoteParseData remoteParseData)
  throws IOException, InterruptedException;

}
