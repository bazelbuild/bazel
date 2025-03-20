// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.repository;

import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;

/** Reports the prgoress of a repository fetch. */
public class RepositoryFetchProgress implements FetchProgress {
  private final RepositoryName repoName;
  private final boolean finished;
  private final String message;

  /** Returns the unique identifying string for a repository fetching event. */
  public static String repositoryFetchContextString(RepositoryName repoName) {
    return "repository " + repoName;
  }

  public static RepositoryFetchProgress ongoing(RepositoryName repoName, String message) {
    return new RepositoryFetchProgress(repoName, /*finished=*/ false, message);
  }

  public static RepositoryFetchProgress finished(RepositoryName repoName) {
    return new RepositoryFetchProgress(repoName, /*finished=*/ true, "finished.");
  }

  private RepositoryFetchProgress(RepositoryName repoName, boolean finished, String message) {
    this.repoName = repoName;
    this.finished = finished;
    this.message = message;
  }

  @Override
  public String getResourceIdentifier() {
    return repositoryFetchContextString(repoName);
  }

  @Override
  public String getProgress() {
    return message;
  }

  @Override
  public boolean isFinished() {
    return finished;
  }
}
