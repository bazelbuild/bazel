// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;

/** Event reporting on progress mading fetching a remote git repository. */
public class GitFetchProgress implements FetchProgress {
  private final String remote;
  private final String message;
  private final boolean isFinished;

  GitFetchProgress(String remote, String message, boolean isFinished) {
    this.remote = remote;
    this.message = message;
    this.isFinished = isFinished;
  }

  GitFetchProgress(String remote, String message) {
    this(remote, message, false);
  }

  @Override
  public String getResourceIdentifier() {
    return remote;
  }

  @Override
  public String getProgress() {
    return message;
  }

  @Override
  public boolean isFinished() {
    return isFinished;
  }
}
