// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;

/** An interface used to check whether Bazel should trust a remote artifact. */
public interface RemoteArtifactChecker {
  RemoteArtifactChecker TRUST_ALL = (file, metadata) -> true;
  RemoteArtifactChecker IGNORE_ALL = (file, metadata) -> false;

  /**
   * Returns true if Bazel should trust (and not verify) build artifacts that were last seen
   * remotely and do not exist locally.
   */
  boolean shouldTrustRemoteArtifact(ActionInput file, RemoteFileArtifactValue metadata);
}
