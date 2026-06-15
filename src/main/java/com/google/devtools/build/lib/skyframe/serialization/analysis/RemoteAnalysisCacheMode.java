// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;

/** The transport direction for the remote analysis cache. */
@SkybridgeInterface
public enum RemoteAnalysisCacheMode {
  /** Serializes and uploads Skyframe analysis nodes after the build command finishes. */
  UPLOAD,

  /**
   * Dumps the manifest of SkyKeys computed in the frontier and the active set. This mode does not
   * serialize and upload the keys.
   */
  DUMP_UPLOAD_MANIFEST_ONLY,

  /** Fetches and deserializes the Skyframe analysis nodes during the build. */
  DOWNLOAD,

  /** Disabled. */
  OFF;

  /** Returns true if the selected mode needs to connect to a backend. */
  public boolean requiresBackendConnectivity() {
    return switch (this) {
      case UPLOAD, DOWNLOAD -> true;
      case DUMP_UPLOAD_MANIFEST_ONLY, OFF -> false;
    };
  }

  public boolean isRetrievalEnabled() {
    return this == DOWNLOAD;
  }

  /**
   * Returns true if the mode serializes <i>values</i>.
   *
   * <p>{@link DOWNLOAD} serializes keys, but not values.
   */
  public boolean serializesValues() {
    return switch (this) {
      case UPLOAD, DUMP_UPLOAD_MANIFEST_ONLY -> true;
      case DOWNLOAD, OFF -> false;
    };
  }
}
