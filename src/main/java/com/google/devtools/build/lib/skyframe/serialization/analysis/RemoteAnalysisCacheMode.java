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
import java.util.List;

/** The transport direction for the remote analysis cache. */
@SkybridgeInterface
public final class RemoteAnalysisCacheMode {
  /** Serializes and uploads Skyframe analysis nodes after the build command finishes. */
  public static final RemoteAnalysisCacheMode UPLOAD = new RemoteAnalysisCacheMode("UPLOAD");

  /**
   * Dumps the manifest of SkyKeys computed in the frontier and the active set. This mode does not
   * serialize and upload the keys.
   */
  public static final RemoteAnalysisCacheMode DUMP_UPLOAD_MANIFEST_ONLY =
      new RemoteAnalysisCacheMode("DUMP_UPLOAD_MANIFEST_ONLY");

  /** Fetches and deserializes the Skyframe analysis nodes during the build. */
  public static final RemoteAnalysisCacheMode DOWNLOAD = new RemoteAnalysisCacheMode("DOWNLOAD");

  /** Both fetches and serializes Skyframe analysis nodes during the build. */
  public static final RemoteAnalysisCacheMode ASYNC_BIDI =
      new RemoteAnalysisCacheMode("ASYNC_BIDI");

  /** Serializes and uploads Skyframe analysis nodes during the build (no download). */
  public static final RemoteAnalysisCacheMode ASYNC_UPLOAD =
      new RemoteAnalysisCacheMode("ASYNC_UPLOAD");

  /**
   * Fetches analysis nodes during the build and serializes/uploads analysis nodes after the build
   * command finishes.
   */
  public static final RemoteAnalysisCacheMode BIDI = new RemoteAnalysisCacheMode("BIDI");

  /** Disabled. */
  public static final RemoteAnalysisCacheMode OFF = new RemoteAnalysisCacheMode("OFF");

  private final String name;

  private RemoteAnalysisCacheMode(String name) {
    this.name = name;
  }

  /** Returns true if the selected mode needs to connect to a backend. */
  public boolean requiresBackendConnectivity() {
    return isUploadEnabled() || isRetrievalEnabled();
  }

  public boolean isRetrievalEnabled() {
    return this == DOWNLOAD || this == ASYNC_BIDI || this == BIDI;
  }

  public boolean isUploadEnabled() {
    return isSyncUpload() || isAsyncUpload();
  }

  public boolean isBidirectional() {
    return isRetrievalEnabled() && isUploadEnabled();
  }

  public boolean isSyncUpload() {
    return this == UPLOAD || this == BIDI;
  }

  public boolean isAsyncUpload() {
    return this == ASYNC_BIDI || this == ASYNC_UPLOAD;
  }

  /**
   * Returns true if the mode serializes <i>values</i>.
   *
   * <p>{@link DOWNLOAD} serializes keys, but not values.
   */
  public boolean serializesValues() {
    if (isUploadEnabled()) {
      return true;
    }

    return this == DUMP_UPLOAD_MANIFEST_ONLY;
  }

  @Override
  public String toString() {
    return name;
  }

  @SuppressWarnings("JdkImmutableCollections")
  private static final List<RemoteAnalysisCacheMode> values =
      List.of(UPLOAD, DUMP_UPLOAD_MANIFEST_ONLY, DOWNLOAD, ASYNC_BIDI, ASYNC_UPLOAD, BIDI, OFF);

  public static List<RemoteAnalysisCacheMode> values() {
    return values;
  }
}
