// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertWithMessage;
import static java.util.Arrays.stream;
import static java.util.stream.Collectors.joining;

import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;

/** Mixin helpers for Skycache integration tests. */
interface SkycacheIntegrationTestHelpers {
  static final String UPLOAD_MODE_OPTION = "--experimental_remote_analysis_cache_mode=upload";
  static final String DOWNLOAD_MODE_OPTION = "--experimental_remote_analysis_cache_mode=download";
  static final String DUMP_MANIFEST_MODE_OPTION =
      "--experimental_remote_analysis_cache_mode=dump_upload_manifest_only";

  void addOptions(String... args);

  @CanIgnoreReturnValue
  Path write(String relativePath, String... lines) throws IOException;

  @CanIgnoreReturnValue
  BuildResult buildTarget(String... targets) throws Exception;

  CommandEnvironment getCommandEnvironment();

  default void assertUploadSuccess(String... targets) throws Exception {
    addOptions(UPLOAD_MODE_OPTION);
    buildTarget(targets);
    assertWithMessage("expected to serialize at least one Skyframe node")
        .that(getCommandEnvironment().getRemoteAnalysisCachingEventListener().getSerializedKeys())
        .isNotEmpty();
    assertWithMessage("expected to not have any SerializationExceptions")
        .that(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializationExceptionCounts())
        .isEqualTo(0);
  }

  default void assertDownloadSuccess(String... targets) throws Exception {
    addOptions(DOWNLOAD_MODE_OPTION);
    buildTarget(targets);
    assertWithMessage("expected to deserialize at least one Skyframe node")
        .that(getCommandEnvironment().getRemoteAnalysisCachingEventListener().getCacheHits())
        .isNotEmpty();
    assertWithMessage("expected to not have any SerializationExceptions")
        .that(
            getCommandEnvironment()
                .getRemoteAnalysisCachingEventListener()
                .getSerializationExceptionCounts())
        .isEqualTo(0);
  }

  default void writeProjectSclWithActiveDirs(String path, String... activeDirs) throws IOException {
    String activeDirsString = stream(activeDirs).map(s -> "\"" + s + "\"").collect(joining(", "));
    write(
        path + "/PROJECT.scl",
        String.format(
            "project = { \"active_directories\": { \"default\": [%s] } }", activeDirsString));
  }

  default void writeProjectSclWithActiveDirs(String path) throws IOException {
    // Overload for the common case where the path is the only active directory.
    writeProjectSclWithActiveDirs(path, path);
  }
}
