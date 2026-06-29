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

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.FingerprintValueService;
import com.google.devtools.build.lib.skyframe.serialization.FrontierNodeVersion;
import com.google.devtools.build.lib.skyframe.serialization.KeyValueWriter;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** Various bits of data and functionality serialization needs. */
public interface SerializationDependenciesProvider {
  RemoteAnalysisCacheMode mode();

  /**
   * Returns the string distinguisher to invalidate SkyValues, in addition to the corresponding
   * SkyKey.
   */
  FrontierNodeVersion getSkyValueVersion() throws InterruptedException;

  /**
   * Returns the {@link ObjectCodecs} supplier for remote analysis caching.
   *
   * <p>Calling this can be an expensive process as the codec registry will be initialized.
   */
  ObjectCodecs getObjectCodecs() throws InterruptedException;

  /** Returns the {@link FingerprintValueService} implementation. */
  FingerprintValueService getFingerprintValueService() throws InterruptedException;

  String getSerializedFrontierProfile();

  Optional<Predicate<PackageIdentifier>> getActiveDirectoriesMatcher();

  /** Returns the destination for file invalidation data when uploading. */
  @Nullable
  KeyValueWriter getFileInvalidationWriter() throws InterruptedException;

  @Nullable
  RemoteAnalysisMetadataWriter getMetadataWriter() throws InterruptedException;

  void waitForUploadCompletion() throws InterruptedException, ExecutionException;

  boolean shouldMinimizeMemory();

  boolean getSkycacheAnalysisOnly();

  boolean getEmitUploadedEvents();
}
