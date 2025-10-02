// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import java.time.Duration;
import java.util.Collection;
import java.util.Map;
import java.util.Set;

/**
 * Stores the parameters needed for Skycache's metadata insertion and queries about builds stored in
 * the remote analysis cache. Not every parameter is obtained at the same point during the build.
 *
 * <p>Metadata queries help UX by printing ahead of the build whether there will be any cache hits
 * or if there could potentially be cache hits by syncing to a different evaluating version or using
 * a different build configuration.
 */
public interface SkycacheMetadataParams {

  /**
   * TODO: b/425247333 - Metadata insertions and queries should be very fast. 5 seconds is already
   * very generous. As part of normal workflows we expect subsecond operations. We can re-evaluate
   * this in production when we have more data.
   */
  Duration TIMEOUT = Duration.ofSeconds(5);

  void init(
      long clNumber,
      String bazelVersion,
      Collection<String> targets,
      boolean useFakeStampData,
      Map<String, String> userOptions,
      Set<String> projectSclOptions);

  /**
   * Using the user options map (setUserOptionsMap must have been called before this method) and
   * using a set of all the existing options that affect the configuration checksum used by
   * Skycache, we compute the top level flags (contracted as opposed to flags expanded from their
   * configs) in order to print a diff of the flags vs cached writer builds for the user during
   * reader builds if there was a config mismatch causing Skycache misses.
   */
  void setOriginalConfigurationOptions(Set<String> configOptions);

  void setConfigurationHash(String configurationHash);

  /** The evaluating version, i.e.: changelist, commit, etc.. */
  long getEvaluatingVersion();

  /** Top level build config checksum for the build. */
  String getConfigurationHash();

  /** The Bazel version which consists of the release name + the md5 install hash. */
  String getBazelVersion();

  /** Geographical area of the build. */
  String getArea();

  /**
   * Whether real stamp data is used for the build. This flag is not considered for the top level
   * configuration checksum.
   */
  boolean getUseFakeStampData();

  /** The invocation targets */
  Collection<String> getTargets();

  /** The user flags passed in this invocation */
  Collection<String> getConfigFlags();
}
