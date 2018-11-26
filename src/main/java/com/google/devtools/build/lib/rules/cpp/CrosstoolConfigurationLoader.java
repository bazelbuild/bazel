// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import com.google.protobuf.UninitializedMessageException;
import java.io.IOException;
import java.util.concurrent.ExecutionException;

/**
 * A loader that reads Crosstool configuration files and creates CToolchain
 * instances from them.
 *
 * <p>Note that this class contains a cache for the text format -> proto objects mapping of
 * Crosstool protos that is completely independent from Skyframe or anything else. This should be
 * done in a saner way.
 */
public class CrosstoolConfigurationLoader {
  static final String CROSSTOOL_CONFIGURATION_FILENAME = "CROSSTOOL";

  /**
   * Cache for storing result of toReleaseConfiguration function based on path and md5 sum of
   * input file. We can use md5 because result of this function depends only on the file content.
   */
  private static final Cache<String, CrosstoolRelease> crosstoolReleaseCache =
      CacheBuilder.newBuilder().concurrencyLevel(4).maximumSize(100).build();

  private CrosstoolConfigurationLoader() {}

  @FunctionalInterface
  interface CrosstoolDataFunction {
    String apply() throws IOException;
  }

  /**
   * Reads the given <code>data</code> String, which must be in ascii format, into a protocol
   * buffer. It uses the <code>name</code> parameter for error messages.
   *
   * @param name for the error messages
   * @param dataFunction returns data in the text proto format
   * @param digestOrNull to be used or null; will be computed from data when null
   * @throws IOException if the parsing failed
   */
  @VisibleForTesting
  public static CrosstoolRelease toReleaseConfiguration(
      String name, CrosstoolDataFunction dataFunction, String digestOrNull)
      throws InvalidConfigurationException {
    if (digestOrNull != null) {
      // We were given the digest, let's first consult the cache before reading the data, saving
      // not only proto parsing time, but also IO in case of cache hit.
      return getCachedReleaseConfiguration(name, dataFunction, digestOrNull);
    } else {
      // We were not given the digest, we have to read the data to compute the digest. We will
      // still save the proto parsing time in case of the cache hit.
      try {
        String data = dataFunction.apply();
        String digest = computeCrosstoolDigest(data);
        return getCachedReleaseConfiguration(name, () -> data, digest);
      } catch (IOException e) {
        throw new InvalidConfigurationException(e);
      }
    }
  }

  private static CrosstoolRelease getCachedReleaseConfiguration(
      String name, CrosstoolDataFunction dataFunction, String digest)
      throws InvalidConfigurationException {
    try {
      return crosstoolReleaseCache.get(
          digest, () -> getUncachedReleaseConfiguration(name, dataFunction));
    } catch (ExecutionException e) {
      throw new InvalidConfigurationException(e.getCause().getMessage());
    }
  }

  @VisibleForTesting
  static CrosstoolRelease getUncachedReleaseConfiguration(
      String name, CrosstoolDataFunction dataFunction)
      throws IOException, InvalidConfigurationException {
    CrosstoolRelease.Builder builder = CrosstoolRelease.newBuilder();
    try {
      TextFormat.merge(dataFunction.apply(), builder);
      return builder.build();
    } catch (ParseException e) {
      throw new InvalidConfigurationException(
          "Could not read the crosstool configuration file '"
              + name
              + "', "
              + "because of a parser error ("
              + e.getMessage()
              + ")");
    } catch (UninitializedMessageException e) {
      throw new InvalidConfigurationException(
          "Could not read the crosstool configuration file '"
              + name
              + "', "
              + "because of an incomplete protocol buffer ("
              + e.getMessage()
              + ")");
    }
  }

  private static String computeCrosstoolDigest(String data) {
    return BaseEncoding.base16()
        .lowerCase()
        .encode(new Fingerprint().addBytes(data.getBytes(UTF_8)).digestAndReset());
  }
}
