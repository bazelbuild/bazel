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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.util.StringUtilities;

import java.util.Map;
import java.util.logging.Logger;

/**
 * Determines the version information of the current process.
 *
 * <p>The version information is a dictionary mapping from string keys to string values.  For
 * build stamping, it should have the key "Build label", which contains among others a
 * XXXXXXX-YYYY.MM.DD string to indicate the version of the release.  If no data is available
 * (eg. when running non-released version), {@link #isAvailable()} returns false.
 */
public class BlazeVersionInfo {
  public static final String BUILD_LABEL = "Build label";
  
  private final Map<String, String> buildData = Maps.newTreeMap();
  private static BlazeVersionInfo instance = null;

  private static final Logger LOG = Logger.getLogger(BlazeVersionInfo.class.getName());

  public BlazeVersionInfo(Map<String, String> info) {
    buildData.putAll(info);
  }

  /**
   * Accessor method for BlazeVersionInfo singleton.
   *
   * <p>If setBuildInfo was not called, returns an empty BlazeVersionInfo instance, which should
   * not be persisted.
   */
  public static synchronized BlazeVersionInfo instance() {
    if (instance == null) {
      return new BlazeVersionInfo(ImmutableMap.<String, String>of());
    }
    return instance;
  }

  private static void logVersionInfo(BlazeVersionInfo info) {
    if (info.getSummary() == null) {
      LOG.warning("Blaze release version information not available");
    } else {
      LOG.info("Blaze version info: " + info.getSummary());
    }
  }

  /**
   * Sets build info.
   *
   * <p>This should be called once in the program execution, as early soon as possible, so we
   * can have the version information even before modules are initialized.
   */
  public static synchronized void setBuildInfo(Map<String, String> info) {
    if (instance != null) {
      throw new IllegalStateException("setBuildInfo called twice.");
    }
    instance = new BlazeVersionInfo(info);
    logVersionInfo(instance);
  }

  /**
   * Indicates whether version information is available.
   */
  public boolean isAvailable() {
    return !buildData.isEmpty();
  }

  /**
   * Returns the summary which gets displayed in the 'version' command.
   * The summary is a list of formatted key / value pairs.
   */
  public String getSummary() {
    if (buildData.isEmpty()) {
      return null;
    }
    return StringUtilities.layoutTable(buildData);
  }

  /**
   * Returns true iff this binary is released--that is, a
   * binary built with a release label.
   */
  public boolean isReleasedBlaze() {
    String buildLabel = buildData.get(BUILD_LABEL);
    return buildLabel != null && buildLabel.length() > 0;
  }

  /**
   * Returns the release label, if any, or "development version".
   */
  public String getReleaseName() {
    String buildLabel = buildData.get(BUILD_LABEL);
    return (buildLabel != null && buildLabel.length() > 0)
        ? "release " + buildLabel
        : "development version";
  }
}
