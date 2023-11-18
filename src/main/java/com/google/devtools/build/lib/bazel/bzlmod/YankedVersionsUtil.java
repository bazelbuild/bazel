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
package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

/** Utility class to parse and evaluate yanked version specifications and exceptions. */
public final class YankedVersionsUtil {

  public static final PrecomputedValue.Precomputed<List<String>> ALLOWED_YANKED_VERSIONS =
      new PrecomputedValue.Precomputed<>("allowed_yanked_versions");
  public static final String BZLMOD_ALLOWED_YANKED_VERSIONS_ENV = "BZLMOD_ALLOW_YANKED_VERSIONS";

  /**
   * Parse a set of allowed yanked version from command line flag (--allowed_yanked_versions) and
   * environment variable (ALLOWED_YANKED_VERSIONS). If `all` is specified, return Optional.empty();
   * otherwise returns the set of parsed modulel key.
   */
  static Optional<ImmutableSet<ModuleKey>> parseAllowedYankedVersions(
      String allowedYankedVersionsFromEnv, List<String> allowedYankedVersionsFromFlag)
      throws ExternalDepsException {
    ImmutableSet.Builder<ModuleKey> allowedYankedVersionBuilder = new ImmutableSet.Builder<>();
    if (allowedYankedVersionsFromEnv != null) {
      if (parseModuleKeysFromString(
          allowedYankedVersionsFromEnv,
          allowedYankedVersionBuilder,
          String.format(
              "environment variable %s=%s",
              BZLMOD_ALLOWED_YANKED_VERSIONS_ENV, allowedYankedVersionsFromEnv))) {
        return Optional.empty();
      }
    }
    for (String allowedYankedVersions : allowedYankedVersionsFromFlag) {
      if (parseModuleKeysFromString(
          allowedYankedVersions,
          allowedYankedVersionBuilder,
          String.format("command line flag --allow_yanked_versions=%s", allowedYankedVersions))) {
        return Optional.empty();
      }
    }
    return Optional.of(allowedYankedVersionBuilder.build());
  }

  /**
   * Returns the reason for the given module being yanked, or {@code Optional.empty()} if the module
   * is not yanked or explicitly allowed despite being yanked.
   */
  static Optional<String> getYankedInfo(
      Registry registry,
      ModuleKey key,
      Optional<ImmutableSet<ModuleKey>> allowedYankedVersions,
      ExtendedEventHandler eventHandler)
      throws InterruptedException {
    Optional<ImmutableMap<Version, String>> yankedVersions;
    try {
      yankedVersions = registry.getYankedVersions(key.getName(), eventHandler);
    } catch (IOException e) {
      eventHandler.handle(
          Event.warn(
              String.format(
                  "Could not read metadata file for module %s: %s", key, e.getMessage())));
      // This is failing open: If we can't read the metadata file, we allow yanked modules to be
      // fetched.
      return Optional.empty();
    }
    if (yankedVersions.isEmpty()) {
      return Optional.empty();
    }
    String yankedInfo = yankedVersions.get().get(key.getVersion());
    if (yankedInfo != null
        && allowedYankedVersions.isPresent()
        && !allowedYankedVersions.get().contains(key)) {
      return Optional.of(yankedInfo);
    } else {
      return Optional.empty();
    }
  }

  /**
   * Parse of a comma-separated list of module version(s) of the form '<module name>@<version>' or
   * 'all' from the string. Returns true if 'all' is present, otherwise returns false.
   */
  private static boolean parseModuleKeysFromString(
      String input, ImmutableSet.Builder<ModuleKey> allowedYankedVersionBuilder, String context)
      throws ExternalDepsException {
    ImmutableList<String> moduleStrs = ImmutableList.copyOf(Splitter.on(',').split(input));

    for (String moduleStr : moduleStrs) {
      if (moduleStr.equals("all")) {
        return true;
      }

      if (moduleStr.isEmpty()) {
        continue;
      }

      String[] pieces = moduleStr.split("@", 2);

      if (pieces.length != 2) {
        throw ExternalDepsException.withMessage(
            FailureDetails.ExternalDeps.Code.VERSION_RESOLUTION_ERROR,
            "Parsing %s failed, module versions must be of the form '<module name>@<version>'",
            context);
      }

      if (!RepositoryName.VALID_MODULE_NAME.matcher(pieces[0]).matches()) {
        throw ExternalDepsException.withMessage(
            FailureDetails.ExternalDeps.Code.VERSION_RESOLUTION_ERROR,
            "Parsing %s failed, invalid module name '%s': valid names must 1) only contain"
                + " lowercase letters (a-z), digits (0-9), dots (.), hyphens (-), and"
                + " underscores (_); 2) begin with a lowercase letter; 3) end with a lowercase"
                + " letter or digit.",
            context,
            pieces[0]);
      }

      Version version;
      try {
        version = Version.parse(pieces[1]);
      } catch (Version.ParseException e) {
        throw ExternalDepsException.withCauseAndMessage(
            FailureDetails.ExternalDeps.Code.VERSION_RESOLUTION_ERROR,
            e,
            "Parsing %s failed, invalid version specified for module: %s",
            context,
            pieces[1]);
      }

      allowedYankedVersionBuilder.add(ModuleKey.create(pieces[0], version));
    }
    return false;
  }

  private YankedVersionsUtil() {}
}
