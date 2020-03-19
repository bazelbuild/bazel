// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.testutil.Scratch;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import javax.annotation.Nullable;

class BuildFileBuilder {

  private static class Version {
    String name;
    String version;
    String[] aliases;

    Version(String name, String version, String... aliases) {
      this.name = name;
      this.version = version;
      this.aliases = aliases;
    }
  }

  private final HashMap<String, Version> allVersions = new HashMap<>();
  private final List<Version> localVersions = new ArrayList<>();
  private final List<Version> remoteVersions = new ArrayList<>();
  private final List<Version> explicitVersions = new ArrayList<>();

  @Nullable private String localDefaultLabel;
  @Nullable private String remoteDefaultLabel;
  @Nullable private String explicitDefaultLabel;

  /**
   * Registers a new local version.
   *
   * <p>Only one local version may set {@code isDefault} to true.
   *
   * @param name the name of the version
   * @param versionNumber the version number
   * @param isDefault whether this version is the local default
   * @param aliases the aliases for this version
   */
  BuildFileBuilder addLocalVersion(
      String name, String versionNumber, boolean isDefault, String... aliases) {
    Version version = new Version(name, versionNumber, aliases);
    allVersions.put(name, version);
    localVersions.add(version);
    if (isDefault) {
      checkState(localDefaultLabel == null, "Only one local version may set 'isDefault=true'");
      localDefaultLabel = name;
    }
    return this;
  }

  /**
   * Registers a new remote version.
   *
   * <p>Only one remote version may set {@code isDefault} to true.
   *
   * @param name the name of the version
   * @param versionNumber the version number
   * @param isDefault whether this version is the remote default
   * @param aliases the aliases for this version
   */
  BuildFileBuilder addRemoteVersion(
      String name, String versionNumber, boolean isDefault, String... aliases) {
    Version version = new Version(name, versionNumber, aliases);
    allVersions.put(name, version);
    remoteVersions.add(version);
    if (isDefault) {
      checkState(remoteDefaultLabel == null, "Only one remote version may set 'isDefault=true'");
      remoteDefaultLabel = name;
    }
    return this;
  }

  /**
   * Registers a new explicit version.
   *
   * <p>Only one explicit version may set {@code isDefault} to true.
   *
   * @param name the name of the version
   * @param versionNumber the version number
   * @param isDefault whether this version is the default
   * @param aliases the aliases for this version
   */
  BuildFileBuilder addExplicitVersion(
      String name, String versionNumber, boolean isDefault, String... aliases) {
    Version version = new Version(name, versionNumber, aliases);
    allVersions.put(name, version);
    explicitVersions.add(version);
    if (isDefault) {
      checkState(
          explicitDefaultLabel == null, "Only one explicit version may set 'isDefault=true'");
      explicitDefaultLabel = name;
    }
    return this;
  }

  private static void writeVersion(Version version, List<String> lines) {
    lines.add("xcode_version(");
    lines.add(String.format("    name = '%s',", version.name));
    lines.add(String.format("    version = '%s',", version.version));
    if (version.aliases.length != 0) {
      lines.add(String.format("    aliases = ['%s'],", String.join("', '", version.aliases)));
    }
    lines.add(")");
  }

  private static String formatVersionNames(List<Version> versions) {
    String versionNames = "";
    for (Version version : versions) {
      versionNames += String.format("':%s', ", version.name);
    }
    return "[" + versionNames + "]";
  }

  private static void writeAvailableXcodes(
      String name, String defaultVersion, List<Version> versions, List<String> lines) {
    lines.add("available_xcodes(");
    lines.add(String.format("    name = '%s',", name));
    lines.add(String.format("    default = ':%s',", defaultVersion));
    lines.add(String.format("    versions = %s,", formatVersionNames(versions)));
    lines.add(")");
  }

  private void writeAllAvailableXcodes(List<String> lines) {
    if (!localVersions.isEmpty()) {
      checkNotNull(localDefaultLabel, "One local version must be labeled as the default");
      writeAvailableXcodes("local", localDefaultLabel, localVersions, lines);
    }
    if (!remoteVersions.isEmpty()) {
      checkNotNull(remoteDefaultLabel, "One remote version must be labeled as the default");
      writeAvailableXcodes("remote", remoteDefaultLabel, remoteVersions, lines);
    }
  }

  private static void writeLocalRemoteXcodeConfig(List<String> lines) {
    lines.add("xcode_config(");
    lines.add("    name = 'foo',");
    lines.add("    local_versions = 'local',");
    lines.add("    remote_versions = 'remote',");
    lines.add(")");
  }

  private void writeStandardXcodeConfig(List<String> lines) {
    if (!explicitVersions.isEmpty()) {
      checkNotNull(
          explicitDefaultLabel, "'default' is a required field for the 'xcode_config' rule");
      lines.add("xcode_config(");
      lines.add("    name = 'foo',");
      lines.add(String.format("    default = '%s',", explicitDefaultLabel));
      lines.add(String.format("    versions = %s,", formatVersionNames(explicitVersions)));
      lines.add(")");
    }
  }

  void write(Scratch scratch, String filename) throws IOException {
    List<String> lines = new ArrayList<>();
    for (Version version : allVersions.values()) {
      writeVersion(version, lines);
    }

    if (!localVersions.isEmpty() && !remoteVersions.isEmpty()) {
      writeAllAvailableXcodes(lines);
      writeLocalRemoteXcodeConfig(lines);
    }
    if (!explicitVersions.isEmpty()) {
      writeStandardXcodeConfig(lines);
    }
    scratch.file(filename, lines.toArray(new String[0]));
  }
}
