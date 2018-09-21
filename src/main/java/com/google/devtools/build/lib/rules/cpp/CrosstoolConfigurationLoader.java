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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import com.google.protobuf.UninitializedMessageException;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

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

  private CrosstoolConfigurationLoader() {
  }

  /**
   * Reads the given <code>data</code> String, which must be in ascii format, into a protocol
   * buffer. It uses the <code>name</code> parameter for error messages.
   *
   * @throws IOException if the parsing failed
   */
  @VisibleForTesting
  public static CrosstoolRelease toReleaseConfiguration(String name, String data)
      throws InvalidConfigurationException {
    CrosstoolRelease.Builder builder = CrosstoolRelease.newBuilder();
    try {
      TextFormat.merge(data, builder);
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

  private static CrosstoolRelease getCrosstoolProtoFromBuildFile(
      ConfigurationEnvironment env, Label crosstoolTop)
      throws InterruptedException, InvalidConfigurationException {
    Target target;
    try {
      target = env.getTarget(crosstoolTop);
    } catch (NoSuchThingException e) {
      throw new IllegalStateException(e);  // Should have beeen evaluated by RedirectChaser
    }

    if (!(target instanceof Rule)) {
      return null;
    }

    Rule rule = (Rule) target;
    if (!(rule.getRuleClass().equals("cc_toolchain_suite"))
        || !rule.isAttributeValueExplicitlySpecified("proto")) {
      return null;
    }

    final String contents = NonconfigurableAttributeMapper.of(rule).get("proto", Type.STRING);
    return toReleaseConfiguration("cc_toolchain_suite rule " + crosstoolTop, contents);
  }

  private static CrosstoolRelease findCrosstoolConfiguration(
      ConfigurationEnvironment env, Label crosstoolTop)
      throws IOException, InvalidConfigurationException, InterruptedException {

    CrosstoolRelease crosstoolProtoFromBuildFile =
        getCrosstoolProtoFromBuildFile(env, crosstoolTop);
    if (crosstoolProtoFromBuildFile != null) {
      return crosstoolProtoFromBuildFile;
    }
    Path path;
    try {
      Package containingPackage;
      containingPackage = env.getTarget(crosstoolTop).getPackage();
      if (containingPackage == null) {
        return null;
      }
      path = env.getPath(containingPackage, CROSSTOOL_CONFIGURATION_FILENAME);
    } catch (NoSuchThingException e) {
      // Handled later
      return null;
    }

    if (path == null || !path.exists()) {
      // Normally you'd expect to return null when path is null (so Skyframe computes the value),
      // and throw when path doesn't exist. But because {@link ConfigurationFragmentFunction}
      // doesn't propagate the exceptions when there are valuesMissing(), we need to throw
      // this exception always just to be sure it gets through.
      throw new InvalidConfigurationException("The crosstool_top you specified was resolved to '" +
          crosstoolTop + "', which does not contain a CROSSTOOL file.");
    }

    // Do this before we read the data, so if it changes, we get a different MD5 the next time.
    // Alternatively, we could calculate the MD5 of the contents, which we also read, but this
    // is faster if the file comes from a file system with md5 support.
    String md5 = BaseEncoding.base16().lowerCase().encode(path.getDigest());
    CrosstoolRelease release;
    try {
      release =
          crosstoolReleaseCache.get(
              md5,
              () -> {
                try (InputStream inputStream = path.getInputStream()) {
                  return toReleaseConfiguration(
                      "CROSSTOOL file " + path,
                      new String(FileSystemUtils.readContentAsLatin1(inputStream)));
                }
              });
    } catch (ExecutionException e) {
      throw new InvalidConfigurationException(e);
    }

    return release;
  }

  /** Reads a crosstool file. */
  @Nullable
  public static CrosstoolRelease readCrosstool(ConfigurationEnvironment env, Label crosstoolTop)
      throws InvalidConfigurationException, InterruptedException {
    crosstoolTop = RedirectChaser.followRedirects(env, crosstoolTop, "crosstool_top");
    if (crosstoolTop == null) {
      return null;
    }
    try {
      return findCrosstoolConfiguration(env, crosstoolTop);
    } catch (IOException e) {
      throw new InvalidConfigurationException(e);
    }
  }
}
