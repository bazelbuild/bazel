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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import com.google.protobuf.UninitializedMessageException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.concurrent.ExecutionException;

import javax.annotation.Nullable;

/**
 * A loader that reads Crosstool configuration files and creates CToolchain
 * instances from them.
 */
public class CrosstoolConfigurationLoader {
  private static final String CROSSTOOL_CONFIGURATION_FILENAME = "CROSSTOOL";

  /**
   * Cache for storing result of toReleaseConfiguration function based on path and md5 sum of
   * input file. We can use md5 because result of this function depends only on the file content.
   */
  private static final LoadingCache<Pair<Path, String>, CrosstoolConfig.CrosstoolRelease> 
      crosstoolReleaseCache = CacheBuilder.newBuilder().concurrencyLevel(4).maximumSize(100).build(
        new CacheLoader<Pair<Path, String>, CrosstoolConfig.CrosstoolRelease>() {
          @Override
          public CrosstoolConfig.CrosstoolRelease load(Pair<Path, String> key) throws IOException {
            char[] data = FileSystemUtils.readContentAsLatin1(key.first);
            return toReleaseConfiguration(key.first.getPathString(), new String(data));
          }
        });

  /**
   * A class that holds the results of reading a CROSSTOOL file.
   */
  public static class CrosstoolFile {
    private final Label crosstoolTop;
    private Path crosstoolPath;
    private CrosstoolConfig.CrosstoolRelease crosstool;
    private String md5;

    CrosstoolFile(Label crosstoolTop) {
      this.crosstoolTop = crosstoolTop;
    }

    void setCrosstoolPath(Path crosstoolPath) {
      this.crosstoolPath = crosstoolPath;
    }

    void setCrosstool(CrosstoolConfig.CrosstoolRelease crosstool) {
      this.crosstool = crosstool;
    }

    void setMd5(String md5) {
      this.md5 = md5;
    }

    /**
     * Returns the crosstool top as resolved.
     */
    public Label getCrosstoolTop() {
      return crosstoolTop;
    }

    /**
     * Returns the absolute path from which the CROSSTOOL file was read.
     */
    public Path getCrosstoolPath() {
      return crosstoolPath;
    }

    /**
     * Returns the parsed contents of the CROSSTOOL file.
     */
    public CrosstoolConfig.CrosstoolRelease getProto() {
      return crosstool;
    }

    /**
     * Returns an MD5 hash of the CROSSTOOL file contents.
     */
    public String getMd5() {
      return md5;
    }
  }

  private CrosstoolConfigurationLoader() {
  }

  /**
   * Reads the given <code>data</code> String, which must be in ascii format,
   * into a protocol buffer. It uses the <code>name</code> parameter for error
   * messages.
   *
   * @throws IOException if the parsing failed
   */
  @VisibleForTesting
  static CrosstoolConfig.CrosstoolRelease toReleaseConfiguration(String name, String data)
      throws IOException {
    CrosstoolConfig.CrosstoolRelease.Builder builder =
        CrosstoolConfig.CrosstoolRelease.newBuilder();
    try {
      TextFormat.merge(data, builder);
      return builder.build();
    } catch (ParseException e) {
      throw new IOException("Could not read the crosstool configuration file '" + name + "', "
          + "because of a parser error (" + e.getMessage() + ")");
    } catch (UninitializedMessageException e) {
      throw new IOException("Could not read the crosstool configuration file '" + name + "', "
          + "because of an incomplete protocol buffer (" + e.getMessage() + ")");
    }
  }

  private static boolean findCrosstoolConfiguration(
      ConfigurationEnvironment env,
      CrosstoolConfigurationLoader.CrosstoolFile file)
      throws IOException, InvalidConfigurationException {
    Label crosstoolTop = file.getCrosstoolTop();
    Path path = null;
    try {
      Package containingPackage = env.getTarget(crosstoolTop.getLocalTargetLabel("BUILD"))
          .getPackage();
      if (containingPackage == null) {
        return false;
      }
      path = env.getPath(containingPackage, CROSSTOOL_CONFIGURATION_FILENAME);
    } catch (SyntaxException e) {
      throw new InvalidConfigurationException(e);
    } catch (NoSuchThingException e) {
      // Handled later
    }

    // If we can't find a file, fall back to the provided alternative.
    if (path == null || !path.exists()) {
      throw new InvalidConfigurationException("The crosstool_top you specified was resolved to '" +
          crosstoolTop + "', which does not contain a CROSSTOOL file. " +
          "You can use a crosstool from the depot by specifying its label.");
    } else {
      // Do this before we read the data, so if it changes, we get a different MD5 the next time.
      // Alternatively, we could calculate the MD5 of the contents, which we also read, but this
      // is faster if the file comes from a file system with md5 support.
      file.setCrosstoolPath(path);
      String md5 = BaseEncoding.base16().lowerCase().encode(path.getMD5Digest());
      CrosstoolConfig.CrosstoolRelease release;
      try {
        release = crosstoolReleaseCache.get(new Pair<Path, String>(path, md5));
        file.setCrosstool(release);
        file.setMd5(md5);
      } catch (ExecutionException e) {
        throw new InvalidConfigurationException(e);
      }
    }
    return true;
  }

  /**
   * Reads a crosstool file.
   */
  @Nullable
  public static CrosstoolConfigurationLoader.CrosstoolFile readCrosstool(
      ConfigurationEnvironment env, Label crosstoolTop) throws InvalidConfigurationException {
    crosstoolTop = RedirectChaser.followRedirects(env, crosstoolTop, "crosstool_top");
    if (crosstoolTop == null) {
      return null;
    }
    CrosstoolConfigurationLoader.CrosstoolFile file =
        new CrosstoolConfigurationLoader.CrosstoolFile(crosstoolTop);
    try {
      boolean allDependenciesPresent = findCrosstoolConfiguration(env, file);
      return allDependenciesPresent ? file : null;
    } catch (IOException e) {
      throw new InvalidConfigurationException(e);
    }
  }

  /**
   * Selects a crosstool toolchain corresponding to the given crosstool
   * configuration options. If all of these options are null, it returns the default
   * toolchain specified in the crosstool release. If only cpu is non-null, it
   * returns the default toolchain for that cpu, as specified in the crosstool
   * release. Otherwise, all values must be non-null, and this method
   * returns the toolchain which matches all of the values.
   *
   * @throws NullPointerException if {@code release} is null
   * @throws InvalidConfigurationException if no matching toolchain can be found, or
   *     if the input parameters do not obey the constraints described above
   */
  public static CrosstoolConfig.CToolchain selectToolchain(
      CrosstoolConfig.CrosstoolRelease release, BuildOptions options,
      Function<String, String> cpuTransformer)
          throws InvalidConfigurationException {
    CrosstoolConfigurationIdentifier config =
        CrosstoolConfigurationIdentifier.fromReleaseAndCrosstoolConfiguration(release, options);
    if ((config.getCompiler() != null) || (config.getLibc() != null)) {
      ArrayList<CrosstoolConfig.CToolchain> candidateToolchains = new ArrayList<>();
      for (CrosstoolConfig.CToolchain toolchain : release.getToolchainList()) {
        if (config.isCandidateToolchain(toolchain)) {
          candidateToolchains.add(toolchain);
        }
      }
      switch (candidateToolchains.size()) {
        case 0: {
          StringBuilder message = new StringBuilder();
          message.append("No toolchain found for");
          message.append(config.describeFlags());
          message.append(". Valid toolchains are: ");
          describeToolchainList(message, release.getToolchainList());
          throw new InvalidConfigurationException(message.toString());
        }
        case 1:
          return candidateToolchains.get(0);
        default: {
          StringBuilder message = new StringBuilder();
          message.append("Multiple toolchains found for");
          message.append(config.describeFlags());
          message.append(": ");
          describeToolchainList(message, candidateToolchains);
          throw new InvalidConfigurationException(message.toString());
        }
      }
    }
    String selectedIdentifier = null;
    // We use fake CPU values to allow cross-platform builds for other languages that use the
    // C++ toolchain. Translate to the actual target architecture.
    String desiredCpu = cpuTransformer.apply(config.getCpu());
    for (CrosstoolConfig.DefaultCpuToolchain selector : release.getDefaultToolchainList()) {
      if (selector.getCpu().equals(desiredCpu)) {
        selectedIdentifier = selector.getToolchainIdentifier();
        break;
      }
    }
    checkToolChain(selectedIdentifier, desiredCpu);
    for (CrosstoolConfig.CToolchain toolchain : release.getToolchainList()) {
      if (toolchain.getToolchainIdentifier().equals(selectedIdentifier)) {
        return toolchain;
      }
    }
    throw new InvalidConfigurationException("Inconsistent crosstool configuration; no toolchain "
        + "corresponding to '" + selectedIdentifier + "' found for cpu '" + config.getCpu() + "'");
  }

  private static String describeToolchainFlags(CrosstoolConfig.CToolchain toolchain) {
    return CrosstoolConfigurationIdentifier.fromToolchain(toolchain).describeFlags();
  }

  /**
   * Appends a series of toolchain descriptions (as the blaze command line flags
   * that would specify that toolchain) to 'message'.
   */
  private static void describeToolchainList(StringBuilder message,
      Collection<CrosstoolConfig.CToolchain> toolchains) {
    message.append("[");
    for (CrosstoolConfig.CToolchain toolchain : toolchains) {
      message.append(describeToolchainFlags(toolchain));
      message.append(",");
    }
    message.append("]");
  }

  /**
   * Makes sure that {@code selectedIdentifier} is a valid identifier for a toolchain,
   * i.e. it starts with a letter or an underscore and continues with only dots, dashes,
   * spaces, letters, digits or underscores (i.e. matches the following regular expression:
   * "[a-zA-Z_][\.\- \w]*").
   *
   * @throws InvalidConfigurationException if selectedIdentifier is null or does not match the
   *         aforementioned regular expression.
   */
  private static void checkToolChain(String selectedIdentifier, String cpu)
      throws InvalidConfigurationException {
    if (selectedIdentifier == null) {
      throw new InvalidConfigurationException("No toolchain found for cpu '" + cpu + "'");
    }
    // If you update this regex, please do so in the javadoc comment too, and also in the
    // crosstool_config.proto file.
    String rx = "[a-zA-Z_][\\.\\- \\w]*";
    if (!selectedIdentifier.matches(rx)) {
      throw new InvalidConfigurationException("Toolchain identifier for cpu '" + cpu + "' " +
          "is illegal (does not match '" + rx + "')");
    }
  }

  public static CrosstoolConfig.CrosstoolRelease getCrosstoolReleaseProto(
      ConfigurationEnvironment env, BuildOptions options,
      Label crosstoolTop, Function<String, String> cpuTransformer)
      throws InvalidConfigurationException {
    CrosstoolConfigurationLoader.CrosstoolFile file =
        readCrosstool(env, crosstoolTop);
    // Make sure that we have the requested toolchain in the result. Throw an exception if not.
    selectToolchain(file.getProto(), options, cpuTransformer);
    return file.getProto();
  }
}
