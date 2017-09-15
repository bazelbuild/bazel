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
import com.google.common.base.Function;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.analysis.RedirectChaser;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import com.google.protobuf.UninitializedMessageException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
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
  private static final String CROSSTOOL_CONFIGURATION_FILENAME = "CROSSTOOL";

  /**
   * Cache for storing result of toReleaseConfiguration function based on path and md5 sum of
   * input file. We can use md5 because result of this function depends only on the file content.
   */
  private static final Cache<String, CrosstoolRelease> crosstoolReleaseCache =
      CacheBuilder.newBuilder().concurrencyLevel(4).maximumSize(100).build();
  /**
   * A class that holds the results of reading a CROSSTOOL file.
   */
  public static class CrosstoolFile {
    private final String location;
    private final CrosstoolConfig.CrosstoolRelease crosstool;
    private final String md5;

    CrosstoolFile(String location, CrosstoolConfig.CrosstoolRelease crosstool, String md5) {
      this.location = location;
      this.crosstool = crosstool;
      this.md5 = md5;
    }

    /**
     * Returns a user-friendly location of the CROSSTOOL proto for error messages.
     */
    public String getLocation() {
      return location;
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

  /**
   * This class is the in-memory representation of a text-formatted Crosstool proto file.
   *
   * <p>This layer of abstraction is here so that we can load these protos either from BUILD files
   * or from CROSSTOOL files.
   *
   * <p>An implementation of this class should override {@link #getContents()} and call
   * the constructor with the MD5 checksum of what that method will return and a human-readable name
   * used in error messages.
   */
  private abstract static class CrosstoolProto {
    private final byte[] md5;
    private final String name;

    private CrosstoolProto(byte[] md5, String name) {
      this.md5 = md5;
      this.name = name;
    }

    /**
     * The binary MD5 checksum of the proto.
     */
    public byte[] getMd5() {
      return md5;
    }

    /**
     * A user-friendly string describing the location of the proto.
     */
    public String getName() {
      return name;
    }

    /**
     * The proto itself.
     */
    public abstract String getContents() throws IOException;
  }

  private static CrosstoolProto getCrosstoolProtofromBuildFile(
      ConfigurationEnvironment env, Label crosstoolTop) throws InterruptedException {
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
    byte[] md5 = new Fingerprint().addBytes(contents.getBytes(UTF_8)).digestAndReset();
    return new CrosstoolProto(md5, "cc_toolchain_suite rule " + crosstoolTop.toString()) {

      @Override
      public String getContents() throws IOException {
        return contents;
      }
    };
  }

  private static CrosstoolProto getCrosstoolProtoFromCrosstoolFile(
      ConfigurationEnvironment env, Label crosstoolTop)
      throws IOException, InvalidConfigurationException, InterruptedException {
    final Path path;
    try {
      Package containingPackage = env.getTarget(crosstoolTop.getLocalTargetLabel("BUILD"))
          .getPackage();
      if (containingPackage == null) {
        return null;
      }
      path = env.getPath(containingPackage, CROSSTOOL_CONFIGURATION_FILENAME);
    } catch (LabelSyntaxException e) {
      throw new InvalidConfigurationException(e);
    } catch (NoSuchThingException e) {
      // Handled later
      return null;
    }

    if (path == null || !path.exists()) {
      return null;
    }

    return new CrosstoolProto(path.getDigest(), "CROSSTOOL file " + path.getPathString()) {
      @Override
      public String getContents() throws IOException {
        try (InputStream inputStream = path.getInputStream()) {
          return new String(FileSystemUtils.readContentAsLatin1(inputStream));
        }
      }
    };
  }

  private static CrosstoolFile findCrosstoolConfiguration(
      ConfigurationEnvironment env, Label crosstoolTop)
      throws IOException, InvalidConfigurationException, InterruptedException {

    CrosstoolProto crosstoolProto = getCrosstoolProtofromBuildFile(env, crosstoolTop);
    if (crosstoolProto == null) {
      crosstoolProto = getCrosstoolProtoFromCrosstoolFile(env, crosstoolTop);
    }

    if (crosstoolProto == null) {
      throw new InvalidConfigurationException("The crosstool_top you specified was resolved to '" +
          crosstoolTop + "', which does not contain a CROSSTOOL file. " +
          "You can use a crosstool from the depot by specifying its label.");
    } else {
      // Do this before we read the data, so if it changes, we get a different MD5 the next time.
      // Alternatively, we could calculate the MD5 of the contents, which we also read, but this
      // is faster if the file comes from a file system with md5 support.
      final CrosstoolProto finalProto = crosstoolProto;
      String md5 = BaseEncoding.base16().lowerCase().encode(finalProto.getMd5());
      CrosstoolConfig.CrosstoolRelease release;
      try {
        release =
            crosstoolReleaseCache.get(
                md5, () -> toReleaseConfiguration(finalProto.getName(), finalProto.getContents()));
      } catch (ExecutionException e) {
        throw new InvalidConfigurationException(e);
      }

      return new CrosstoolFile(finalProto.getName(), release, md5);
    }
  }

  /** Reads a crosstool file. */
  @Nullable
  public static CrosstoolConfigurationLoader.CrosstoolFile readCrosstool(
      ConfigurationEnvironment env, Label crosstoolTop)
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
        CrosstoolConfigurationIdentifier.fromOptions(options);
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
    CppOptions cppOptions = options.get(CppOptions.class);
    boolean needsLipo =
        cppOptions.getLipoMode() != LipoMode.OFF && !cppOptions.convertLipoToThinLto;
    for (CrosstoolConfig.DefaultCpuToolchain selector : release.getDefaultToolchainList()) {
      if (needsLipo && !selector.getSupportsLipo()) {
        continue;
      }
      if (selector.getCpu().equals(desiredCpu)) {
        selectedIdentifier = selector.getToolchainIdentifier();
        break;
      }
    }

    if (selectedIdentifier == null) {
      StringBuilder cpuBuilder = new StringBuilder();
      for (CrosstoolConfig.DefaultCpuToolchain selector : release.getDefaultToolchainList()) {
        cpuBuilder.append("  ").append(selector.getCpu()).append(",\n");
      }
      throw new InvalidConfigurationException(
          "No toolchain found for cpu '" + desiredCpu
          + "'. Valid cpus are: [\n" + cpuBuilder + "]");
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

  /**
   * Appends a series of toolchain descriptions (as the blaze command line flags
   * that would specify that toolchain) to 'message'.
   */
  private static void describeToolchainList(StringBuilder message,
      Collection<CrosstoolConfig.CToolchain> toolchains) {
    message.append("[\n");
    for (CrosstoolConfig.CToolchain toolchain : toolchains) {
      message.append("  ");
      message.append(
          CrosstoolConfigurationIdentifier.fromToolchain(toolchain).describeFlags().trim());
      message.append(",\n");
    }
    message.append("]");
  }

  /**
   * Makes sure that {@code selectedIdentifier} is a valid identifier for a toolchain,
   * i.e. it starts with a letter or an underscore and continues with only dots, dashes,
   * spaces, letters, digits or underscores (i.e. matches the following regular expression:
   * "[a-zA-Z_][\.\- \w]*").
   *
   * @throws InvalidConfigurationException if selectedIdentifier does not match the
   *         aforementioned regular expression.
   */
  private static void checkToolChain(String selectedIdentifier, String cpu)
      throws InvalidConfigurationException {
    // If you update this regex, please do so in the javadoc comment too, and also in the
    // crosstool_config.proto file.
    String rx = "[a-zA-Z_][\\.\\- \\w]*";
    if (!selectedIdentifier.matches(rx)) {
      throw new InvalidConfigurationException(String.format(
          "Toolchain identifier '%s' for cpu '%s' is illegal (does not match '%s')",
          selectedIdentifier, cpu, rx));
    }
  }

  public static CrosstoolConfig.CrosstoolRelease getCrosstoolReleaseProto(
      ConfigurationEnvironment env,
      BuildOptions options,
      Label crosstoolTop,
      Function<String, String> cpuTransformer)
      throws InvalidConfigurationException, InterruptedException {
    CrosstoolConfigurationLoader.CrosstoolFile file =
        readCrosstool(env, crosstoolTop);
    // Make sure that we have the requested toolchain in the result. Throw an exception if not.
    selectToolchain(file.getProto(), options, cpuTransformer);
    return file.getProto();
  }
}
