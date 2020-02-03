// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec.local;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.logging.Logger;

/**
 * Adds to the given environment all variables that are dependent on system state of the host
 * machine.
 *
 * <p>Admittedly, hermeticity is "best effort" in such cases; these environment values should be as
 * tied to configuration parameters as possible.
 *
 * <p>For example, underlying iOS toolchains require that SDKROOT resolve to an absolute system
 * path, but, when selecting which SDK to resolve, the version number comes from build
 * configuration.
 */
public final class XcodeLocalEnvProvider implements LocalEnvProvider {

  private static final Logger log = Logger.getLogger(XcodeLocalEnvProvider.class.getName());

  private final Map<String, String> clientEnv;

  private static final ConcurrentMap<String, String> sdkRootCache = new ConcurrentHashMap<>();
  private static final ConcurrentMap<String, String> developerDirCache = new ConcurrentHashMap<>();

  /**
   * Creates a new {@link XcodeLocalEnvProvider}.
   *
   * <p>Use {@link LocalEnvProvider#forCurrentOs(Map)} to instantiate this.
   *
   * @param clientEnv a map of the current Bazel command's environment
   */
  XcodeLocalEnvProvider(Map<String, String> clientEnv) {
    this.clientEnv = clientEnv;
  }

  @Override
  public ImmutableMap<String, String> rewriteLocalEnv(
      Map<String, String> env, BinTools binTools, String fallbackTmpDir) throws IOException {
    boolean containsXcodeVersion = env.containsKey(AppleConfiguration.XCODE_VERSION_ENV_NAME);
    boolean containsAppleSdkVersion =
        env.containsKey(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME);

    ImmutableMap.Builder<String, String> newEnvBuilder = ImmutableMap.builder();
    newEnvBuilder.putAll(Maps.filterKeys(env, k -> !k.equals("TMPDIR")));
    String p = clientEnv.get("TMPDIR");
    if (Strings.isNullOrEmpty(p)) {
      // Do not use `fallbackTmpDir`, use `/tmp` instead. This way if the user didn't export TMPDIR
      // in their environment, Bazel will still set a TMPDIR that's Posixy enough and plays well
      // with heavily path-length-limited scenarios, such as the socket creation scenario that
      // motivated https://github.com/bazelbuild/bazel/issues/4376.
      p = "/tmp";
    }
    newEnvBuilder.put("TMPDIR", p);

    if (!containsXcodeVersion && !containsAppleSdkVersion) {
      return newEnvBuilder.build();
    }

    // Empty developer dir indicates to use the system default.
    // TODO(bazel-team): Bazel's view of the xcode version and developer dir should be explicitly
    // set for build hermeticity.
    String developerDir = "";
    if (containsXcodeVersion) {
      String version = env.get(AppleConfiguration.XCODE_VERSION_ENV_NAME);
      developerDir = getDeveloperDir(binTools, DottedVersion.fromStringUnchecked(version));
      newEnvBuilder.put("DEVELOPER_DIR", developerDir);
    }
    if (containsAppleSdkVersion) {
      // The Apple platform is needed to select the appropriate SDK.
      if (!env.containsKey(AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME)) {
        throw new IOException("Could not resolve apple platform for determining SDK");
      }
      String iosSdkVersion = env.get(AppleConfiguration.APPLE_SDK_VERSION_ENV_NAME);
      String appleSdkPlatform = env.get(AppleConfiguration.APPLE_SDK_PLATFORM_ENV_NAME);
      newEnvBuilder.put("SDKROOT", getSdkRoot(developerDir, iosSdkVersion, appleSdkPlatform));
    }

    return newEnvBuilder.build();
  }

  /**
   * Queries the path to the target Apple SDK on the host system for a given version of Xcode.
   *
   * <p>This spawns a subprocess to run the {@code /usr/bin/xcrun} binary to locate the target SDK.
   * As this is a costly operation, always call {@link #getSdkRoot(String, String, String)} instead,
   * which does caching.
   *
   * @param developerDir the value of {@code DEVELOPER_DIR} for the target version of xcode
   * @param sdkVersion the sdk version; for example, {@code 9.1}
   * @param appleSdkPlatform the sdk platform; for example, {@code iPhoneOS}
   * @return an absolute path to the root of the target Apple SDK
   * @throws IOException if there is an issue with obtaining the root from the spawned process,
   *     either because the SDK platform/version pair doesn't exist, or there was an unexpected
   *     issue finding or running the tool
   */
  private static String querySdkRoot(
      String developerDir,
      String sdkVersion,
      String appleSdkPlatform)
      throws IOException {
    try {
      String sdkString = appleSdkPlatform.toLowerCase() + sdkVersion;
      Map<String, String> env =
          Strings.isNullOrEmpty(developerDir)
              ? ImmutableMap.<String, String>of()
              : ImmutableMap.of("DEVELOPER_DIR", developerDir);
      CommandResult xcrunResult =
          new Command(
              new String[] {"/usr/bin/xcrun", "--sdk", sdkString, "--show-sdk-path"},
              env,
              null)
              .execute();

      return new String(xcrunResult.getStdout(), StandardCharsets.UTF_8).trim();
    } catch (AbnormalTerminationException e) {
      TerminationStatus terminationStatus = e.getResult().getTerminationStatus();

      if (terminationStatus.exited()) {
        throw new IOException(
            String.format(
                "xcrun failed with code %s.\n"
                    + "This most likely indicates that SDK version [%s] for platform [%s] is "
                    + "unsupported for the target version of xcode.\n"
                    + "%s\n"
                    + "stdout: %s"
                    + "stderr: %s",
                terminationStatus.getExitCode(),
                sdkVersion,
                appleSdkPlatform,
                terminationStatus.toString(),
                new String(e.getResult().getStdout(), StandardCharsets.UTF_8),
                new String(e.getResult().getStderr(), StandardCharsets.UTF_8)));
      }
      String message =
          String.format(
              "xcrun failed.\n" + "%s\n" + "stdout: %s\n" + "stderr: %s",
              e.getResult().getTerminationStatus(),
              new String(e.getResult().getStdout(), StandardCharsets.UTF_8),
              new String(e.getResult().getStderr(), StandardCharsets.UTF_8));
      throw new IOException(message, e);
    } catch (CommandException e) {
      throw new IOException(e);
    }
  }

  /**
   * Returns the path to the target Apple SDK on the host system for a given version of Xcode.
   *
   * <p>This may delegate to {@link #querySdkRoot(String, String, String)} to obtain the path from
   * external sources in the system. Values are cached in-memory throughout the lifetime of the
   * Bazel server.
   *
   * @param developerDir the value of {@code DEVELOPER_DIR} for the target version of xcode
   * @param sdkVersion the sdk version; for example, {@code 9.1}
   * @param appleSdkPlatform the sdk platform; for example, {@code iPhoneOS}
   * @return an absolute path to the root of the target Apple SDK
   * @throws IOException if there is an issue with obtaining the root from the spawned process,
   *     either because the SDK platform/version pair doesn't exist, or there was an unexpected
   *     issue finding or running the tool
   */
  private static String getSdkRoot(String developerDir, String sdkVersion, String appleSdkPlatform)
      throws IOException {
    try {
      return sdkRootCache.computeIfAbsent(
          developerDir + ":" + appleSdkPlatform.toLowerCase() + ":" + sdkVersion,
          (key) -> {
            try {
              String sdkRoot = querySdkRoot(developerDir, sdkVersion, appleSdkPlatform);
              log.info("Queried Xcode SDK root with key " + key + " and got " + sdkRoot);
              return sdkRoot;
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
          });
    } catch (UncheckedIOException e) {
      throw e.getCause();
    }
  }

  /**
   * Queries the path to the Xcode developer directory on the host system for the given Xcode
   * version.
   *
   * <p>This spawns a subprocess to run the {@code xcode-locator} binary. As this is a costly
   * operation, always call {@link #getDeveloperDir(Path, DottedVersion)} instead, which does
   * caching.
   *
   * @param binTools the {@link BinTools}, used to locate the cache file
   * @param version the xcode version number to look up
   * @return an absolute path to the root of the Xcode developer directory
   * @throws IOException if there is an issue with obtaining the path from the spawned process,
   *     either because there is no installed xcode with the given version, or there was an
   *     unexpected issue finding or running the tool
   */
  private static String queryDeveloperDir(BinTools binTools, DottedVersion version)
      throws IOException {
    String xcodeLocatorPath = binTools.getEmbeddedPath("xcode-locator").getPathString();
    try {
      CommandResult xcodeLocatorResult =
          new Command(new String[] {xcodeLocatorPath, version.toString()}).execute();

      return new String(xcodeLocatorResult.getStdout(), StandardCharsets.UTF_8).trim();
    } catch (AbnormalTerminationException e) {
      TerminationStatus terminationStatus = e.getResult().getTerminationStatus();

      String message;
      if (e.getResult().getTerminationStatus().exited()) {
        message =
            String.format(
                "Running '%s %s' failed with code %s.\n"
                    + "This most likely indicates that xcode version %s is not available on the "
                    + "host machine.\n"
                    + "%s\n"
                    + "stdout: %s\n"
                    + "stderr: %s",
                xcodeLocatorPath,
                version,
                terminationStatus.getExitCode(),
                version,
                terminationStatus.toString(),
                new String(e.getResult().getStdout(), StandardCharsets.UTF_8),
                new String(e.getResult().getStderr(), StandardCharsets.UTF_8));
      } else {
        message =
            String.format(
                "Running '%s %s' failed.\n" + "%s\n" + "stdout: %s\n" + "stderr: %s",
                xcodeLocatorPath,
                version,
                e.getResult().getTerminationStatus(),
                new String(e.getResult().getStdout(), StandardCharsets.UTF_8),
                new String(e.getResult().getStderr(), StandardCharsets.UTF_8));
      }
      throw new IOException(message, e);
    } catch (CommandException e) {
      throw new IOException(e);
    }
  }

  /**
   * Returns the absolute root path of the xcode developer directory on the host system for the
   * given Xcode version.
   *
   * <p>This may delegate to {@link #queryDeveloperDir(Path, DottedVersion)} to obtain the path from
   * external sources in the system. Values are cached in-memory throughout the lifetime of the
   * Bazel server.
   *
   * @param binTools the {@link BinTools} path, used to locate the cache file
   * @param version the xcode version number to look up
   * @return an absolute path to the root of the Xcode developer directory
   * @throws IOException if there is an issue with obtaining the path from the spawned process,
   *     either because there is no installed xcode with the given version, or there was an
   *     unexpected issue finding or running the tool
   */
  private static String getDeveloperDir(BinTools binTools, DottedVersion version)
      throws IOException {
    try {
      return developerDirCache.computeIfAbsent(
          version.toString(),
          (key) -> {
            try {
              String developerDir = queryDeveloperDir(binTools, version);
              log.info("Queried Xcode developer dir with key " + key + " and got " + developerDir);
              return developerDir;
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
          });
    } catch (UncheckedIOException e) {
      throw e.getCause();
    }
  }
}
