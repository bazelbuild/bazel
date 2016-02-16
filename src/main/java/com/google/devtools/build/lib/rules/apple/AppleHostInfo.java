// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * Obtains information pertaining to Apple host machines required for using Apple toolkits in
 * local action execution.
 */
public class AppleHostInfo {

  private static final String XCRUN_CACHE_FILENAME = "__xcruncache";
  private static final String XCODE_LOCATOR_CACHE_FILENAME = "__xcodelocatorcache";

  /**
   * Returns the absolute root path of the target Apple SDK on the host system for a given
   * version of xcode (as defined by the given {@code developerDir}). This may spawn a
   * process and use the {@code /usr/bin/xcrun} binary to locate the target SDK. This uses a local
   * cache file under {@code bazel-out}, and will only spawn a new {@code xcrun} process in the case
   * of a cache miss.
   *
   * @param execRoot the execution root path, used to locate the cache file
   * @param developerDir the value of {@code DEVELOPER_DIR} for the target version of xcode
   * @param sdkVersion the sdk version, for example, "9.1"
   * @param appleSdkPlatform the sdk platform, for example, "iPhoneOS"
   * @throws UserExecException if there is an issue with obtaining the root from the spawned
   *     process, either because the SDK platform/version pair doesn't exist, or there was an
   *     unexpected issue finding or running the tool
   */
  public static String getSdkRoot(Path execRoot, String developerDir,
      String sdkVersion, String appleSdkPlatform) throws UserExecException {
    try {
      CacheManager cacheManager =
          new CacheManager(execRoot.getRelative(BlazeDirectories.RELATIVE_OUTPUT_PATH),
              XCRUN_CACHE_FILENAME);

      String sdkString = appleSdkPlatform.toLowerCase() + sdkVersion;
      String cacheResult = cacheManager.getValue(developerDir, sdkString);
      if (cacheResult != null) {
        return cacheResult;
      } else {
        Map<String, String> env = Strings.isNullOrEmpty(developerDir)
            ? ImmutableMap.<String, String>of() : ImmutableMap.of("DEVELOPER_DIR", developerDir);
        CommandResult xcrunResult = new Command(
            new String[] {"/usr/bin/xcrun", "--sdk", sdkString, "--show-sdk-path"},
            env, null).execute();

        // calling xcrun via Command returns a value with a newline on the end.
        String sdkRoot = new String(xcrunResult.getStdout(), StandardCharsets.UTF_8).trim();

        cacheManager.writeEntry(ImmutableList.of(developerDir, sdkString), sdkRoot);
        return sdkRoot;
      }
    } catch (AbnormalTerminationException e) {
      TerminationStatus terminationStatus = e.getResult().getTerminationStatus();

      if (terminationStatus.exited()) {
        throw new UserExecException(
            String.format("xcrun failed with code %s.\n"
                + "This most likely indicates that SDK version [%s] for platform [%s] is "
                + "unsupported for the target version of xcode.\n"
                + "%s\n"
                + "Stderr: %s",
                terminationStatus.getExitCode(),
                sdkVersion, appleSdkPlatform,
                terminationStatus.toString(),
                new String(e.getResult().getStderr(), StandardCharsets.UTF_8)));
      }
      String message = String.format("xcrun failed.\n%s\n%s",
          e.getResult().getTerminationStatus(),
          new String(e.getResult().getStderr(), StandardCharsets.UTF_8));
      throw new UserExecException(message, e);
    } catch (CommandException | IOException e) {
      throw new UserExecException(e);
    }
  }
  
  /**
   * Returns the absolute root path of the xcode developer directory on the host system for
   * the given xcode version. This may spawn a process and use the {@code xcode-locator} binary.
   * This uses a local cache file under {@code bazel-out}, and will only spawn a new process in the
   * case of a cache miss.
   *
   * @param execRoot the execution root path, used to locate the cache file
   * @param version the xcode version number to look up
   * @throws UserExecException if there is an issue with obtaining the path from the spawned
   *     process, either because there is no installed xcode with the given version, or
   *     there was an unexpected issue finding or running the tool
   */
  public static String getDeveloperDir(Path execRoot, DottedVersion version)
      throws UserExecException {
    try {
      CacheManager cacheManager =
          new CacheManager(execRoot.getRelative(BlazeDirectories.RELATIVE_OUTPUT_PATH),
              XCODE_LOCATOR_CACHE_FILENAME);

      String cacheResult = cacheManager.getValue(version.toString());
      if (cacheResult != null) {
        return cacheResult;
      } else {
        CommandResult xcodeLocatorResult = new Command(new String[] {
            execRoot.getRelative("_bin/xcode-locator").getPathString(), version.toString()})
            .execute();

        String developerDir =
            new String(xcodeLocatorResult.getStdout(), StandardCharsets.UTF_8).trim();

        cacheManager.writeEntry(ImmutableList.of(version.toString()), developerDir);
        return developerDir;
      }
    } catch (AbnormalTerminationException e) {
      TerminationStatus terminationStatus = e.getResult().getTerminationStatus();

      String message;
      if (e.getResult().getTerminationStatus().exited()) {
        message = String.format("xcode-locator failed with code %s.\n"
            + "This most likely indicates that xcode version %s is not available on the host "
            + "machine.\n"
            + "%s\n"
            + "stderr: %s",
            terminationStatus.getExitCode(),
            version,
            terminationStatus.toString(),
            new String(e.getResult().getStderr(), StandardCharsets.UTF_8));
      } else {
        message = String.format("xcode-locator failed. %s\nstderr: %s",
            e.getResult().getTerminationStatus(),
            new String(e.getResult().getStderr(), StandardCharsets.UTF_8));
      }
      throw new UserExecException(message, e);
    } catch (CommandException | IOException e) {
      throw new UserExecException(e);
    }
  }
}
