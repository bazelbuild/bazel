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

import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.shell.AbnormalTerminationException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.shell.CommandResult;
import com.google.devtools.build.lib.shell.TerminationStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

/**
 * Obtains information pertaining to Apple host machines required for using Apple toolkits in
 * local action execution.
 */
public class AppleHostInfo {

  private static final String XCRUN_CACHE_FILENAME = "__xcruncache";

  /**
   * Returns the absolute root path of the target Apple SDK on the host system. This may spawn a
   * process and use the {@code /usr/bin/xcrun} binary to locate the target SDK. This uses a local
   * cache file under {@code bazel-out}, and will only spawn a new {@code xcrun} process in the case
   * of a cache miss.
   *
   * @throws UserExecException if there is an issue with obtaining the root from the spawned
   *     process, either because the SDK platform/version pair doesn't exist, or there was an
   *     unexpected issue finding or running the tool
   */
  public static String getSdkRoot(Path execRoot, String sdkVersion, String appleSdkPlatform)
      throws UserExecException {
    try {
      Path cacheFilePath = execRoot.getRelative(BlazeDirectories.RELATIVE_OUTPUT_PATH)
          .getRelative(XCRUN_CACHE_FILENAME);
      FileSystemUtils.touchFile(cacheFilePath);
      // TODO(bazel-team): Pass DEVELOPER_DIR to the cache manager.
      XcrunCacheManager cacheManager =
          new XcrunCacheManager(cacheFilePath, "" /* Developer dir empty */);

      String sdkString = appleSdkPlatform.toLowerCase() + sdkVersion;
      String cacheResult = cacheManager.getSdkRoot(sdkString);
      if (cacheResult != null) {
        return cacheResult;
      } else {
        // TODO(bazel-team): Pass DEVELOPER_DIR to the xcrun call.
        CommandResult xcrunResult = new Command(new String[] {"/usr/bin/xcrun", "--sdk",
            sdkString, "--show-sdk-path"}).execute();
    
        TerminationStatus xcrunStatus = xcrunResult.getTerminationStatus();
        if (!xcrunResult.getTerminationStatus().exited()) {
          throw new UserExecException(String.format("xcrun failed.\n%s\nStderr: %s",
              xcrunStatus.toString(), new String(xcrunResult.getStderr(), StandardCharsets.UTF_8)));
        }
        
        // calling xcrun via Command returns a value with a newline on the end.
        String sdkRoot = new String(xcrunResult.getStdout(), StandardCharsets.UTF_8).trim();

        cacheManager.writeSdkRoot(sdkString, sdkRoot);
        return sdkRoot;
      }
    } catch (AbnormalTerminationException e) {
      String message = String.format("%s : %s",
          e.getResult().getTerminationStatus(),
          new String(e.getResult().getStderr(), StandardCharsets.UTF_8));
      throw new UserExecException(message, e);
    } catch (CommandException | IOException e) {
      throw new UserExecException(e);
    }
  }
}
