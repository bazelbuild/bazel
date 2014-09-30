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

package com.google.devtools.build.lib.shell;

import java.util.logging.Logger;

/**
 * <p>Represents an OS shell, such as "cmd" on Windows or "sh" on Unix-like
 * platforms. Currently, Linux and Windows XP are supported.</p>
 *
 * <p>This class encapsulates shell-specific logic, like how to
 * create a command line that uses the shell to invoke another command.
 */
public abstract class Shell {

  private static final Logger log =
    Logger.getLogger("com.google.devtools.build.lib.shell.Shell");
  
  private static final Shell platformShell;

  static {
    final String osName = System.getProperty("os.name");
    if ("Linux".equals(osName)) {
      platformShell = new SHShell();
    } else if ("Windows XP".equals(osName)) {
      platformShell = new WindowsCMDShell();
    } else {
      log.severe("OS not supported; will not be able to execute commands");
      platformShell = null;
    }
    log.config("Loaded shell support '" + platformShell +
               "' for OS '" + osName + "'");
  }

  private Shell() {
    // do nothing
  }

  /**
   * @return {@link Shell} subclass appropriate for the current platform
   * @throws UnsupportedOperationException if no such subclass exists
   */
  public static Shell getPlatformShell() {
    if (platformShell == null) {
      throw new UnsupportedOperationException("OS is not supported");
    }
    return platformShell;
  }

  /**
   * Creates a command line suitable for execution by
   * {@link Runtime#exec(String[])} from the given command string,
   * a command line which uses a shell appropriate for a particular
   * platform to execute the command (e.g. "/bin/sh" on Linux).
   *
   * @param command command for which to create a command line
   * @return String[] suitable for execution by
   *  {@link Runtime#exec(String[])}
   */
  public abstract String[] shellify(final String command);


  /**
   * Represents the <code>sh</code> shell commonly found on Unix-like
   * operating systems, including Linux.
   */
  private static final class SHShell extends Shell {

    /**
     * <p>Returns a command line which uses <code>cmd</code> to execute
     * the {@link Command}. Given the command <code>foo bar baz</code>,
     * for example, this will return a String array corresponding
     * to the command line:</p>
     *
     * <p><code>/bin/sh -c "foo bar baz"</code></p>
     *
     * <p>That is, it always returns a 3-element array.</p>
     *
     * @param command command for which to create a command line
     * @return String[] suitable for execution by
     *  {@link Runtime#exec(String[])}
     */
    @Override public String[] shellify(final String command) {
      if (command == null || command.length() == 0) {
        throw new IllegalArgumentException("command is null or empty");
      }
      return new String[] { "/bin/sh", "-c", command };
    }

  }

  /**
   * Represents the Windows command shell <code>cmd</code>.
   */
  private static final class WindowsCMDShell extends Shell {

    /**
     * <p>Returns a command line which uses <code>cmd</code> to execute
     * the {@link Command}. Given the command <code>foo bar baz</code>,
     * for example, this will return a String array corresponding
     * to the command line:</p>
     *
     * <p><code>cmd /S /C "foo bar baz"</code></p>
     *
     * <p>That is, it always returns a 4-element array.</p>
     *
     * @param command command for which to create a command line
     * @return String[] suitable for execution by
     *  {@link Runtime#exec(String[])}
     */
    @Override public String[] shellify(final String command) {
      if (command == null || command.length() == 0) {
        throw new IllegalArgumentException("command is null or empty");
      }
      return new String[] { "cmd", "/S", "/C", command };
    }

  }

}
