// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.windows;

import com.google.common.base.Strings;
import java.io.PrintStream;
import java.nio.charset.Charset;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * Mock subprocess to be used for testing Windows process management. Command line usage:
 *
 * <ul>
 *   <li><code>I&lt;register&gt;&lt;count&gt;</code>: Read count bytes to the specified register
 *   <li><code>O-&lt;string&gt;</code>: Write a string to stdout</li>
 *   <li><code>E-&lt;string&gt;</code>: Write a string to stderr</li>
 *   <li><code>O$&lt;variable&gt;</code>: Write an environment variable to stdout</li>
 *   <li><code>E$&lt;variable&gt;</code>: Write an environment variable to stderr</li>
 *   <li><code>O.</code>: Write the cwd stdout</li>
 *   <li><code>E.</code>: Write the cwd stderr</li>
 *   <li><code>O&lt;register&gt;</code>: Write the contents of a register to stdout</li>
 *   <li><code>E&lt;register&gt;</code>: Write the contents of a register to stderr</li>
 *   <li><code>X&lt;exit code%gt;</code>: Exit with the specified exit code</li>
 *   <li><code>S&lt;seconds&gt;</code>: Wait the specified number of seconds</li>
 * </ul>
 *
 * <p>Registers are single characters. Each command line argument is interpreted as a single
 * operation. Example:
 *
 * <code>
 *   Ia10 Oa Oa Ea E-OVER X42
 * </code>
 *
 * Means: read 10 bytes from stdin, write them back twice to stdout and once to stderr, write
 * the string "OVER" to stderr then exit with exit code 42.
 */
public class MockSubprocess {
  private static Map<Character, byte[]> registers = new HashMap<>();
  private static final Charset UTF8 = Charset.forName("UTF-8");

  private static void writeBytes(PrintStream stream, String arg) throws Exception {

    byte[] buf;
    switch (arg.charAt(1)) {
      case '-':
        // Immediate string
        buf = arg.substring(2).getBytes(UTF8);
        break;

      case '$':
        // Environment variable
        buf = Strings.nullToEmpty(System.getenv(arg.substring(2))).getBytes(UTF8);
        break;

      case '.':
        buf = Paths.get(".").toAbsolutePath().normalize().toString().getBytes(UTF8);
        break;

      default:
        buf = registers.get(arg.charAt(1));
        break;
    }

    stream.write(buf, 0, buf.length);
}

  public static void main(String[] args) throws Exception {
    for (String arg : args) {
      switch (arg.charAt(0)) {
        case 'I':
          char register = arg.charAt(1);
          int length = Integer.parseInt(arg.substring(2));
          byte[] buf;
          if (length > 0) {
            buf = new byte[length];
            System.in.read(buf, 0, length);
          } else {
            buf = System.in.readAllBytes();
          }
          registers.put(register, buf);
          break;

        case 'E':
          writeBytes(System.err, arg);
          break;

        case 'O':
          writeBytes(System.out, arg);
          break;

        case 'W':
          try {
            Thread.sleep(Integer.parseInt(arg.substring(1)) * 1000);
          } catch (InterruptedException e) {
            // This is good enough for a mock process
            throw new IllegalStateException(e);
          }
          break;

        case 'X':
          System.exit(Integer.parseInt(arg.substring(1)));
        default: // fall out
      }
    }
  }
}
