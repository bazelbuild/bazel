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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Maps;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import java.io.File;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CommandUtilsTest {

  @Test
  public void longCommand() throws Exception {
    String[] args = new String[40];
    args[0] = "this_command_will_not_be_found";
    for (int i = 1; i < args.length; i++) {
      args[i] = "arg" + i;
    }
    Map<String, String> env = Maps.newTreeMap();
    env.put("PATH", "/usr/bin:/bin:/sbin");
    env.put("FOO", "foo");
    File directory = new File("/tmp");
    CommandException exception =
        assertThrows(CommandException.class, () -> new Command(args, env, directory).execute());
    String message = CommandUtils.describeCommandError(false, exception.getCommand());
      String verboseMessage = CommandUtils.describeCommandError(true, exception.getCommand());
    assertThat(message)
        .isEqualTo(
            "error executing command this_command_will_not_be_found arg1 "
                + "arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10 "
                + "arg11 arg12 arg13 arg14 arg15 arg16 arg17 arg18 "
                + "arg19 arg20 arg21 arg22 arg23 arg24 arg25 arg26 "
                + "arg27 arg28 arg29 arg30 "
                + "... (remaining 9 arguments skipped)");
    assertThat(verboseMessage)
        .isEqualTo(
            "error executing command \n"
                + "  (cd /tmp && \\\n"
                + "  exec env - \\\n"
                + "    FOO=foo \\\n"
                + "    PATH=/usr/bin:/bin:/sbin \\\n"
                + "  this_command_will_not_be_found arg1 "
                + "arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10 "
                + "arg11 arg12 arg13 arg14 arg15 arg16 arg17 arg18 "
                + "arg19 arg20 arg21 arg22 arg23 arg24 arg25 arg26 "
                + "arg27 arg28 arg29 arg30 arg31 arg32 arg33 arg34 "
                + "arg35 arg36 arg37 arg38 arg39)");
  }

  @Test
  public void failingCommand() throws Exception {
    String[] args = new String[3];
    args[0] = "/bin/sh";
    args[1] = "-c";
    args[2] = "echo Some errors 1>&2; echo Some output; exit 42";
    Map<String, String> env = Maps.newTreeMap();
    env.put("FOO", "foo");
    env.put("PATH", "/usr/bin:/bin:/sbin");
    CommandException exception =
        assertThrows(CommandException.class, () -> new Command(args, env, null).execute());
    String message = CommandUtils.describeCommandFailure(false, exception);
      String verboseMessage = CommandUtils.describeCommandFailure(true, exception);
      assertThat(message)
          .isEqualTo(
              "sh failed: error executing command "
                  + "/bin/sh -c 'echo Some errors 1>&2; echo Some output; exit 42': "
                  + "Process exited with status 42\n"
                  + "Some output\n"
                  + "Some errors\n");
    assertThat(verboseMessage)
        .isEqualTo(
            "sh failed: error executing command \n"
                + "  (exec env - \\\n"
                + "    FOO=foo \\\n"
                + "    PATH=/usr/bin:/bin:/sbin \\\n"
                + "  /bin/sh -c 'echo Some errors 1>&2; echo Some output; exit 42'): "
                + "Process exited with status 42\n"
                + "Some output\n"
                + "Some errors\n");
  }
}
