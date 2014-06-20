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

package com.google.devtools.build.lib.view.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;

import java.util.Set;

/**
 * A collection of information about a machine, a set of system names which
 * describe the binaries it can run, and information about the availability of
 * standard tools.
 *
 * <p>This class is based on the notion of system names. The origin of these
 * names is the GNU C Compiler. Typically, the system name contains an
 * identifier for the cpu (e.g. {@code x86_64} or {@code alpha}), an identifier
 * for the machine (e.g. {@code pc}, or {@code unknown}), and an identifier for
 * the operating system (e.g. {@code cygwin} or {@code linux-gnu}).
 *
 * <p>Each system name describes a kind of exectuable. Every host has a primary
 * system name, but can run potentially any number of different system names.
 * For example a linux system ({@code x86_64-unknown-linux-gnu}) could also
 * execute windows code ({@code i686-unknown-cygwin}) via an emulation layer.
 * However, the system name itself does not specify additional requirements,
 * such as availability of runtime libraries or runtime tools.
 */
public final class MachineSpecification {

  private final String primarySystemName;
  private final Set<String> runnableSystemNames;

  /**
   * Creates a host information with the given primary system name, a set of
   * system names that can be run on the host, and the availability of gnu
   * build helpers on the host.
   */
  public MachineSpecification(String primarySystemName, Set<String> runnableSystemNames) {
    Preconditions.checkArgument(runnableSystemNames.contains(primarySystemName));
    this.primarySystemName = primarySystemName;
    this.runnableSystemNames = ImmutableSet.copyOf(runnableSystemNames);
  }

  /**
   * Returns true iff the host can run binaries compiled for {@code
   * systemName}.
   */
  public boolean canRun(String systemName) {
    return runnableSystemNames.contains(systemName);
  }

  @Override
  public String toString() {
    return primarySystemName;
  }

  /**
   * Calls `uname -m` and returns an instance based on the result. Typical
   * linux system names are {@code i686-unknown-linux-gnu} or {@code
   * x86_64-unknown-linux-gnu}.
   */
  public static MachineSpecification getLinuxHostSpecification() {
    // TODO(bazel-team): (2009) add JNI syscall binding for uname so we don't need to fork/exec.
    try {
      String uname = new String(new CommandBuilder().addArgs("/bin/uname", "-m").useTempDir()
          .build().execute().getStdout()).trim();
      String primarySystemName = uname + "-unknown-linux-gnu";
      Set<String> runnableSystemNames = Sets.newHashSet(primarySystemName);
      if (uname.equals("x86_64")) {
        runnableSystemNames.add("i686-unknown-linux-gnu");
      }
      return new MachineSpecification(primarySystemName, runnableSystemNames);
    } catch (CommandException e) {
      throw new IllegalStateException("'/bin/uname -m' failed: " + e.getMessage(), e);
    }
  }

  /**
   * Returns the machine configuration for remote execution machines.
   */
  @VisibleForTesting
  public static MachineSpecification getRemoteMachineSpecification() {
    String primarySystemName = "x86_64-unknown-linux-gnu";
    Set<String> runnableSystemNames = Sets.newHashSet(primarySystemName);
    runnableSystemNames.add("i686-unknown-linux-gnu");
    return new MachineSpecification(primarySystemName, runnableSystemNames);
  }

  /**
   * Converts the cpu name to a GNU system name, as used by {@link
   * MachineSpecification}. If the cpu is not a known value, it returns
   * <code>"unknown-unknown-linux-gnu"</code>.
   */
  public static String convertCpuToGnuSystemName(String cpu) {
    if ("piii".equals(cpu)) {
      return "i686-unknown-linux-gnu";
    } else if ("k8".equals(cpu)) {
      return "x86_64-unknown-linux-gnu";
    } else {
      return "unknown-unknown-linux-gnu";
    }
  }
}
