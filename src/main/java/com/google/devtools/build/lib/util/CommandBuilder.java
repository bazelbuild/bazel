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

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.vfs.Path;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Implements {@link Command} builder.
 *
 * <p>TODO(bazel-team): (2010) Some of the code here is very similar to the {@link
 * com.google.devtools.build.lib.shell.Shell} class. This should be looked at.
 */
public final class CommandBuilder {

  private final ArrayList<String> argv = new ArrayList<>();
  private final HashMap<String, String> env = new HashMap<>();
  private final File workingDir;

  public CommandBuilder(Path workingDir) {
    this(workingDir.getPathFile());
  }

  public CommandBuilder(File workingDir) {
    this.workingDir = Preconditions.checkNotNull(workingDir);
  }

  public CommandBuilder addArg(String arg) {
    Preconditions.checkNotNull(arg);
    argv.add(arg);
    return this;
  }

  public CommandBuilder addArgs(Iterable<String> args) {
    Preconditions.checkArgument(!Iterables.contains(args, null));
    Iterables.addAll(argv, args);
    return this;
  }

  public CommandBuilder addArgs(String... args) {
    return addArgs(Arrays.asList(args));
  }

  public CommandBuilder addEnv(Map<String, String> env) {
    Preconditions.checkNotNull(env);
    this.env.putAll(env);
    return this;
  }

  public CommandBuilder emptyEnv() {
    env.clear();
    return this;
  }

  public CommandBuilder setEnv(Map<String, String> env) {
    emptyEnv();
    addEnv(env);
    return this;
  }

  public Command build() {
    Preconditions.checkState(!argv.isEmpty(), "At least one argument is expected");
    return new Command(argv.toArray(new String[0]), env, workingDir);
  }
}
