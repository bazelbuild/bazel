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

package com.google.devtools.build.lib.view.actions;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.ArrayList;
import java.util.List;

/**
 * A customizable, serializable class for building memory efficient command lines.
 */
@Immutable
public final class CustomCommandLine extends CommandLine {

  private abstract static class ArgvFragment {
    abstract void eval(ImmutableList.Builder<String> builder);
  }

  // It's better to avoid anonymous classes if we want to serialize command lines

  private static final class ObjectArg extends ArgvFragment {
    private final Object arg;

    private ObjectArg(Object arg) {
      this.arg = arg;
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(arg.toString());
    }
  }

  private static final class StringsArg extends ArgvFragment {
    private final ImmutableList<String> args;

    private StringsArg(Iterable<String> args) {
      this.args = ImmutableList.copyOf(args);
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.addAll(args);
    }
  }

  // TODO(bazel-team): implement root relative path, join exec path, etc.
  private static final class ExecPathsArg extends ArgvFragment {

    private final Iterable<Artifact> artifacts;

    private ExecPathsArg(Iterable<Artifact> artifacts) {
      this.artifacts = ImmutableList.copyOf(artifacts);
    }

    private ExecPathsArg(NestedSet<Artifact> artifacts) {
      this.artifacts = artifacts;
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      ArrayList<String> args = new ArrayList<>();
      Artifact.addExecPaths(artifacts, args);
      builder.addAll(args);
    }
  }

  private static final class JoinExecPathsArg extends ArgvFragment {

    private final String delimiter;
    private final ImmutableList<Artifact> artifacts;

    private JoinExecPathsArg(String delimiter, Iterable<Artifact> artifacts) {
      this.delimiter = delimiter;
      this.artifacts = ImmutableList.copyOf(artifacts);
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(Artifact.joinExecPaths(delimiter, artifacts));
    }
  }

  private static final class PathWithTemplateArg extends ArgvFragment {

    private final String template;
    private final PathFragment[] paths;

    private PathWithTemplateArg(String template, PathFragment... paths) {
      this.template = template;
      this.paths = paths;
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      // PathFragment.toString() uses getPathString()
      builder.add(String.format(template, (Object[]) paths));
    }
  }

  // TODO(bazel-team): CustomArgv and CustomMultiArgv is  going to be difficult to expose
  // in Skylark. Maybe we can get rid of them by refactoring JavaCompileAction. It also
  // raises immutability / serialization issues.
  /**
   * Custom Java code producing a String argument. Usage of this class is discouraged.
   */
  public abstract static class CustomArgv extends ArgvFragment {

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(argv());
    }

    public abstract String argv();
  }

  /**
   * Custom Java code producing a List of String arguments. Usage of this class is discouraged.
   */
  public abstract static class CustomMultiArgv extends ArgvFragment {

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.addAll(argv());
    }

    public abstract Iterable<String> argv();
  }

  private static final class JoinPathsArg extends ArgvFragment {

    private final String delimiter;
    private final ImmutableList<PathFragment> paths;

    private JoinPathsArg(String delimiter, Iterable<PathFragment> paths) {
      this.delimiter = delimiter;
      this.paths = ImmutableList.copyOf(paths);
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(Joiner.on(delimiter).join(paths));
    }
  }

  /**
   * A Builder class for CustomCommandLine with the appropriate methods.
   */
  public static final class Builder {

    private final List<ArgvFragment> arguments = new ArrayList<>();

    public Builder add(CharSequence arg) {
      if (arg != null) {
        arguments.add(new ObjectArg(arg));
      }
      return this;
    }

    public Builder add(Label arg) {
      if (arg != null) {
        arguments.add(new ObjectArg(arg));
      }
      return this;
    }

    public Builder add(String arg, Iterable<String> args) {
      if (arg != null && args != null) {
        arguments.add(new ObjectArg(arg));
        arguments.add(new StringsArg(args));
      }
      return this;
    }

    public Builder add(Iterable<String> args) {
      if (args != null) {
        arguments.add(new StringsArg(args));
      }
      return this;
    }

    public Builder addExecPath(String arg, Artifact artifact) {
      if (arg != null && artifact != null) {
        arguments.add(new ObjectArg(arg));
        arguments.add(new ObjectArg(artifact.getExecPath()));
      }
      return this;
    }

    public Builder addExecPaths(String arg, Iterable<Artifact> artifacts) {
      if (arg != null && artifacts != null) {
        arguments.add(new ObjectArg(arg));
        arguments.add(new ExecPathsArg(artifacts));
      }
      return this;
    }

    public Builder addExecPaths(Iterable<Artifact> artifacts) {
      if (artifacts != null) {
        arguments.add(new ExecPathsArg(artifacts));
      }
      return this;
    }

    public Builder addExecPaths(NestedSet<Artifact> artifacts) {
      if (artifacts != null) {
        arguments.add(new ExecPathsArg(artifacts));
      }
      return this;
    }

    public Builder addJoinExecPaths(String arg, String delimiter, Iterable<Artifact> artifacts) {
      if (arg != null && artifacts != null) {
        arguments.add(new ObjectArg(arg));
        arguments.add(new JoinExecPathsArg(delimiter, artifacts));
      }
      return this;
    }

    public Builder addPath(PathFragment path) {
      if (path != null) {
        arguments.add(new ObjectArg(path));
      }
      return this;
    }

    public Builder addPaths(String template, PathFragment... path) {
      if (template != null && path != null) {
        arguments.add(new PathWithTemplateArg(template, path));
      }
      return this;
    }

    public Builder addJoinPaths(String delimiter, Iterable<PathFragment> paths) {
      if (delimiter != null && paths != null) {
        arguments.add(new JoinPathsArg(delimiter, paths));
      }
      return this;
    }

    public Builder add(CustomArgv arg) {
      if (arg != null) {
        arguments.add(arg);
      }
      return this;
    }

    public Builder add(CustomMultiArgv arg) {
      if (arg != null) {
        arguments.add(arg);
      }
      return this;
    }

    public CustomCommandLine build() {
      return new CustomCommandLine(arguments);
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  private final ImmutableList<ArgvFragment> arguments;

  private CustomCommandLine(List<ArgvFragment> arguments) {
    this.arguments = ImmutableList.copyOf(arguments);
  }

  @Override
  public Iterable<String> arguments() {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (ArgvFragment arg : arguments) {
      arg.eval(builder);
    }
    return builder.build();
  }
}
