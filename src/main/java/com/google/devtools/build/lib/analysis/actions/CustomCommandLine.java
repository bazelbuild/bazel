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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
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

  private static final class JoinExecPathsArg extends ArgvFragment {

    private final String delimiter;
    private final Iterable<Artifact> artifacts;

    private JoinExecPathsArg(String delimiter, Iterable<Artifact> artifacts) {
      this.delimiter = delimiter;
      this.artifacts = CollectionUtils.makeImmutable(artifacts);
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
    private final Iterable<PathFragment> paths;

    private JoinPathsArg(String delimiter, Iterable<PathFragment> paths) {
      this.delimiter = delimiter;
      this.paths = CollectionUtils.makeImmutable(paths);
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(Joiner.on(delimiter).join(paths));
    }
  }

  /**
   * Arguments that intersperse strings between the items in a sequence. There are two forms of
   * interspersing, and either may be used by this implementation:
   * <ul>
   *   <li>before each - a string is added before each item in a sequence. e.g.
   *       {@code -f foo -f bar -f baz}
   *   <li>format each - a format string is used to format each item in a sequence. e.g.
   *       {@code -I/foo -I/bar -I/baz} for the format {@code "-I%s"}
   * </ul>
   *
   * <p>This class could be used both with both the "before" and "format" features at the same
   * time, but this is probably more confusion than it is worth. If you need this functionality,
   * consider using "before" only but storing the strings pre-formated in a {@link NestedSet}.
   */
  private static final class InterspersingArgs extends ArgvFragment {
    private final Iterable<?> sequence;
    private final String beforeEach;
    private final String formatEach;

    /**
     * Do not call from outside this class because this does not guarantee that {@code sequence} is
     * immutable.
     */
    private InterspersingArgs(Iterable<?> sequence, String beforeEach, String formatEach) {
      this.sequence = sequence;
      this.beforeEach = beforeEach;
      this.formatEach = formatEach;
    }

    static InterspersingArgs fromStrings(
        Iterable<?> sequence, String beforeEach, String formatEach) {
      return new InterspersingArgs(
          CollectionUtils.makeImmutable(sequence), beforeEach, formatEach);
    }

    static InterspersingArgs fromExecPaths(
        Iterable<Artifact> sequence, String beforeEach, String formatEach) {
      return new InterspersingArgs(
          Artifact.toExecPaths(CollectionUtils.makeImmutable(sequence)), beforeEach, formatEach);
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      for (Object item : sequence) {
        if (item == null) {
          continue;
        }

        if (beforeEach != null) {
          builder.add(beforeEach);
        }
        String arg = item.toString();
        if (formatEach != null) {
          arg = String.format(formatEach, arg);
        }
        builder.add(arg);
      }
    }
  }

  /**
   * A Builder class for CustomCommandLine with the appropriate methods.
   *
   * <p>{@link Iterable} instances passed to {@code add*} methods will be stored internally as
   * collections that are known to be immutable copies. This means that any {@link Iterable} that is
   * not a {@link NestedSet} or {@link ImmutableList} may be copied.
   *
   * <p>{@code addFormatEach*} methods take an {@link Iterable} but use these as arguments to
   * {@link String#format(String, Object...)} with a certain constant format string. For instance,
   * if {@code format} is {@code "-I%s"}, then the final arguments may be
   * {@code -Ifoo -Ibar -Ibaz}
   *
   * <p>{@code addBeforeEach*} methods take an {@link Iterable} but insert a certain {@link String}
   * once before each element in the string, meaning the total number of elements added is twice the
   * length of the {@link Iterable}. For instance: {@code -f foo -f bar -f baz}
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
        arguments.add(InterspersingArgs.fromStrings(args, /*beforeEach=*/null, "%s"));
      }
      return this;
    }

    public Builder add(Iterable<String> args) {
      if (args != null) {
        arguments.add(InterspersingArgs.fromStrings(args, /*beforeEach=*/null, "%s"));
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
        arguments.add(InterspersingArgs.fromExecPaths(artifacts, /*beforeEach=*/null, "%s"));
      }
      return this;
    }

    public Builder addExecPaths(Iterable<Artifact> artifacts) {
      if (artifacts != null) {
        arguments.add(InterspersingArgs.fromExecPaths(artifacts, /*beforeEach=*/null, "%s"));
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

    public Builder addBeforeEachPath(String repeated, Iterable<PathFragment> paths) {
      if (repeated != null && paths != null) {
        arguments.add(InterspersingArgs.fromStrings(paths, repeated, "%s"));
      }
      return this;
    }

    public Builder addBeforeEach(String repeated, Iterable<String> strings) {
      if (repeated != null && strings != null) {
        arguments.add(InterspersingArgs.fromStrings(strings, repeated, "%s"));
      }
      return this;
    }

    public Builder addBeforeEachExecPath(String repeated, Iterable<Artifact> artifacts) {
      if (repeated != null && artifacts != null) {
        arguments.add(InterspersingArgs.fromExecPaths(artifacts, repeated, "%s"));
      }
      return this;
    }

    public Builder addFormatEach(String format, Iterable<String> strings) {
      if (format != null && strings != null) {
        arguments.add(InterspersingArgs.fromStrings(strings, /*beforeEach=*/null, format));
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
