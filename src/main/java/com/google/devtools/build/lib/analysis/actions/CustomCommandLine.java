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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import javax.annotation.Nullable;

/**
 * A customizable, serializable class for building memory efficient command lines.
 */
@Immutable
public final class CustomCommandLine extends CommandLine {

  private abstract static class ArgvFragment {
    abstract void eval(ImmutableList.Builder<String> builder);
  }

  /**
   * A command line argument for {@link TreeFileArtifact}.
   *
   * <p>Since {@link TreeFileArtifact} is not known or available at analysis time, subclasses should
   * enclose its parent TreeFileArtifact instead at analysis time. This interface provides method
   * {@link #substituteTreeArtifact} to generate another argument object that replaces the enclosed
   * TreeArtifact with one of its {@link TreeFileArtifact} at execution time.
   */
  private abstract static class TreeFileArtifactArgvFragment {
    /**
     * Substitutes this ArgvFragment with another arg object, with the original TreeArtifacts
     * contained in this ArgvFragment replaced by their associated TreeFileArtifacts.
     *
     * @param substitutionMap A map between TreeArtifacts and their associated TreeFileArtifacts
     *     used to replace them.
     */
    abstract Object substituteTreeArtifact(Map<Artifact, TreeFileArtifact> substitutionMap);
  }

  /**
   * A command line argument that can expand enclosed TreeArtifacts into a list of child
   * {@link TreeFileArtifact}s at execution time before argument evaluation.
   *
   * <p>The main difference between this class and {@link TreeFileArtifactArgvFragment} is that
   * {@link TreeFileArtifactArgvFragment} is used in {@link SpawnActionTemplate} to substitutes a
   * TreeArtifact with *one* of its child TreeFileArtifacts, while this class expands a TreeArtifact
   * into *all* of its child TreeFileArtifacts.
   *
   */
  private abstract static class TreeArtifactExpansionArgvFragment extends ArgvFragment {
    /**
     * Evaluates this argument fragment into an argument string and adds it into {@code builder}.
     * The enclosed TreeArtifact will be expanded using {@code artifactExpander}.
     */
    abstract void eval(ImmutableList.Builder<String> builder, ArtifactExpander artifactExpander);

    /**
     * Returns a string that describes this argument fragment. The string can be used as part of
     * an action key for the command line at analysis time.
     */
    abstract String describe();

    /**
     * Evaluates this argument fragment by serializing it into a string. Note that the returned
     * argument is not suitable to be used as part of an actual command line. The purpose of this
     * method is to provide a unique command line argument string to be used as part of an action
     * key at analysis time.
     *
     * <p>Internally this method just calls {@link #describe}.
     */
    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(describe());
    }
  }

  // It's better to avoid anonymous classes if we want to serialize command lines
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

  private static final class JoinExpandedTreeArtifactExecPathsArg
      extends TreeArtifactExpansionArgvFragment {

    private final String delimiter;
    private final Artifact treeArtifact;

    private JoinExpandedTreeArtifactExecPathsArg(String delimiter, Artifact treeArtifact) {
      Preconditions.checkArgument(
          treeArtifact.isTreeArtifact(), "%s is not a TreeArtifact", treeArtifact);
      this.delimiter = delimiter;
      this.treeArtifact = treeArtifact;
    }

    @Override
    void eval(ImmutableList.Builder<String> builder, ArtifactExpander artifactExpander) {
      Set<Artifact> expandedArtifacts = new TreeSet<>();
      artifactExpander.expand(treeArtifact, expandedArtifacts);

      if (!expandedArtifacts.isEmpty()) {
        builder.add(Artifact.joinExecPaths(delimiter, expandedArtifacts));
      }
    }

    @Override
    public String describe() {
      return String.format(
          "JoinExpandedTreeArtifactExecPathsArg{ delimiter: %s, treeArtifact: %s}",
          delimiter,
          treeArtifact.getExecPathString());
    }
  }

  private static final class ExpandedTreeArtifactExecPathsArg
      extends TreeArtifactExpansionArgvFragment {
    private final Artifact treeArtifact;

    private ExpandedTreeArtifactExecPathsArg(Artifact treeArtifact) {
      Preconditions.checkArgument(
          treeArtifact.isTreeArtifact(), "%s is not a TreeArtifact", treeArtifact);
      this.treeArtifact = treeArtifact;
    }

    @Override
    void eval(ImmutableList.Builder<String> builder, ArtifactExpander artifactExpander) {
      Set<Artifact> expandedArtifacts = new TreeSet<>();
      artifactExpander.expand(treeArtifact, expandedArtifacts);

      for (Artifact expandedArtifact : expandedArtifacts) {
        builder.add(expandedArtifact.getExecPathString());
      }
    }

    @Override
    public String describe() {
      return String.format(
          "ExpandedTreeArtifactExecPathsArg{ treeArtifact: %s}",
          treeArtifact.getExecPathString());
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

  private static final class ParamFileArgument extends ArgvFragment {
    private final String paramFilePrefix;
    private final PathFragment path;

    private ParamFileArgument(String paramFilePrefix, PathFragment path) {
      this.paramFilePrefix = paramFilePrefix;
      this.path = path;
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(paramFilePrefix + path);
    }
  }

  /**
   * An argument object that evaluates to a formatted string for {@link TreeFileArtifact} exec
   * paths, enclosing the associated string format template and {@link TreeFileArtifact}s.
   */
  private static final class TreeFileArtifactExecPathWithTemplateArg
      extends TreeFileArtifactArgvFragment {

    private final String template;
    private final Artifact placeHolderTreeArtifact;

    private TreeFileArtifactExecPathWithTemplateArg(String template, Artifact artifact) {
      Preconditions.checkArgument(artifact.isTreeArtifact(), "%s must be a TreeArtifact",
          artifact);
      this.template = template;
      this.placeHolderTreeArtifact = artifact;
    }

    @Override
    ArgvFragment substituteTreeArtifact(Map<Artifact, TreeFileArtifact> substitutionMap) {
      Artifact treeFileArtifact = substitutionMap.get(placeHolderTreeArtifact);
      Preconditions.checkNotNull(treeFileArtifact, "Artifact to substitute: %s",
          placeHolderTreeArtifact);

      return new PathWithTemplateArg(template, treeFileArtifact.getExecPath());
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

  private static final class JoinStringsArg extends ArgvFragment {

    private final String delimiter;
    private final Iterable<String> strings;

    private JoinStringsArg(String delimiter, Iterable<String> strings) {
      this.delimiter = delimiter;
      this.strings = CollectionUtils.makeImmutable(strings);
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.add(Joiner.on(delimiter).join(strings));
    }
  }

  private static final class JoinValuesTransformed<T> extends ArgvFragment {

    private final String delimiter;
    private final Iterable<T> values;
    private final Function<T, String> toString;

    private JoinValuesTransformed(
        String delimiter, Iterable<T> values, Function<T, String> toString) {
      this.delimiter = delimiter;
      this.values = CollectionUtils.makeImmutable(values);
      this.toString = toString;
    }

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      StringBuilder arg = new StringBuilder();
      Iterator<T> parts = values.iterator();
      if (parts.hasNext()) {
        arg.append(toString.apply(parts.next()));
        while (parts.hasNext()) {
          arg.append(delimiter);
          arg.append(toString.apply(parts.next()));
        }
      }
      builder.add(arg.toString());
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
   * consider using "before" only but storing the strings pre-formatted in a {@link NestedSet}.
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
   * An argument object that evaluates to the exec path of a {@link TreeFileArtifact}, enclosing
   * the associated {@link TreeFileArtifact}.
   */
  private static final class TreeFileArtifactExecPathArg extends TreeFileArtifactArgvFragment {
    private final Artifact placeHolderTreeArtifact;

    private TreeFileArtifactExecPathArg(Artifact artifact) {
      Preconditions.checkArgument(artifact.isTreeArtifact(), "%s must be a TreeArtifact", artifact);
      placeHolderTreeArtifact = artifact;
    }

    @Override
    Object substituteTreeArtifact(Map<Artifact, TreeFileArtifact> substitutionMap) {
      Artifact artifact = substitutionMap.get(placeHolderTreeArtifact);
      Preconditions.checkNotNull(artifact, "Artifact to substitute: %s", placeHolderTreeArtifact);
      return artifact.getExecPath();
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
    // In order to avoid unnecessary wrapping, we keep raw objects here, but these objects are
    // always either ArgvFragments or objects whose desired string representations are just their
    // toString() results.
    private final List<Object> arguments = new ArrayList<>();

    public Builder add(@Nullable CharSequence arg) {
      if (arg != null) {
        arguments.add(arg);
      }
      return this;
    }

    public Builder add(@Nullable Label arg) {
      if (arg != null) {
        arguments.add(arg);
      }
      return this;
    }

    public Builder add(String arg, @Nullable Iterable<String> args) {
      Preconditions.checkNotNull(arg);
      if (args != null && !Iterables.isEmpty(args)) {
        arguments.add(arg);
        arguments.add(
            InterspersingArgs.fromStrings(args, /*beforeEach=*/ null, /*formatEach=*/ null));
      }
      return this;
    }

    public Builder add(@Nullable Iterable<String> args) {
      if (args != null && !Iterables.isEmpty(args)) {
        arguments.add(
            InterspersingArgs.fromStrings(args, /*beforeEach=*/ null, /*formatEach=*/ null));
      }
      return this;
    }

    public Builder addExecPath(String arg, @Nullable Artifact artifact) {
      Preconditions.checkNotNull(arg);
      if (artifact != null) {
        arguments.add(arg);
        arguments.add(artifact.getExecPath());
      }
      return this;
    }

    public Builder addExecPaths(String arg, @Nullable Iterable<Artifact> artifacts) {
      Preconditions.checkNotNull(arg);
      if (artifacts != null && !Iterables.isEmpty(artifacts)) {
        arguments.add(arg);
        arguments.add(
            InterspersingArgs.fromExecPaths(artifacts, /*beforeEach=*/ null, /*formatEach=*/ null));
      }
      return this;
    }

    public Builder addExecPaths(@Nullable Iterable<Artifact> artifacts) {
      if (artifacts != null && !Iterables.isEmpty(artifacts)) {
        arguments.add(
            InterspersingArgs.fromExecPaths(artifacts, /*beforeEach=*/ null, /*formatEach=*/ null));
      }
      return this;
    }

    /**
     * Adds a flag with the exec path of a placeholder TreeArtifact. When the command line is used
     * in an action template, the placeholder will be replaced by the exec path of a {@link
     * TreeFileArtifact} inside the TreeArtifact at execution time for each expanded action.
     *
     * @param arg the name of the argument
     * @param treeArtifact the TreeArtifact that will be evaluated to one of its child {@link
     *     TreeFileArtifact} at execution time
     */
    public Builder addPlaceholderTreeArtifactExecPath(String arg, @Nullable Artifact treeArtifact) {
      Preconditions.checkNotNull(arg);
      if (treeArtifact != null) {
        arguments.add(arg);
        arguments.add(new TreeFileArtifactExecPathArg(treeArtifact));
      }
      return this;
    }

    /**
     * Adds a placeholder TreeArtifact exec path. When the command line is used in an action
     * template, the placeholder will be replaced by the exec path of a {@link TreeFileArtifact}
     * inside the TreeArtifact at execution time for each expanded action.
     *
     * @param treeArtifact the TreeArtifact that will be evaluated to one of its child {@link
     *     TreeFileArtifact} at execution time
     */
    public Builder addPlaceholderTreeArtifactExecPath(@Nullable Artifact treeArtifact) {
      if (treeArtifact != null) {
        arguments.add(new TreeFileArtifactExecPathArg(treeArtifact));
      }
      return this;
    }

    public Builder addJoinStrings(
        String arg, String delimiter, @Nullable Iterable<String> strings) {
      Preconditions.checkNotNull(arg);
      Preconditions.checkNotNull(delimiter);
      if (strings != null && !Iterables.isEmpty(strings)) {
        arguments.add(arg);
        arguments.add(new JoinStringsArg(delimiter, strings));
      }
      return this;
    }

    /**
     * Adds a list of values transformed by a function and delimited by a string.
     *
     * <p>Prefer this to transforming nested sets yourself as it is more memory-efficient. By using
     * this class, expansion of the nested set is deferred until action execution instead of
     * retained on the heap.
     *
     * @param arg The argument
     * @param delimiter A delimiter string placed in between each transformed value
     * @param values The values to expand into a list
     * @param toString A function that transforms a value into a string
     */
    public <T> Builder addJoinValues(
        String arg, String delimiter, @Nullable Iterable<T> values, Function<T, String> toString) {
      Preconditions.checkNotNull(arg);
      Preconditions.checkNotNull(delimiter);
      Preconditions.checkNotNull(toString);
      if (values != null && !Iterables.isEmpty(values)) {
        arguments.add(arg);
        arguments.add(new JoinValuesTransformed<T>(delimiter, values, toString));
      }
      return this;
    }

    public Builder addJoinExecPaths(
        String arg, String delimiter, @Nullable Iterable<Artifact> artifacts) {
      Preconditions.checkNotNull(arg);
      Preconditions.checkNotNull(delimiter);
      if (artifacts != null && !Iterables.isEmpty(artifacts)) {
        arguments.add(arg);
        arguments.add(new JoinExecPathsArg(delimiter, artifacts));
      }
      return this;
    }

    public Builder addPath(@Nullable PathFragment path) {
      if (path != null) {
        arguments.add(path);
      }
      return this;
    }

    public Builder addPaths(String template, @Nullable PathFragment... path) {
      Preconditions.checkNotNull(template);
      if (path != null) {
        arguments.add(new PathWithTemplateArg(template, path));
      }
      return this;
    }

    /**
     * Adds a param file as an argument.
     *
     * @param paramFilePrefix The character that denotes a param file, commonly '@'
     * @param paramFile The param file artifact
     */
    public Builder addParamFile(String paramFilePrefix, @Nullable Artifact paramFile) {
      Preconditions.checkNotNull(paramFilePrefix);
      Preconditions.checkNotNull(paramFile);
      arguments.add(new ParamFileArgument(paramFilePrefix, paramFile.getExecPath()));
      return this;
    }

    /**
     * Adds a formatted string containing the exec path of a placeholder TreeArtifact. When the
     * command line is used in an action template, the placeholder will be replaced by the exec path
     * of a {@link TreeFileArtifact} inside the TreeArtifact at execution time for each expanded
     * action.
     *
     * @param template the string format template containing a single string format specifier (%s)
     *     to be replaced by the artifact exec path string.
     * @param treeArtifact the TreeArtifact that will be evaluated to one of their child {@link
     *     TreeFileArtifact} at execution time
     */
    public Builder addPlaceholderTreeArtifactFormattedExecPath(
        String template, @Nullable Artifact treeArtifact) {
      Preconditions.checkNotNull(template);
      if (treeArtifact != null) {
        arguments.add(new TreeFileArtifactExecPathWithTemplateArg(template, treeArtifact));
      }
      return this;
    }

    public Builder addJoinPaths(String delimiter, @Nullable Iterable<PathFragment> paths) {
      Preconditions.checkNotNull(delimiter);
      if (paths != null && !Iterables.isEmpty(paths)) {
        arguments.add(new JoinPathsArg(delimiter, paths));
      }
      return this;
    }

    /**
     * Adds a string joined together by the exec paths of all {@link TreeFileArtifact}s under
     * {@code treeArtifact}.
     *
     * @param delimiter the delimiter used to join the artifact exec paths.
     * @param treeArtifact the TreeArtifact containing the {@link TreeFileArtifact}s to join.
     */
    public Builder addJoinExpandedTreeArtifactExecPath(String delimiter, Artifact treeArtifact) {
      Preconditions.checkNotNull(delimiter);
      Preconditions.checkNotNull(treeArtifact);
      arguments.add(new JoinExpandedTreeArtifactExecPathsArg(delimiter, treeArtifact));
      return this;
    }

    /**
     * Adds the exec paths (one argument per exec path) of all {@link TreeFileArtifact}s under
     * {@code treeArtifact}.
     *
     * @param treeArtifact the TreeArtifact containing the {@link TreeFileArtifact}s to add.
     */
    public Builder addExpandedTreeArtifactExecPaths(Artifact treeArtifact) {
      Preconditions.checkNotNull(treeArtifact);
      arguments.add(new ExpandedTreeArtifactExecPathsArg(treeArtifact));
      return this;
    }

    public Builder addBeforeEachPath(String repeated, @Nullable Iterable<PathFragment> paths) {
      Preconditions.checkNotNull(repeated);
      if (paths != null && !Iterables.isEmpty(paths)) {
        arguments.add(InterspersingArgs.fromStrings(paths, repeated, /*formatEach=*/ null));
      }
      return this;
    }

    public Builder addBeforeEach(String repeated, @Nullable Iterable<String> strings) {
      Preconditions.checkNotNull(repeated);
      if (strings != null && !Iterables.isEmpty(strings)) {
        arguments.add(InterspersingArgs.fromStrings(strings, repeated, /*formatEach=*/ null));
      }
      return this;
    }

    public Builder addBeforeEachExecPath(String repeated, @Nullable Iterable<Artifact> artifacts) {
      Preconditions.checkNotNull(repeated);
      if (artifacts != null && !Iterables.isEmpty(artifacts)) {
        arguments.add(InterspersingArgs.fromExecPaths(artifacts, repeated, /*formatEach=*/ null));
      }
      return this;
    }

    public Builder addFormatEach(String format, @Nullable Iterable<String> strings) {
      Preconditions.checkNotNull(format);
      if (strings != null && !Iterables.isEmpty(strings)) {
        arguments.add(InterspersingArgs.fromStrings(strings, /*beforeEach=*/null, format));
      }
      return this;
    }

    public Builder add(@Nullable CustomArgv arg) {
      if (arg != null) {
        arguments.add(arg);
      }
      return this;
    }

    public Builder add(@Nullable CustomMultiArgv arg) {
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

  private final ImmutableList<Object> arguments;


  /**
   * A map between enclosed TreeArtifacts and their associated {@link TreeFileArtifacts} for
   * substitution.
   *
   * <p> This map is used to support TreeArtifact substitutions in
   * {@link TreeFileArtifactArgvFragment}s.
   */
  private final Map<Artifact, TreeFileArtifact> substitutionMap;

  private CustomCommandLine(List<Object> arguments) {
    this.arguments = ImmutableList.copyOf(arguments);
    this.substitutionMap = null;
  }

  private CustomCommandLine(
      List<Object> arguments, Map<Artifact, TreeFileArtifact> substitutionMap) {
    this.arguments = ImmutableList.copyOf(arguments);
    this.substitutionMap = ImmutableMap.copyOf(substitutionMap);
  }

  /**
   * Given the list of {@link TreeFileArtifact}s, returns another CustomCommandLine that replaces
   * their parent TreeArtifacts with the TreeFileArtifacts in all
   * {@link TreeFileArtifactArgvFragment} argument objects.
   */
  @VisibleForTesting
  public CustomCommandLine evaluateTreeFileArtifacts(Iterable<TreeFileArtifact> treeFileArtifacts) {
    ImmutableMap.Builder<Artifact, TreeFileArtifact> substitutionMap = ImmutableMap.builder();
    for (TreeFileArtifact treeFileArtifact : treeFileArtifacts) {
      substitutionMap.put(treeFileArtifact.getParent(), treeFileArtifact);
    }

    return new CustomCommandLine(arguments, substitutionMap.build());
  }

  @Override
  public Iterable<String> arguments() {
    return argumentsInternal(null);
  }

  @Override
  public Iterable<String> arguments(ArtifactExpander artifactExpander) {
    return argumentsInternal(Preconditions.checkNotNull(artifactExpander));
  }

  private Iterable<String> argumentsInternal(@Nullable ArtifactExpander artifactExpander) {
    ImmutableList.Builder<String> builder = ImmutableList.builder();
    for (Object arg : arguments) {
      Object substitutedArg = substituteTreeFileArtifactArgvFragment(arg);
      if (substitutedArg instanceof ArgvFragment) {
        if (artifactExpander != null
            && substitutedArg instanceof TreeArtifactExpansionArgvFragment) {
          TreeArtifactExpansionArgvFragment expansionArg =
              (TreeArtifactExpansionArgvFragment) substitutedArg;
          expansionArg.eval(builder, artifactExpander);
        } else {
          ((ArgvFragment) substitutedArg).eval(builder);
        }
      } else {
        builder.add(substitutedArg.toString());
      }
    }
    return builder.build();
  }

  /**
   * If the given arg is a {@link TreeFileArtifactArgvFragment} and we have its associated
   * TreeArtifact substitution map, returns another argument object that has its enclosing
   * TreeArtifact substituted by one of its {@link TreeFileArtifact}. Otherwise, returns the given
   * arg unmodified.
   */
  private Object substituteTreeFileArtifactArgvFragment(Object arg) {
    if (arg instanceof TreeFileArtifactArgvFragment) {
      TreeFileArtifactArgvFragment argvFragment = (TreeFileArtifactArgvFragment) arg;
      return argvFragment.substituteTreeArtifact(
          Preconditions.checkNotNull(substitutionMap, argvFragment));
    } else {
      return arg;
    }
  }
}
