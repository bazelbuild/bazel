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
import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CompileTimeConstant;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * A customizable, serializable class for building memory efficient command lines.
 */
@Immutable
public final class CustomCommandLine extends CommandLine {

  private interface ArgvFragment {
    /**
     * Expands this fragment into the passed command line vector.
     *
     * @param arguments The command line's argument vector.
     * @param argi The index of the next available argument.
     * @param builder The command line builder to which we should add arguments.
     * @return The index of the next argument, after the ArgvFragment has consumed its args.
     *   If the ArgvFragment doesn't have any args, it should return {@code argi} unmodified.
     */
    int eval(List<Object> arguments, int argi, ImmutableList.Builder<String> builder);
  }

  /**
   * Helper base class for an ArgvFragment that doesn't use the input argument vector.
   *
   * <p>This can be used for any ArgvFragments that self-contain all the necessary state.
   */
  private abstract static class StandardArgvFragment implements ArgvFragment {
    @Override
    public final int eval(List<Object> arguments, int argi, ImmutableList.Builder<String> builder) {
      eval(builder);
      return argi; // Doesn't consume any arguments, so return argi unmodified
    }

    abstract void eval(ImmutableList.Builder<String> builder);
  }

  // TODO(bazel-team): CustomMultiArgv is  going to be difficult to expose
  // in Skylark. Maybe we can get rid of them by refactoring JavaCompileAction. It also
  // raises immutability / serialization issues.
  /**
   * Custom Java code producing a List of String arguments.
   */
  public abstract static class CustomMultiArgv extends StandardArgvFragment {

    @Override
    void eval(ImmutableList.Builder<String> builder) {
      builder.addAll(argv());
    }

    public abstract Iterable<String> argv();
  }

  /**
   * An ArgvFragment that expands a collection of objects in a user-specified way.
   *
   * <p>Used to cut down on code duplication between the many overloads of add*.
   */
  private static final class VectorArg implements ArgvFragment {
    private static Interner<VectorArg> interner = BlazeInterners.newStrongInterner();

    private final boolean isNestedSet;
    private final boolean hasMapEach;
    private final boolean hasFormatEach;
    private final boolean hasBeforeEach;
    private final boolean hasJoinWith;

    private VectorArg(
        boolean isNestedSet,
        boolean hasMapEach,
        boolean hasFormatEach,
        boolean hasBeforeEach,
        boolean hasJoinWith) {
      this.isNestedSet = isNestedSet;
      this.hasMapEach = hasMapEach;
      this.hasFormatEach = hasFormatEach;
      this.hasBeforeEach = hasBeforeEach;
      this.hasJoinWith = hasJoinWith;
    }

    private static void push(List<Object> arguments, Builder<?> argv) {
      VectorArg vectorArg =
          new VectorArg(
              argv.isNestedSet,
              argv.mapFn != null,
              argv.formatEach != null,
              argv.beforeEach != null,
              argv.joinWith != null);
      vectorArg = interner.intern(vectorArg);
      arguments.add(vectorArg);
      if (vectorArg.isNestedSet) {
        arguments.add(argv.values);
      } else {
        // Simply expand any ordinary collection into the argv
        arguments.add(argv.count);
        Iterables.addAll(arguments, argv.values);
      }
      if (vectorArg.hasMapEach) {
        arguments.add(argv.mapFn);
      }
      if (vectorArg.hasFormatEach) {
        arguments.add(argv.formatEach);
      }
      if (vectorArg.hasBeforeEach) {
        arguments.add(argv.beforeEach);
      }
      if (vectorArg.hasJoinWith) {
        arguments.add(argv.joinWith);
      }
    }

    @SuppressWarnings("unchecked")
    @Override
    public int eval(List<Object> arguments, int argi, ImmutableList.Builder<String> builder) {
      final List<Object> mutatedValues;
      final int count;
      if (isNestedSet) {
        Iterable<Object> values = (Iterable<Object>) arguments.get(argi++);
        mutatedValues = Lists.newArrayList(values);
        count = mutatedValues.size();
      } else {
        count = (Integer) arguments.get(argi++);
        mutatedValues = new ArrayList<>(count);
        for (int i = 0; i < count; ++i) {
          mutatedValues.add(arguments.get(argi++));
        }
      }
      if (hasMapEach) {
        Function<Object, String> mapFn = (Function<Object, String>) arguments.get(argi++);
        for (int i = 0; i < count; ++i) {
          mutatedValues.set(i, mapFn.apply(mutatedValues.get(i)));
        }
      }
      for (int i = 0; i < count; ++i) {
        mutatedValues.set(i, valueToString(mutatedValues.get(i)));
      }
      if (hasFormatEach) {
        String formatStr = (String) arguments.get(argi++);
        for (int i = 0; i < count; ++i) {
          mutatedValues.set(i, String.format(formatStr, mutatedValues.get(i)));
        }
      }
      if (hasBeforeEach) {
        String beforeEach = (String) arguments.get(argi++);
        for (int i = 0; i < count; ++i) {
          builder.add(beforeEach);
          builder.add((String) mutatedValues.get(i));
        }
      } else if (hasJoinWith) {
        String joinWith = (String) arguments.get(argi++);
        builder.add(Joiner.on(joinWith).join(mutatedValues));
      } else {
        for (int i = 0; i < count; ++i) {
          builder.add((String) mutatedValues.get(i));
        }
      }
      return argi;
    }

    static <T> Builder<T> of(@Nullable Collection<T> values) {
      return new Builder<>(values);
    }

    static <T> Builder<T> of(@Nullable NestedSet<T> values) {
      return new Builder<>(values);
    }

    /** Builder for a VectorArg */
    public static class Builder<T> {
      @Nullable private final Iterable<T> values;
      private final boolean isEmpty;
      private final boolean isNestedSet;
      private final int count;
      private String formatEach;
      private String beforeEach;
      private Function<T, String> mapFn;
      private String joinWith;

      private Builder(@Nullable Collection<T> values) {
        this(values, values == null || values.isEmpty(), false, values != null ? values.size() : 0);
      }

      private Builder(@Nullable NestedSet<T> values) {
        this(values, values == null || values.isEmpty(), true, -1);
      }

      private Builder(
          @Nullable Iterable<T> values, boolean isEmpty, boolean isNestedSet, int count) {
        this.values = values;
        this.isEmpty = isEmpty;
        this.isNestedSet = isNestedSet;
        this.count = count;
      }

      /** Each argument is formatted via {@link String#format}. */
      public Builder<T> formatEach(@CompileTimeConstant String formatEach) {
        Preconditions.checkNotNull(formatEach);
        this.formatEach = formatEach;
        return this;
      }

      /** Each argument is prepended by the beforeEach param. */
      public Builder<T> beforeEach(@CompileTimeConstant String beforeEach) {
        Preconditions.checkNotNull(beforeEach);
        this.beforeEach = beforeEach;
        return this;
      }

      /** Each argument is mapped using the supplied map function */
      @SuppressWarnings("unchecked")
      public Builder<String> mapEach(Function<T, String> mapFn) {
        Preconditions.checkNotNull(mapFn);
        this.mapFn = mapFn;

        // To save on GC we can simply cast the object and return it.
        return (Builder<String>) this;
      }

      /** Once all arguments have been evaluated, they are joined with this delimiter */
      public Builder<T> joinWith(String delimiter) {
        Preconditions.checkNotNull(delimiter);
        this.joinWith = delimiter;
        return this;
      }

    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      VectorArg vectorArg = (VectorArg) o;
      return isNestedSet == vectorArg.isNestedSet
          && hasMapEach == vectorArg.hasMapEach
          && hasFormatEach == vectorArg.hasFormatEach
          && hasBeforeEach == vectorArg.hasBeforeEach
          && hasJoinWith == vectorArg.hasJoinWith;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(isNestedSet, hasMapEach, hasFormatEach, hasBeforeEach, hasJoinWith);
    }
  }

  private static class FormatArg implements ArgvFragment {
    private static final FormatArg INSTANCE = new FormatArg();

    private static void push(List<Object> arguments, String formatStr, Object... args) {
      arguments.add(INSTANCE);
      arguments.add(args.length);
      arguments.add(formatStr);
      Collections.addAll(arguments, args);
    }

    @Override
    public int eval(List<Object> arguments, int argi, ImmutableList.Builder<String> builder) {
      int argCount = (Integer) arguments.get(argi++);
      String formatStr = (String) arguments.get(argi++);
      Object[] args = new Object[argCount];
      for (int i = 0; i < argCount; ++i) {
        args[i] = valueToString(arguments.get(argi++));
      }
      builder.add(String.format(formatStr, args));
      return argi;
    }
  }

  private static class PrefixArg implements ArgvFragment {
    private static final PrefixArg INSTANCE = new PrefixArg();

    private static void push(List<Object> arguments, String before, Object arg) {
      arguments.add(INSTANCE);
      arguments.add(before);
      arguments.add(arg);
    }

    @Override
    public int eval(List<Object> arguments, int argi, ImmutableList.Builder<String> builder) {
      String before = (String) arguments.get(argi++);
      Object arg = arguments.get(argi++);
      builder.add(before + valueToString(arg));
      return argi;
    }
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
   * A command line argument that can expand enclosed TreeArtifacts into a list of child {@link
   * TreeFileArtifact}s at execution time before argument evaluation.
   *
   * <p>The main difference between this class and {@link TreeFileArtifactArgvFragment} is that
   * {@link TreeFileArtifactArgvFragment} is used in {@link SpawnActionTemplate} to substitutes a
   * TreeArtifact with *one* of its child TreeFileArtifacts, while this class expands a TreeArtifact
   * into *all* of its child TreeFileArtifacts.
   */
  private abstract static class TreeArtifactExpansionArgvFragment extends StandardArgvFragment {
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
   * <p>{@link Collection} instances passed to {@code add*} methods will copied internally. If you
   * have a {@link NestedSet}, these should never be flattened to a collection before being passed
   * to the command line.
   *
   * <p>{@code addFormatEach*} methods take a {@link Collection} or {@link NestedSet} but use these
   * as arguments to {@link String#format(String, Object...)} with a certain constant format string.
   * For instance, if {@code format} is {@code "-I%s"}, then the final arguments may be {@code -Ifoo
   * -Ibar -Ibaz}
   *
   * <p>{@code addBeforeEach*} methods take a {@link Collection} or {@link NestedSet } but insert a
   * certain {@link String} once before each element in the string, meaning the total number of
   * elements added is twice the length of the {@link Iterable}. For instance: {@code -f foo -f bar
   * -f baz}
   */
  public static final class Builder {
    // In order to avoid unnecessary wrapping, we keep raw objects here, but these objects are
    // always either ArgvFragments or objects whose desired string representations are just their
    // toString() results.
    private final List<Object> arguments = new ArrayList<>();

    public boolean isEmpty() {
      return arguments.isEmpty();
    }

    /**
     * Adds a constant-value string.
     *
     * <p>Prefer this over its dynamic cousin, as using static strings saves memory.
     */
    public Builder add(@CompileTimeConstant String value) {
      return addObjectInternal(value);
    }

    /**
     * Adds a dynamically calculated string.
     *
     * <p>Consider whether using another method could be more efficient. For instance, rather than
     * calling this method with an Artifact's exec path, just add the artifact itself. It will
     * lazily get converted to its exec path. Same with labels, path fragments, and many other
     * objects.
     *
     * <p>If you are joining some list into a single argument, consider using {@link VectorArg}.
     *
     * <p>If you are formatting a string, consider using {@link Builder#addFormatted(String,
     * Object...)}.
     *
     * <p>There are many other ways you can try to avoid calling this. In general, try to use
     * constants or objects that are already on the heap elsewhere.
     */
    public Builder addDynamicString(@Nullable String value) {
      return addObjectInternal(value);
    }

    /**
     * Adds a label value by calling {@link Label#getCanonicalForm}.
     *
     * <p>Prefer this over manually calling {@link Label#getCanonicalForm}, as it avoids a copy of
     * the label value.
     */
    public Builder addLabel(@Nullable Label value) {
      return addObjectInternal(value);
    }

    /**
     * Adds an artifact by calling {@link PathFragment#getPathString}.
     *
     * <p>Prefer this over manually calling {@link PathFragment#getPathString}, as it avoids storing
     * a copy of the path string.
     */
    public Builder addPath(@Nullable PathFragment value) {
      return addObjectInternal(value);
    }

    /**
     * Adds an artifact by calling {@link Artifact#getExecPath}.
     *
     * <p>Prefer this over manually calling {@link Artifact#getExecPath}, as it avoids storing a
     * copy of the artifact path string.
     */
    public Builder addExecPath(@Nullable Artifact value) {
      return addObjectInternal(value);
    }

    /** Adds a lazily expanded string. */
    public Builder addLazyString(@Nullable LazyString value) {
      return addObjectInternal(value);
    }

    /**
     * Adds a string argument to the command line.
     *
     * <p>If the value is null, neither the arg nor the value is added.
     */
    public Builder add(@CompileTimeConstant String arg, @Nullable String value) {
      return addObjectInternal(arg, value);
    }

    /**
     * Adds a label value by calling {@link Label#getCanonicalForm}.
     *
     * <p>Prefer this over manually calling {@link Label#getCanonicalForm}, as it avoids storing a
     * copy of the label value.
     *
     * <p>If the value is null, neither the arg nor the value is added.
     */
    public Builder addLabel(@CompileTimeConstant String arg, @Nullable Label value) {
      return addObjectInternal(arg, value);
    }

    /**
     * Adds an artifact by calling {@link PathFragment#getPathString}.
     *
     * <p>Prefer this over manually calling {@link PathFragment#getPathString}, as it avoids storing
     * a copy of the path string.
     *
     * <p>If the value is null, neither the arg nor the value is added.
     */
    public Builder addPath(@CompileTimeConstant String arg, @Nullable PathFragment value) {
      return addObjectInternal(arg, value);
    }

    /**
     * Adds an artifact by calling {@link Artifact#getExecPath}.
     *
     * <p>Prefer this over manually calling {@link Artifact#getExecPath}, as it avoids storing a
     * copy of the artifact path string.
     *
     * <p>If the value is null, neither the arg nor the value is added.
     */
    public Builder addExecPath(@CompileTimeConstant String arg, @Nullable Artifact value) {
      return addObjectInternal(arg, value);
    }

    /** Adds a lazily expanded string. */
    public Builder addLazyString(@CompileTimeConstant String arg, @Nullable LazyString value) {
      return addObjectInternal(arg, value);
    }

    /** Calls {@link String#format} at command line expansion time. */
    @FormatMethod
    public Builder addFormatted(@FormatString String formatStr, Object... args) {
      Preconditions.checkNotNull(formatStr);
      FormatArg.push(arguments, formatStr, args);
      return this;
    }

    /** Concatenates the passed prefix string and the string. */
    public Builder addPrefixed(@CompileTimeConstant String prefix, @Nullable String arg) {
      return addPrefixedInternal(prefix, arg);
    }

    /** Concatenates the passed prefix string and the label using {@link Label#getCanonicalForm}. */
    public Builder addPrefixedLabel(@CompileTimeConstant String prefix, @Nullable Label arg) {
      return addPrefixedInternal(prefix, arg);
    }

    /** Concatenates the passed prefix string and the path. */
    public Builder addPrefixedPath(@CompileTimeConstant String prefix, @Nullable PathFragment arg) {
      return addPrefixedInternal(prefix, arg);
    }

    /** Concatenates the passed prefix string and the artifact's exec path. */
    public Builder addPrefixedExecPath(@CompileTimeConstant String prefix, @Nullable Artifact arg) {
      return addPrefixedInternal(prefix, arg);
    }

    /**
     * Concatenates the passed prefix string and the object's string representation.
     *
     * <p>Prefer {@link Builder#addPrefixed}, as it will be more memory efficient.
     */
    Builder addWithDynamicPrefix(String prefix, @Nullable Object arg) {
      return addPrefixedInternal(prefix, arg);
    }

    /**
     * Adds the passed strings to the command line.
     *
     * <p>If you are converting long lists or nested sets of a different type to string lists,
     * please try to use a different method that supports what you are trying to do directly.
     */
    public Builder addAll(@Nullable Collection<String> values) {
      return addCollectionInternal(values);
    }

    /** Adds the passed paths to the command line. */
    public Builder addPaths(@Nullable Collection<PathFragment> values) {
      return addCollectionInternal(values);
    }

    /**
     * Adds the artifacts' exec paths to the command line.
     *
     * <p>Do not use this method if the list is derived from a flattened nested set. Instead, figure
     * out how to avoid flattening the set and use {@link
     * Builder#addExecPaths(NestedSet<Artifact>)}.
     */
    public Builder addExecPaths(@Nullable Collection<Artifact> values) {
      return addCollectionInternal(values);
    }

    /** Adds the passed mapped values to the command line. */
    public <T> Builder addAll(@Nullable Collection<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).mapEach(mapFn));
    }

    /**
     * Adds the passed strings to the command line.
     *
     * <p>If you are converting long lists or nested sets of a different type to string lists,
     * please try to use a different method that supports what you are trying to do directly.
     */
    public Builder addAll(@Nullable NestedSet<String> values) {
      return addNestedSetInternal(values);
    }

    /** Adds the passed paths to the command line. */
    public Builder addPaths(@Nullable NestedSet<PathFragment> values) {
      return addNestedSetInternal(values);
    }

    /** Adds the artifacts' exec paths to the command line. */
    public Builder addExecPaths(@Nullable NestedSet<Artifact> values) {
      return addNestedSetInternal(values);
    }

    /** Adds the passed mapped values to the command line. */
    public <T> Builder addAll(@Nullable NestedSet<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).mapEach(mapFn));
    }

    /**
     * Adds the arg followed by the passed strings.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addAll(@CompileTimeConstant String arg, @Nullable Collection<String> values) {
      return addCollectionInternal(arg, values);
    }

    /**
     * Adds the arg followed by the path strings.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addPaths(
        @CompileTimeConstant String arg, @Nullable Collection<PathFragment> values) {
      return addCollectionInternal(arg, values);
    }

    /**
     * Adds the arg followed by the artifacts' exec paths.
     *
     * <p>Do not use this method if the list is derived from a flattened nested set. Instead, figure
     * out how to avoid flattening the set and use {@link Builder#addExecPaths(String,
     * NestedSet<Artifact>)}.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addExecPaths(
        @CompileTimeConstant String arg, @Nullable Collection<Artifact> values) {
      return addCollectionInternal(arg, values);
    }

    /**
     * Adds the arg followed by the passed mapped values.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public <T> Builder addAll(
        @CompileTimeConstant String arg,
        @Nullable Collection<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(arg, VectorArg.of(values).mapEach(mapFn));
    }

    /**
     * Adds the arg followed by the passed strings.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addAll(@CompileTimeConstant String arg, @Nullable NestedSet<String> values) {
      return addNestedSetInternal(arg, values);
    }

    /**
     * Adds the arg followed by the path fragments.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addPaths(
        @CompileTimeConstant String arg, @Nullable NestedSet<PathFragment> values) {
      return addNestedSetInternal(arg, values);
    }

    /**
     * Adds the arg followed by the artifacts' exec paths.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addExecPaths(
        @CompileTimeConstant String arg, @Nullable NestedSet<Artifact> values) {
      return addNestedSetInternal(arg, values);
    }

    /**
     * Adds the arg followed by the mapped values.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public <T> Builder addAll(
        @CompileTimeConstant String arg, @Nullable NestedSet<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(arg, VectorArg.of(values).mapEach(mapFn));
    }

    /** Adds the values joined with the supplied string. */
    public Builder addJoined(String delimiter, Collection<String> values) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the path strings joined with the supplied string. */
    public Builder addJoinedPaths(String delimiter, Collection<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the exec path strings joined with the supplied string. */
    public Builder addJoinedExecPaths(String delimiter, Collection<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the mapped values joined with the supplied string. */
    public <T> Builder addJoined(
        String delimiter, Collection<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the values joined with the supplied string. */
    public Builder addJoined(String delimiter, NestedSet<String> values) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the path strings joined with the supplied string. */
    public Builder addJoinedPaths(String delimiter, NestedSet<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the exec path strings joined with the supplied string. */
    public Builder addJoinedExecPaths(String delimiter, NestedSet<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the mapped values joined with the supplied string. */
    public <T> Builder addJoined(String delimiter, NestedSet<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the values joined with the supplied string. */
    public Builder addJoined(
        @CompileTimeConstant String arg, String delimiter, Collection<String> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the path strings joined with the supplied string. */
    public Builder addJoinedPaths(
        @CompileTimeConstant String arg, String delimiter, Collection<PathFragment> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the exec path strings joined with the supplied string. */
    public Builder addJoinedExecPaths(
        @CompileTimeConstant String arg, String delimiter, Collection<Artifact> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the mapped values joined with the supplied string. */
    public <T> Builder addJoined(
        @CompileTimeConstant String arg,
        String delimiter,
        Collection<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the values joined with the supplied string. */
    public Builder addJoined(
        @CompileTimeConstant String arg, String delimiter, NestedSet<String> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the path strings joined with the supplied string. */
    public Builder addJoinedPaths(
        @CompileTimeConstant String arg, String delimiter, NestedSet<PathFragment> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the exec path strings joined with the supplied string. */
    public Builder addJoinedExecPaths(
        @CompileTimeConstant String arg, String delimiter, NestedSet<Artifact> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter));
    }

    /** Adds the mapped values joined with the supplied string. */
    public <T> Builder addJoined(
        @CompileTimeConstant String arg,
        String delimiter,
        NestedSet<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(arg, VectorArg.of(values).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string. */
    public Builder addFormatEach(@CompileTimeConstant String formatStr, Collection<String> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachPath(
        @CompileTimeConstant String formatStr, Collection<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the exec path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachExecPath(
        @CompileTimeConstant String formatStr, Collection<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the mapped values with each value formatted by the supplied format string. */
    public <T> Builder addFormatEach(
        @CompileTimeConstant String formatStr, Collection<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string. */
    public Builder addFormatEach(@CompileTimeConstant String formatStr, NestedSet<String> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachPath(
        @CompileTimeConstant String formatStr, NestedSet<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the exec path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachExecPath(
        @CompileTimeConstant String formatStr, NestedSet<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the mapped values with each value formatted by the supplied format string. */
    public <T> Builder addFormatEach(
        @CompileTimeConstant String formatStr, NestedSet<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string. */
    public Builder addFormatEach(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        Collection<String> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachPath(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        Collection<PathFragment> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the exec path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachExecPath(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        Collection<Artifact> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the mapped values with each value formatted by the supplied format string. */
    public <T> Builder addFormatEach(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        Collection<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string. */
    public Builder addFormatEach(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        NestedSet<String> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachPath(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        NestedSet<PathFragment> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the exec path strings with each path formatted by the supplied format string. */
    public Builder addFormatEachExecPath(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        NestedSet<Artifact> values) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr));
    }

    /** Adds the mapped values with each value formatted by the supplied format string. */
    public <T> Builder addFormatEach(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        NestedSet<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(arg, VectorArg.of(values).formatEach(formatStr).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string, then joined. */
    public Builder addFormatEachJoined(
        @CompileTimeConstant String formatStr, String delimiter, Collection<String> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the path strings with each path formatted by the supplied format string, then joined.
     */
    public Builder addFormatEachPathJoined(
        @CompileTimeConstant String formatStr, String delimiter, Collection<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the exec path strings with each path formatted by the supplied format string, then
     * joined.
     */
    public Builder addFormatEachExecPathJoined(
        @CompileTimeConstant String formatStr, String delimiter, Collection<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the mapped values with each value formatted by the supplied format string, then joined.
     */
    public <T> Builder addFormatEachJoined(
        @CompileTimeConstant String formatStr,
        String delimiter,
        Collection<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(
          VectorArg.of(values).formatEach(formatStr).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string, then joined. */
    public Builder addFormatEachJoined(
        @CompileTimeConstant String formatStr, String delimiter, NestedSet<String> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the path strings with each path formatted by the supplied format string, then joined.
     */
    public Builder addFormatEachPathJoined(
        @CompileTimeConstant String formatStr, String delimiter, NestedSet<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the exec path strings with each path formatted by the supplied format string, then
     * joined.
     */
    public Builder addFormatEachExecPathJoined(
        @CompileTimeConstant String formatStr, String delimiter, NestedSet<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the mapped values with each value formatted by the supplied format string, then joined.
     */
    public <T> Builder addFormatEachJoined(
        @CompileTimeConstant String formatStr,
        String delimiter,
        NestedSet<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(
          VectorArg.of(values).formatEach(formatStr).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string, then joined. */
    public Builder addFormatEachJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        Collection<String> values) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the path strings with each path formatted by the supplied format string, then joined.
     */
    public Builder addFormatEachPathJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        Collection<PathFragment> values) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the exec path strings with each path formatted by the supplied format string, then
     * joined.
     */
    public Builder addFormatEachExecPathJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        Collection<Artifact> values) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the mapped values with each value formatted by the supplied format string, then joined.
     */
    public <T> Builder addFormatEachJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        Collection<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the values with each value formatted by the supplied format string, then joined. */
    public Builder addFormatEachJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        NestedSet<String> values) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the path strings with each path formatted by the supplied format string, then joined.
     */
    public Builder addFormatEachPathJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        NestedSet<PathFragment> values) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the exec path strings with each path formatted by the supplied format string, then
     * joined.
     */
    public Builder addFormatEachExecPathJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        NestedSet<Artifact> values) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter));
    }

    /**
     * Adds the mapped values with each value formatted by the supplied format string, then joined.
     */
    public <T> Builder addFormatEachJoined(
        @CompileTimeConstant String arg,
        @CompileTimeConstant String formatStr,
        String delimiter,
        NestedSet<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(
          arg, VectorArg.of(values).formatEach(formatStr).joinWith(delimiter).mapEach(mapFn));
    }

    /** Adds the beforeEach string and the values interspersed. */
    public Builder addBeforeEach(
        @CompileTimeConstant String beforeEach, Collection<String> values) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach));
    }

    /** Adds the beforeEach string and the path strings interspersed. */
    public Builder addBeforeEachPath(
        @CompileTimeConstant String beforeEach, Collection<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach));
    }

    /** Adds the beforeEach string and the exec path strings interspersed. */
    public Builder addBeforeEachExecPath(
        @CompileTimeConstant String beforeEach, Collection<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach));
    }

    /** Adds the beforeEach string and the mapped values interspersed. */
    public <T> Builder addBeforeEach(
        @CompileTimeConstant String beforeEach, Collection<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach).mapEach(mapFn));
    }

    /** Adds the beforeEach string and the values interspersed. */
    public Builder addBeforeEach(@CompileTimeConstant String beforeEach, NestedSet<String> values) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach));
    }

    /** Adds the beforeEach string and the path strings interspersed. */
    public Builder addBeforeEachPath(
        @CompileTimeConstant String beforeEach, NestedSet<PathFragment> values) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach));
    }

    /** Adds the beforeEach string and the exec path strings interspersed. */
    public Builder addBeforeEachExecPath(
        @CompileTimeConstant String beforeEach, NestedSet<Artifact> values) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach));
    }

    /** Adds the beforeEach string and the values interspersed. */
    public <T> Builder addBeforeEach(
        @CompileTimeConstant String beforeEach, NestedSet<T> values, Function<T, String> mapFn) {
      return addVectorArgInternal(VectorArg.of(values).beforeEach(beforeEach).mapEach(mapFn));
    }

    /** Adds the beforeEach string and the values interspersed. */
    public Builder addBeforeEachFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        Collection<String> values) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr));
    }

    /** Adds the beforeEach string and the path strings interspersed. */
    public Builder addBeforeEachPathFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        Collection<PathFragment> values) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr));
    }

    /** Adds the beforeEach string and the exec path strings interspersed. */
    public Builder addBeforeEachExecPathFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        Collection<Artifact> values) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr));
    }

    /** Adds the beforeEach string and the mapped values interspersed. */
    public <T> Builder addBeforeEachFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        Collection<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr).mapEach(mapFn));
    }

    /** Adds the beforeEach string and the values interspersed. */
    public Builder addBeforeEachFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        NestedSet<String> values) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr));
    }

    /** Adds the beforeEach string and the path strings interspersed. */
    public Builder addBeforeEachPathFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        NestedSet<PathFragment> values) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr));
    }

    /** Adds the beforeEach string and the exec path strings interspersed. */
    public Builder addBeforeEachExecPathFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        NestedSet<Artifact> values) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr));
    }

    /** Adds the beforeEach string and the mapped values interspersed. */
    public <T> Builder addBeforeEachFormatted(
        @CompileTimeConstant String beforeEach,
        @CompileTimeConstant String formatStr,
        NestedSet<T> values,
        Function<T, String> mapFn) {
      return addVectorArgInternal(
          VectorArg.of(values).beforeEach(beforeEach).formatEach(formatStr).mapEach(mapFn));
    }

    public Builder addCustomMultiArgv(@Nullable CustomMultiArgv arg) {
      if (arg != null) {
        arguments.add(arg);
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

    public CustomCommandLine build() {
      return new CustomCommandLine(arguments);
    }

    private Builder addObjectInternal(@Nullable Object value) {
      if (value != null) {
        arguments.add(value);
      }
      return this;
    }

    /** Adds the arg and the passed value if the value is non-null. */
    private Builder addObjectInternal(@CompileTimeConstant String arg, @Nullable Object value) {
      Preconditions.checkNotNull(arg);
      if (value != null) {
        arguments.add(arg);
        addObjectInternal(value);
      }
      return this;
    }

    private Builder addPrefixedInternal(String prefix, @Nullable Object arg) {
      Preconditions.checkNotNull(prefix);
      if (arg != null) {
        PrefixArg.push(arguments, prefix, arg);
      }
      return this;
    }

    private Builder addCollectionInternal(@Nullable Collection<?> values) {
      if (values != null) {
        addVectorArgInternal(VectorArg.of(values));
      }
      return this;
    }

    private Builder addCollectionInternal(
        @CompileTimeConstant String arg, @Nullable Collection<?> values) {
      Preconditions.checkNotNull(arg);
      if (values != null && !values.isEmpty()) {
        arguments.add(arg);
        addCollectionInternal(values);
      }
      return this;
    }

    private Builder addNestedSetInternal(@Nullable NestedSet<?> values) {
      if (values != null) {
        arguments.add(values);
      }
      return this;
    }

    private Builder addNestedSetInternal(
        @CompileTimeConstant String arg, @Nullable NestedSet<?> values) {
      Preconditions.checkNotNull(arg);
      if (values != null && !values.isEmpty()) {
        arguments.add(arg);
        addNestedSetInternal(values);
      }
      return this;
    }

    private Builder addVectorArgInternal(VectorArg.Builder<?> vectorArg) {
      if (!vectorArg.isEmpty) {
        VectorArg.push(arguments, vectorArg);
      }
      return this;
    }

    private Builder addVectorArgInternal(
        @CompileTimeConstant String arg, VectorArg.Builder<?> vectorArg) {
      Preconditions.checkNotNull(arg);
      if (!vectorArg.isEmpty) {
        arguments.add(arg);
        addVectorArgInternal(vectorArg);
      }
      return this;
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  public static Builder builder(Builder other) {
    Builder builder = new Builder();
    builder.arguments.addAll(other.arguments);
    return builder;
  }

  private final ImmutableList<Object> arguments;

  /**
   * A map between enclosed TreeArtifacts and their associated {@link TreeFileArtifact}s for
   * substitution.
   *
   * <p>This map is used to support TreeArtifact substitutions in {@link
   * TreeFileArtifactArgvFragment}s.
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
    int count = arguments.size();
    for (int i = 0; i < count; ) {
      Object arg = arguments.get(i++);
      Object substitutedArg = substituteTreeFileArtifactArgvFragment(arg);
      if (substitutedArg instanceof Iterable) {
        evalSimpleVectorArg((Iterable<?>) substitutedArg, builder);
      } else if (substitutedArg instanceof ArgvFragment) {
        if (artifactExpander != null
            && substitutedArg instanceof TreeArtifactExpansionArgvFragment) {
          TreeArtifactExpansionArgvFragment expansionArg =
              (TreeArtifactExpansionArgvFragment) substitutedArg;
          expansionArg.eval(builder, artifactExpander);
        } else {
          i = ((ArgvFragment) substitutedArg).eval(arguments, i, builder);
        }
      } else {
        builder.add(valueToString(substitutedArg));
      }
    }
    return builder.build();
  }

  private void evalSimpleVectorArg(Iterable<?> arg, ImmutableList.Builder<String> builder) {
    for (Object value : arg) {
      builder.add(valueToString(value));
    }
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

  private static String valueToString(Object value) {
    return value instanceof Artifact
        ? ((Artifact) value).getExecPath().getPathString()
        : value.toString();
  }
}
