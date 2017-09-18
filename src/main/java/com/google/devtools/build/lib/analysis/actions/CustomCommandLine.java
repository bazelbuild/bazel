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

/** A customizable, serializable class for building memory efficient command lines. */
@Immutable
public final class CustomCommandLine extends CommandLine {

  private interface ArgvFragment {
    /**
     * Expands this fragment into the passed command line vector.
     *
     * @param arguments The command line's argument vector.
     * @param argi The index of the next available argument.
     * @param builder The command line builder to which we should add arguments.
     * @return The index of the next argument, after the ArgvFragment has consumed its args. If the
     *     ArgvFragment doesn't have any args, it should return {@code argi} unmodified.
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
  /** Custom Java code producing a List of String arguments. */
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
   * <p>Vector args support formatting, interspersing args (adding strings before each value),
   * joining, and mapping custom types. Please use this whenever you need to transform lists or
   * nested sets instead of doing it manually, as use of this class is more memory efficient.
   *
   * <p>The order of evaluation is:
   *
   * <ul>
   *   <li>Map the type T to a string using a custom map function, if any, or
   *   <li>Map any non-string type {PathFragment, Artifact} to their path/exec path
   *   <li>Format the string using the supplied format string, if any
   *   <li>Add the arguments each prepended by the before string, if any, or
   *   <li>Join the arguments with the join string, if any, or
   *   <li>Simply add all arguments
   * </ul>
   *
   * <pre>
   *   Examples:
   *
   *   List<String> values = ImmutableList.of("1", "2", "3");
   *
   *   commandBuilder.addAll(VectorArg.format("-l%s").each(values))
   *   -> ["-l1", "-l2", "-l3"]
   *
   *   commandBuilder.addAll(VectorArg.addBefore("-l").each(values))
   *   -> ["-l", "1", "-l", "2", "-l", "3"]
   *
   *   commandBuilder.addAll(VectorArg.join(":").each(values))
   *   -> ["1:2:3"]
   * </pre>
   */
  public static class VectorArg<T> {
    final boolean isNestedSet;
    final boolean isEmpty;
    final int count;
    final String formatEach;
    final String beforeEach;
    final String joinWith;

    private VectorArg(
        boolean isNestedSet,
        boolean isEmpty,
        int count,
        String formatEach,
        String beforeEach,
        String joinWith) {
      this.isNestedSet = isNestedSet;
      this.isEmpty = isEmpty;
      this.count = count;
      this.formatEach = formatEach;
      this.beforeEach = beforeEach;
      this.joinWith = joinWith;
    }

    /**
     * A vector arg that doesn't map its parameters.
     *
     * <p>Call {@link SimpleVectorArg#mapped} to produce a vector arg that maps from a given type to
     * a string.
     */
    public static class SimpleVectorArg<T> extends VectorArg<T> {
      private final Iterable<T> values;

      private SimpleVectorArg(Builder builder, @Nullable Collection<T> values) {
        super(
            false /* isNestedSet */,
            values == null || values.isEmpty(),
            values != null ? values.size() : 0,
            builder.formatEach,
            builder.beforeEach,
            builder.joinWith);
        this.values = values;
      }

      private SimpleVectorArg(Builder builder, @Nullable NestedSet<T> values) {
        super(
            true /* isNestedSet */,
            values == null || values.isEmpty(),
            -1 /* count */,
            builder.formatEach,
            builder.beforeEach,
            builder.joinWith);
        this.values = values;
      }

      /** Each argument is mapped using the supplied map function */
      public MappedVectorArg<T> mapped(Function<T, String> mapFn) {
        return new MappedVectorArg<>(this, mapFn);
      }
    }

    /** A vector arg that maps some type T to strings. */
    public static class MappedVectorArg<T> extends VectorArg<String> {
      private final Iterable<T> values;
      private final Function<T, String> mapFn;

      private MappedVectorArg(SimpleVectorArg<T> other, Function<T, String> mapFn) {
        super(
            other.isNestedSet,
            other.isEmpty,
            other.count,
            other.formatEach,
            other.beforeEach,
            other.joinWith);
        this.values = other.values;
        this.mapFn = mapFn;
      }
    }

    public static <T> SimpleVectorArg<T> of(Collection<T> values) {
      return new Builder().each(values);
    }

    public static <T> SimpleVectorArg<T> of(NestedSet<T> values) {
      return new Builder().each(values);
    }

    /** Each argument is formatted via {@link String#format}. */
    public static Builder format(@CompileTimeConstant String formatEach) {
      return new Builder().format(formatEach);
    }
    /** Each argument is prepended by the beforeEach param. */
    public static Builder addBefore(@CompileTimeConstant String beforeEach) {
      return new Builder().addBefore(beforeEach);
    }

    /** Once all arguments have been evaluated, they are joined with this delimiter */
    public static Builder join(String delimiter) {
      return new Builder().join(delimiter);
    }

    /** Builder for {@link VectorArg}. */
    public static class Builder {
      private String formatEach;
      private String beforeEach;
      private String joinWith;

      /** Each argument is formatted via {@link String#format}. */
      public Builder format(@CompileTimeConstant String formatEach) {
        Preconditions.checkNotNull(formatEach);
        this.formatEach = formatEach;
        return this;
      }

      /** Each argument is prepended by the beforeEach param. */
      public Builder addBefore(@CompileTimeConstant String beforeEach) {
        Preconditions.checkNotNull(beforeEach);
        this.beforeEach = beforeEach;
        return this;
      }

      /** Once all arguments have been evaluated, they are joined with this delimiter */
      public Builder join(String delimiter) {
        Preconditions.checkNotNull(delimiter);
        this.joinWith = delimiter;
        return this;
      }

      public <T> SimpleVectorArg<T> each(Collection<T> values) {
        return new SimpleVectorArg<>(this, values);
      }

      public <T> SimpleVectorArg<T> each(NestedSet<T> values) {
        return new SimpleVectorArg<>(this, values);
      }
    }

    @SuppressWarnings("unchecked")
    private static void push(List<Object> arguments, VectorArg<?> vectorArg) {
      final Iterable<?> values;
      final Function<?, String> mapFn;
      if (vectorArg instanceof SimpleVectorArg) {
        values = ((SimpleVectorArg) vectorArg).values;
        mapFn = null;
      } else {
        values = ((MappedVectorArg) vectorArg).values;
        mapFn = ((MappedVectorArg) vectorArg).mapFn;
      }
      VectorArgFragment vectorArgFragment =
          new VectorArgFragment(
              vectorArg.isNestedSet,
              mapFn != null,
              vectorArg.formatEach != null,
              vectorArg.beforeEach != null,
              vectorArg.joinWith != null);
      if (vectorArgFragment.hasBeforeEach && vectorArgFragment.hasJoinWith) {
        throw new IllegalArgumentException("Cannot use both 'before' and 'join' in vector arg.");
      }
      vectorArgFragment = VectorArgFragment.interner.intern(vectorArgFragment);
      arguments.add(vectorArgFragment);
      if (vectorArgFragment.isNestedSet) {
        arguments.add(values);
      } else {
        // Simply expand any ordinary collection into the argv
        arguments.add(vectorArg.count);
        Iterables.addAll(arguments, values);
      }
      if (vectorArgFragment.hasMapEach) {
        arguments.add(mapFn);
      }
      if (vectorArgFragment.hasFormatEach) {
        arguments.add(vectorArg.formatEach);
      }
      if (vectorArgFragment.hasBeforeEach) {
        arguments.add(vectorArg.beforeEach);
      }
      if (vectorArgFragment.hasJoinWith) {
        arguments.add(vectorArg.joinWith);
      }
    }

    private static final class VectorArgFragment implements ArgvFragment {
      private static Interner<VectorArgFragment> interner = BlazeInterners.newStrongInterner();

      private final boolean isNestedSet;
      private final boolean hasMapEach;
      private final boolean hasFormatEach;
      private final boolean hasBeforeEach;
      private final boolean hasJoinWith;

      private VectorArgFragment(
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

      @Override
      public boolean equals(Object o) {
        if (this == o) {
          return true;
        }
        if (o == null || getClass() != o.getClass()) {
          return false;
        }
        VectorArgFragment vectorArgFragment = (VectorArgFragment) o;
        return isNestedSet == vectorArgFragment.isNestedSet
            && hasMapEach == vectorArgFragment.hasMapEach
            && hasFormatEach == vectorArgFragment.hasFormatEach
            && hasBeforeEach == vectorArgFragment.hasBeforeEach
            && hasJoinWith == vectorArgFragment.hasJoinWith;
      }

      @Override
      public int hashCode() {
        return Objects.hashCode(isNestedSet, hasMapEach, hasFormatEach, hasBeforeEach, hasJoinWith);
      }
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
     * Returns a string that describes this argument fragment. The string can be used as part of an
     * action key for the command line at analysis time.
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
          "ExpandedTreeArtifactExecPathsArg{ treeArtifact: %s}", treeArtifact.getExecPathString());
    }
  }

  /**
   * An argument object that evaluates to the exec path of a {@link TreeFileArtifact}, enclosing the
   * associated {@link TreeFileArtifact}.
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
   * <p>Try to avoid coercing items to strings unnecessarily. Instead, use a more memory-efficient
   * form that defers the string coercion until the last moment. In particular, avoid flattening
   * lists and nested sets (see {@link VectorArg}).
   *
   * <p>Three types are given special consideration:
   *
   * <ul>
   *   <li>Any labels added will be added using {@link Label#getCanonicalForm()}
   *   <li>Path fragments will be added using {@link PathFragment#toString}
   *   <li>Artifacts will be added using {@link Artifact#getExecPathString()}.
   * </ul>
   *
   * <p>Any other type must be mapped to a string. For collections, please use {@link
   * VectorArg.SimpleVectorArg#mapped}.
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

    /**
     * Adds the arg followed by the passed strings.
     *
     * <p>If you are converting long lists or nested sets of a different type to string lists,
     * please try to use a different method that supports what you are trying to do directly.
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

    /** Adds the passed vector arg. See {@link VectorArg}. */
    public Builder addAll(VectorArg<String> vectorArg) {
      return addVectorArgInternal(vectorArg);
    }

    /** Adds the passed vector arg. See {@link VectorArg}. */
    public Builder addPaths(VectorArg<PathFragment> vectorArg) {
      return addVectorArgInternal(vectorArg);
    }

    /** Adds the passed vector arg. See {@link VectorArg}. */
    public Builder addExecPaths(VectorArg<Artifact> vectorArg) {
      return addVectorArgInternal(vectorArg);
    }

    /**
     * Adds the arg followed by the passed vector arg. See {@link VectorArg}.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addAll(@CompileTimeConstant String arg, VectorArg<String> vectorArg) {
      return addVectorArgInternal(arg, vectorArg);
    }

    /**
     * Adds the arg followed by the passed vector arg. See {@link VectorArg}.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addPaths(@CompileTimeConstant String arg, VectorArg<PathFragment> vectorArg) {
      return addVectorArgInternal(arg, vectorArg);
    }

    /**
     * Adds the arg followed by the passed vector arg. See {@link VectorArg}.
     *
     * <p>If values is empty, the arg isn't added.
     */
    public Builder addExecPaths(@CompileTimeConstant String arg, VectorArg<Artifact> vectorArg) {
      return addVectorArgInternal(arg, vectorArg);
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

    private Builder addVectorArgInternal(VectorArg<?> vectorArg) {
      if (!vectorArg.isEmpty) {
        VectorArg.push(arguments, vectorArg);
      }
      return this;
    }

    private Builder addVectorArgInternal(@CompileTimeConstant String arg, VectorArg<?> vectorArg) {
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
   * their parent TreeArtifacts with the TreeFileArtifacts in all {@link
   * TreeFileArtifactArgvFragment} argument objects.
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
