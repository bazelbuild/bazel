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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.MoreObjects.firstNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleConfiguredTargetUtil;
import com.google.devtools.build.lib.collect.ImmutableSharedKeyMap;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.starlarkbuildapi.OutputGroupInfoApi;
import com.google.errorprone.annotations.ForOverride;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkIndexable;
import net.starlark.java.eval.StarlarkIterable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * {@code ConfiguredTarget}s implementing this interface can provide artifacts that <b>can</b> be
 * built when the target is mentioned on the command line (as opposed to being always built, like
 * {@link com.google.devtools.build.lib.analysis.FileProvider})
 *
 * <p>The artifacts are grouped into "output groups". Which output groups are built is controlled by
 * the {@code --output_groups} undocumented command line option, which in turn is added to the
 * command line at the discretion of the build command being run.
 *
 * <p>Output groups starting with an underscore are "not important". This means that artifacts built
 * because such an output group is mentioned in a {@code --output_groups} command line option are
 * not mentioned on the output.
 *
 * <p>Implementations are optimized for memory footprint based on common usage, including compact
 * representations for groups with an empty set of files, a small number of groups (1 or 2), and the
 * frequently used {@link #DEFAULT_GROUPS}. See detection of special cases in the {@link
 * #singleGroup} and {@link #createInternal} factory methods.
 */
@Immutable
public abstract class OutputGroupInfo extends StructImpl
    implements StarlarkIndexable, StarlarkIterable<String>, OutputGroupInfoApi {
  public static final String STARLARK_NAME = "output_groups";

  public static final OutputGroupInfoProvider STARLARK_CONSTRUCTOR = new OutputGroupInfoProvider();

  /**
   * Prefix for output groups that are not reported to the user on the terminal output of Blaze when
   * they are built.
   */
  public static final String HIDDEN_OUTPUT_GROUP_PREFIX = "_";

  /**
   * Suffix for output groups that are internal to bazel and may not be referenced from a filegroup.
   */
  public static final String INTERNAL_SUFFIX = "_INTERNAL_";

  /**
   * Building these artifacts only results in the compilation (and not e.g. linking) of the
   * associated target. Mostly useful for C++, less so for e.g. Java.
   */
  public static final String FILES_TO_COMPILE = "compilation_outputs";

  /**
   * These artifacts are the direct requirements for compilation, but building these does not
   * actually compile the target. Mostly useful when IDEs want Blaze to emit generated code so that
   * they can do the compilation in their own way.
   */
  public static final String COMPILATION_PREREQUISITES =
      "compilation_prerequisites" + INTERNAL_SUFFIX;

  /**
   * These files are built when a target is mentioned on the command line, but are not reported to
   * the user. This is mostly runfiles, which is necessary because we don't want a target to
   * successfully build if a file in its runfiles is broken.
   */
  public static final String HIDDEN_TOP_LEVEL =
      HIDDEN_OUTPUT_GROUP_PREFIX + "hidden_top_level" + INTERNAL_SUFFIX;

  /**
   * This output group contains artifacts that are the outputs of validation actions. These actions
   * should be run even if no other action depends on their outputs, therefore this output group is:
   *
   * <ul>
   *   <li>built even if <code>--output_groups</code> overrides the default output groups
   *   <li>not affected by the subtraction operation of <code>--output_groups</code> (i.e. <code>
   *       "--output_groups=-_validation"</code>)
   * </ul>
   *
   * The only way to disable this output group is with <code>--run_validations=false</code>.
   */
  public static final String VALIDATION = HIDDEN_OUTPUT_GROUP_PREFIX + "validation";

  /** Helper output group used to request {@link #VALIDATION} outputs from top-level aspect. */
  public static final String VALIDATION_TOP_LEVEL =
      HIDDEN_OUTPUT_GROUP_PREFIX + "validation_top_level" + INTERNAL_SUFFIX;

  /** Helper output group to override {@link #VALIDATION} outputs from dependencies */
  public static final String VALIDATION_TRANSITIVE =
      HIDDEN_OUTPUT_GROUP_PREFIX + "validation_transitive";

  /**
   * Temporary files created during building a rule, for example, .i, .d and .s files for C++
   * compilation.
   *
   * <p>This output group is somewhat special: it is always built, but it only contains files when
   * the {@code --save_temps} command line option present. I'm not sure if this is to save RAM by
   * not creating the associated actions and artifacts if we don't need them or just historical
   * baggage.
   */
  public static final String TEMP_FILES = "temp_files" + INTERNAL_SUFFIX;

  /** The default group of files built by a target when it is mentioned on the command line. */
  public static final String DEFAULT = "default";

  /** The default set of OutputGroups we typically want to build. */
  public static final ImmutableSortedSet<String> DEFAULT_GROUPS =
      ImmutableSortedSet.of(DEFAULT, TEMP_FILES, HIDDEN_TOP_LEVEL);

  private static final NestedSet<Artifact> EMPTY_FILES =
      NestedSetBuilder.emptySet(Order.STABLE_ORDER);

  /** Request parameter for {@link #determineOutputGroups}. */
  public enum ValidationMode {
    /** Validation outputs not built. */
    OFF,
    /**
     * Validation outputs built by requesting {@link #VALIDATION} output group Blaze core collects.
     */
    OUTPUT_GROUP,
    /**
     * Validation outputs built by {@code ValidateTarget} aspect "promoting" {@link #VALIDATION}
     * output group Blaze core collects to {@link #VALIDATION_TOP_LEVEL} and requesting the latter.
     */
    ASPECT
  }

  @Nullable
  public static OutputGroupInfo get(ProviderCollection collection) {
    return collection.get(STARLARK_CONSTRUCTOR);
  }

  /**
   * Merges output groups from a list of output providers. The set of output groups must be
   * disjoint.
   */
  @Nullable
  public static OutputGroupInfo merge(List<OutputGroupInfo> providers) throws DuplicateException {
    if (providers.isEmpty()) {
      return null;
    }
    if (providers.size() == 1) {
      return providers.get(0);
    }

    Map<String, NestedSet<Artifact>> outputGroups = new TreeMap<>();
    for (OutputGroupInfo provider : providers) {
      for (String group : provider) {
        if (outputGroups.put(group, provider.getOutputGroup(group)) != null) {
          throw new DuplicateException("Output group " + group + " provided twice");
        }
      }
    }
    return createInternal(ImmutableMap.copyOf(outputGroups));
  }

  public static ImmutableSortedSet<String> determineOutputGroups(
      List<String> outputGroups, ValidationMode validationMode, boolean shouldRunTests) {
    return determineOutputGroups(DEFAULT_GROUPS, outputGroups, validationMode, shouldRunTests);
  }

  @VisibleForTesting
  static ImmutableSortedSet<String> determineOutputGroups(
      Set<String> defaultOutputGroups,
      List<String> outputGroups,
      ValidationMode validationMode,
      boolean shouldRunTests) {

    Set<String> current = Sets.newHashSet();

    // If all of the requested output groups start with "+" or "-", then these are added or
    // subtracted to the set of default output groups.
    // If any of them don't start with "+" or "-", then the list of requested output groups
    // overrides the default set of output groups, except for the validation output group.
    boolean addDefaultOutputGroups = true;
    for (String outputGroup : outputGroups) {
      if (!(outputGroup.startsWith("+") || outputGroup.startsWith("-"))) {
        addDefaultOutputGroups = false;
        break;
      }
    }
    if (addDefaultOutputGroups) {
      current.addAll(defaultOutputGroups);
    }

    for (String outputGroup : outputGroups) {
      if (outputGroup.startsWith("+")) {
        current.add(outputGroup.substring(1));
      } else if (outputGroup.startsWith("-")) {
        current.remove(outputGroup.substring(1));
      } else {
        current.add(outputGroup);
      }
    }

    // Add the validation output group regardless of the additions and subtractions above.
    switch (validationMode) {
      case OUTPUT_GROUP:
        current.add(VALIDATION);
        break;
      case ASPECT:
        current.add(VALIDATION_TOP_LEVEL);
        break;
      case OFF: // fall out
    }

    // The `test` command ultimately requests artifacts from the `default` output group in order to
    // execute the tests, so we should ensure these artifacts are requested by the targets for
    // proper failure reporting.
    if (shouldRunTests) {
      current.add(DEFAULT);
    }

    return ImmutableSortedSet.copyOf(current);
  }

  public static OutputGroupInfo singleGroup(String group, NestedSet<Artifact> files) {
    if (files.isEmpty()) {
      return EmptyFiles.of(ImmutableSet.of(group));
    }
    switch (group) {
      case HIDDEN_TOP_LEVEL:
        return new HiddenTopLevelOnly(files);
      case VALIDATION:
        return new ValidationOnly(files);
      case DEFAULT:
        return new DefaultOnly(files);
      default:
        return new OtherGroupOnly(group, files);
    }
  }

  static OutputGroupInfo fromBuilders(SortedMap<String, NestedSetBuilder<Artifact>> builders) {
    var outputGroups =
        ImmutableMap.<String, NestedSet<Artifact>>builderWithExpectedSize(builders.size());
    builders.forEach((group, files) -> outputGroups.put(group, files.build()));
    return createInternal(outputGroups.buildOrThrow());
  }

  private static OutputGroupInfo createInternal(
      ImmutableMap<String, NestedSet<Artifact>> outputGroups) {
    if (outputGroups.values().stream().allMatch(NestedSet::isEmpty)) {
      @SuppressWarnings("ImmutableSetCopyOfImmutableSet") // keySet retains a reference to the map.
      ImmutableSet<String> groups = ImmutableSet.copyOf(outputGroups.keySet());
      return EmptyFiles.of(groups);
    }

    if (outputGroups.size() == 1) {
      String onlyGroup = Iterables.getOnlyElement(outputGroups.keySet());
      return singleGroup(onlyGroup, outputGroups.get(onlyGroup));
    }

    if (outputGroups.size() == 2) {
      ImmutableList<String> groups = outputGroups.keySet().asList();
      if (groups.get(0).equals(HIDDEN_TOP_LEVEL)) {
        String otherGroup = groups.get(1);
        return new HiddenTopLevelAndOneOther(
            outputGroups.get(HIDDEN_TOP_LEVEL), otherGroup, outputGroups.get(otherGroup));
      }
    }

    return new ArbitraryGroups(ImmutableSharedKeyMap.copyOf(outputGroups));
  }

  private OutputGroupInfo() {}

  @Override
  public final OutputGroupInfoProvider getProvider() {
    return STARLARK_CONSTRUCTOR;
  }

  @Override
  public final boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  /**
   * Returns the artifacts in a particular output group.
   *
   * @return the artifacts in the output group with the given name. The return value is never null.
   *     If the specified output group is not present, the empty set is returned.
   */
  public abstract NestedSet<Artifact> getOutputGroup(String name);

  @Override
  public final Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    if (!(key instanceof String)) {
      throw Starlark.errorf(
          "Output group names must be strings, got %s instead", Starlark.type(key));
    }
    Depset result = getValue((String) key);
    if (result == null) {
      throw Starlark.errorf("Output group %s not present", key);
    }
    return result;
  }

  @Override
  public final boolean containsKey(StarlarkSemantics semantics, Object key) {
    return key instanceof String && containsKey((String) key);
  }

  @ForOverride
  abstract boolean containsKey(String name);

  @Nullable
  @Override
  public final Depset getValue(String name) {
    NestedSet<Artifact> result = getOutputGroup(name);
    if (result.isEmpty() && !containsKey(name)) {
      return null;
    }
    return Depset.of(Artifact.class, result);
  }

  /** All output groups are empty. */
  private static final class EmptyFiles extends OutputGroupInfo {
    private static final Interner<EmptyFiles> interner = BlazeInterners.newWeakInterner();

    static EmptyFiles of(ImmutableSet<String> groups) {
      return interner.intern(new EmptyFiles(groups));
    }

    private final ImmutableSet<String> groups;

    private EmptyFiles(ImmutableSet<String> groups) {
      this.groups = groups;
    }

    @Override
    public NestedSet<Artifact> getOutputGroup(String name) {
      return EMPTY_FILES;
    }

    @Override
    boolean containsKey(String name) {
      return groups.contains(name);
    }

    @Override
    public Iterator<String> iterator() {
      return groups.iterator();
    }

    @Override
    public ImmutableSet<String> getFieldNames() {
      return groups;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof EmptyFiles)) {
        return false;
      }
      EmptyFiles other = (EmptyFiles) o;
      return groups.equals(other.groups);
    }

    @Override
    public int hashCode() {
      return groups.hashCode();
    }
  }

  private abstract static class SingleGroup extends OutputGroupInfo {
    private final NestedSet<Artifact> files;

    private SingleGroup(NestedSet<Artifact> files) {
      this.files = files;
    }

    @Override
    public final NestedSet<Artifact> getOutputGroup(String name) {
      return containsKey(name) ? files : EMPTY_FILES;
    }

    @Override
    final boolean containsKey(String name) {
      return name.equals(groupName());
    }

    @Override
    public final Iterator<String> iterator() {
      return Iterators.singletonIterator(groupName());
    }

    @Override
    public final ImmutableSet<String> getFieldNames() {
      return ImmutableSet.of(groupName());
    }

    @ForOverride
    abstract String groupName();
  }

  /** A single non-empty output group: {@link #HIDDEN_TOP_LEVEL}. */
  private static final class HiddenTopLevelOnly extends SingleGroup {
    HiddenTopLevelOnly(NestedSet<Artifact> files) {
      super(files);
    }

    @Override
    String groupName() {
      return HIDDEN_TOP_LEVEL;
    }
  }

  /** A single non-empty output group: {@link #VALIDATION}. */
  private static final class ValidationOnly extends SingleGroup {
    ValidationOnly(NestedSet<Artifact> files) {
      super(files);
    }

    @Override
    String groupName() {
      return VALIDATION;
    }
  }

  /** A single non-empty output group: {@link #DEFAULT}. */
  private static final class DefaultOnly extends SingleGroup {
    DefaultOnly(NestedSet<Artifact> files) {
      super(files);
    }

    @Override
    String groupName() {
      return DEFAULT;
    }
  }

  /** A single non-empty output group besides the common groups special-cased above. */
  private static final class OtherGroupOnly extends SingleGroup {
    private final String groupName;

    OtherGroupOnly(String groupName, NestedSet<Artifact> files) {
      super(files);
      this.groupName = groupName;
    }

    @Override
    String groupName() {
      return groupName;
    }
  }

  /**
   * Two output groups: {@link #HIDDEN_TOP_LEVEL} and one other, at least one of which is non-empty.
   */
  private static final class HiddenTopLevelAndOneOther extends OutputGroupInfo {
    private final NestedSet<Artifact> hiddenTopLevelFiles;
    private final String otherGroup;
    private final NestedSet<Artifact> otherFiles;

    private HiddenTopLevelAndOneOther(
        NestedSet<Artifact> hiddenTopLevelFiles,
        String otherGroup,
        NestedSet<Artifact> otherFiles) {
      this.hiddenTopLevelFiles = hiddenTopLevelFiles;
      this.otherGroup = otherGroup;
      this.otherFiles = otherFiles;
    }

    @Override
    public NestedSet<Artifact> getOutputGroup(String name) {
      if (name.equals(HIDDEN_TOP_LEVEL)) {
        return hiddenTopLevelFiles;
      }
      if (name.equals(otherGroup)) {
        return otherFiles;
      }
      return EMPTY_FILES;
    }

    @Override
    boolean containsKey(String name) {
      return name.equals(HIDDEN_TOP_LEVEL) || name.equals(otherGroup);
    }

    @Override
    public Iterator<String> iterator() {
      return Iterators.forArray(HIDDEN_TOP_LEVEL, otherGroup);
    }

    @Override
    public ImmutableSet<String> getFieldNames() {
      return ImmutableSet.of(HIDDEN_TOP_LEVEL, otherGroup);
    }
  }

  /** Handles the arbitrary case for when none of the special cases above match. */
  private static final class ArbitraryGroups extends OutputGroupInfo {
    private final ImmutableSharedKeyMap<String, NestedSet<Artifact>> map;

    ArbitraryGroups(ImmutableSharedKeyMap<String, NestedSet<Artifact>> map) {
      this.map = map;
    }

    @Override
    public NestedSet<Artifact> getOutputGroup(String name) {
      return firstNonNull(map.get(name), EMPTY_FILES);
    }

    @Override
    boolean containsKey(String name) {
      return map.containsKey(name);
    }

    @Override
    public Iterator<String> iterator() {
      return map.iterator();
    }

    @Override
    public ImmutableSet<String> getFieldNames() {
      return ImmutableSet.copyOf(map);
    }
  }

  /** Provider implementation for {@link OutputGroupInfoApi.OutputGroupInfoApiProvider}. */
  public static final class OutputGroupInfoProvider extends BuiltinProvider<OutputGroupInfo>
      implements OutputGroupInfoApi.OutputGroupInfoApiProvider {

    OutputGroupInfoProvider() {
      super("OutputGroupInfo", OutputGroupInfo.class);
    }

    @Override
    public OutputGroupInfoApi constructor(Dict<String, Object> kwargs) throws EvalException {
      var outputGroups =
          ImmutableMap.<String, NestedSet<Artifact>>builderWithExpectedSize(kwargs.size());
      for (var entry : ImmutableList.sortedCopyOf(Map.Entry.comparingByKey(), kwargs.entrySet())) {
        outputGroups.put(
            entry.getKey(),
            StarlarkRuleConfiguredTargetUtil.convertToOutputGroupValue(
                entry.getKey(), entry.getValue()));
      }
      return createInternal(outputGroups.buildOrThrow());
    }
  }
}
