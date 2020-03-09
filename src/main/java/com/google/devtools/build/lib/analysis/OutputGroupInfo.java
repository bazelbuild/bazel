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

import static com.google.devtools.build.lib.syntax.EvalUtils.SKYLARK_COMPARATOR;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.configuredtargets.MergedConfiguredTarget.DuplicateException;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleConfiguredTargetUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.OutputGroupInfoApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkIndexable;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkIterable;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

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
 */
@Immutable
@AutoCodec
public final class OutputGroupInfo extends StructImpl
    implements SkylarkIndexable, StarlarkIterable<String>, OutputGroupInfoApi {
  public static final String SKYLARK_NAME = "output_groups";

  public static final OutputGroupInfoProvider SKYLARK_CONSTRUCTOR = new OutputGroupInfoProvider();

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

  /**
   * The default group of files built by a target when it is mentioned on the command line.
   */
  public static final String DEFAULT = "default";

  /**
   * The default set of OutputGroups we typically want to build.
   */
  public static final ImmutableSortedSet<String> DEFAULT_GROUPS =
      ImmutableSortedSet.of(DEFAULT, TEMP_FILES, HIDDEN_TOP_LEVEL);

  private final ImmutableMap<String, NestedSet<Artifact>> outputGroups;

  public OutputGroupInfo(ImmutableMap<String, NestedSet<Artifact>> outputGroups) {
    super(SKYLARK_CONSTRUCTOR, Location.BUILTIN);
    this.outputGroups = outputGroups;
  }

  @Nullable
  public static OutputGroupInfo get(TransitiveInfoCollection collection) {
    return collection.get(OutputGroupInfo.SKYLARK_CONSTRUCTOR);
  }

  @Nullable
  public static OutputGroupInfo get(ConfiguredAspect aspect) {
    return (OutputGroupInfo) aspect.get(SKYLARK_CONSTRUCTOR.getKey());
  }


  /** Return the artifacts in a particular output group.
   *
   * @return the artifacts in the output group with the given name. The return value is never null.
   *     If the specified output group is not present, the empty set is returned.
   */
  public NestedSet<Artifact> getOutputGroup(String outputGroupName) {
    return outputGroups.containsKey(outputGroupName)
        ? outputGroups.get(outputGroupName)
        : NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);
  }

  /**
   * Merges output groups from two output providers. The set of output groups must be disjoint.
   *
   * @param providers providers to merge {@code this} with.
   */
  @Nullable
  public static OutputGroupInfo merge(List<OutputGroupInfo> providers)
      throws DuplicateException {
    if (providers.size() == 0) {
      return null;
    }
    if (providers.size() == 1) {
      return providers.get(0);
    }

    ImmutableMap.Builder<String, NestedSet<Artifact>> resultBuilder = new ImmutableMap.Builder<>();
    Set<String> seenGroups = new HashSet<>();
    for (OutputGroupInfo provider : providers) {
      for (String outputGroup : provider.outputGroups.keySet()) {
        if (!seenGroups.add(outputGroup)) {
          throw new DuplicateException(
              "Output group " + outputGroup + " provided twice");
        }

        resultBuilder.put(outputGroup, provider.getOutputGroup(outputGroup));
      }
    }
    return new OutputGroupInfo(resultBuilder.build());
  }

  public static ImmutableSortedSet<String> determineOutputGroups(
      List<String> outputGroups, boolean includeValidationOutputGroup) {
    return determineOutputGroups(DEFAULT_GROUPS, outputGroups, includeValidationOutputGroup);
  }

  public static ImmutableSortedSet<String> determineOutputGroups(
      Set<String> defaultOutputGroups,
      List<String> outputGroups,
      boolean includeValidationOutputGroup) {

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
    if (includeValidationOutputGroup) {
      current.add(VALIDATION);
    }

    return ImmutableSortedSet.copyOf(current);
  }

  @Override
  public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    if (!(key instanceof String)) {
      throw Starlark.errorf(
          "Output group names must be strings, got %s instead", Starlark.type(key));
    }

    NestedSet<Artifact> result = outputGroups.get(key);
    if (result != null) {
      return Depset.of(Artifact.TYPE, result);
    } else {
      throw Starlark.errorf("Output group %s not present", key);
    }
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    return outputGroups.containsKey(key);
  }

  @Override
  public Iterator<String> iterator() {
    return SKYLARK_COMPARATOR.sortedCopy(outputGroups.keySet()).iterator();
  }

  @Override
  public Object getValue(String name) {
    NestedSet<Artifact> result = outputGroups.get(name);
    if (result == null) {
      return null;
    }
    return Depset.of(Artifact.TYPE, result);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return outputGroups.keySet();
  }

  /**
   * Provider implementation for {@link OutputGroupInfoApi.OutputGroupInfoApiProvider}.
   */
  public static class OutputGroupInfoProvider extends BuiltinProvider<OutputGroupInfo>
      implements OutputGroupInfoApi.OutputGroupInfoApiProvider {
    private OutputGroupInfoProvider() {
      super("OutputGroupInfo", OutputGroupInfo.class);
    }

    @Override
    public OutputGroupInfoApi constructor(Dict<?, ?> kwargs) throws EvalException {
      Map<String, Object> kwargsMap = kwargs.getContents(String.class, Object.class, "kwargs");

      ImmutableMap.Builder<String, NestedSet<Artifact>> builder = ImmutableMap.builder();
      for (Map.Entry<String, Object> entry : kwargsMap.entrySet()) {
        builder.put(
            entry.getKey(),
            SkylarkRuleConfiguredTargetUtil.convertToOutputGroupValue(
                entry.getKey(), entry.getValue()));
      }
      return new OutputGroupInfo(builder.build());
    }
  }
}
