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

import static java.util.stream.Collectors.joining;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Expands $(location) and $(locations) tags inside target attributes. You can specify something
 * like this in the BUILD file:
 *
 * <pre>
 * somerule(name='some name',
 *          someopt = [ '$(location //mypackage:myhelper)' ],
 *          ...)
 * </pre>
 *
 * and location will be substituted with //mypackage:myhelper executable output.
 *
 * <p>Note that this expander will always expand labels in srcs, deps, and tools attributes, with
 * data being optional.
 *
 * <p>DO NOT USE DIRECTLY! Use RuleContext.getExpander() instead.
 */
public final class LocationExpander {

  /**
   * List of options to tweak the LocationExpander.
   */
  public static enum Options {
    /** output the execPath instead of the relative path */
    EXEC_PATHS,
    /** Allow to take label from the data attribute */
    ALLOW_DATA,
  }

  private static final String LOCATION = "$(location";

  private final RuleErrorConsumer ruleErrorConsumer;
  private final Function<String, String> locationFunction;
  private final Function<String, String> locationsFunction;

  @VisibleForTesting
  LocationExpander(
      RuleErrorConsumer ruleErrorConsumer,
      Function<String, String> locationFunction,
      Function<String, String> locationsFunction) {
    this.ruleErrorConsumer = ruleErrorConsumer;
    this.locationFunction = locationFunction;
    this.locationsFunction = locationsFunction;
  }

  private LocationExpander(
      RuleErrorConsumer ruleErrorConsumer,
      Label root,
      Supplier<Map<Label, Collection<Artifact>>> locationMap,
      boolean execPaths) {
    this(
        ruleErrorConsumer,
        new LocationFunction(root, locationMap, execPaths, false),
        new LocationFunction(root, locationMap, execPaths, true));
  }

  /**
   * Creates location expander helper bound to specific target and with default location map.
   *
   * @param ruleContext BUILD rule
   * @param labelMap A mapping of labels to build artifacts.
   * @param options options
   */
  private LocationExpander(
      RuleContext ruleContext,
      @Nullable ImmutableMap<Label, ImmutableCollection<Artifact>> labelMap,
      ImmutableSet<Options> options) {
    this(
        ruleContext,
        ruleContext.getLabel(),
        // Use a memoizing supplier to avoid eagerly building the location map.
        Suppliers.memoize(
            () -> LocationExpander.buildLocationMap(
                ruleContext, labelMap, options.contains(Options.ALLOW_DATA))),
        options.contains(Options.EXEC_PATHS));
  }

  /**
   * Creates location expander helper bound to specific target and with default location map.
   *
   * @param ruleContext BUILD rule
   * @param labelMap A mapping of labels to build artifacts.
   * @param options the list of options, see {@link Options}
   */
  public LocationExpander(
      RuleContext ruleContext, ImmutableMap<Label, ImmutableCollection<Artifact>> labelMap,
      Options... options) {
    this(ruleContext, Preconditions.checkNotNull(labelMap), ImmutableSet.copyOf(options));
  }

  /**
   * Creates location expander helper bound to specific target.
   *
   * @param ruleContext the BUILD rule's context
   * @param options the list of options, see {@link Options}.
   */
  public LocationExpander(RuleContext ruleContext, Options... options) {
    this(ruleContext, null, ImmutableSet.copyOf(options));
  }

  public String expand(String input) {
    return expand(input, new RuleErrorReporter(ruleErrorConsumer));
  }

  /**
   * Expands attribute's location and locations tags based on the target and
   * location map.
   *
   * @param attrName  name of the attribute; only used for error reporting
   * @param attrValue initial value of the attribute
   * @return attribute value with expanded location tags or original value in
   *         case of errors
   */
  public String expandAttribute(String attrName, String attrValue) {
    return expand(attrValue, new AttributeErrorReporter(ruleErrorConsumer, attrName));
  }

  private String expand(String value, ErrorReporter reporter) {
    int restart = 0;

    int attrLength = value.length();
    StringBuilder result = new StringBuilder(value.length());

    while (true) {
      // (1) Find '$(location ' or '$(locations '.
      Function<String, String> func = locationFunction;
      int start = value.indexOf(LOCATION, restart);
      int scannedLength = LOCATION.length();
      if (start == -1 || start + scannedLength == attrLength) {
        result.append(value.substring(restart));
        break;
      }
      if (value.charAt(start + scannedLength) == 's') {
        scannedLength++;
        if (start + scannedLength == attrLength) {
          result.append(value.substring(restart));
          break;
        }
        func = locationsFunction;
      }
      if (value.charAt(start + scannedLength) != ' ') {
        result.append(value, restart, start + scannedLength);
        restart = start + scannedLength;
        continue;
      }

      result.append(value, restart, start);
      scannedLength++;

      int end = value.indexOf(')', start + scannedLength);
      if (end == -1) {
        reporter.report(
            String.format(
                "unterminated $(%s) expression",
                value.substring(start + 2, start + scannedLength - 1)));
        return value;
      }

      // (2) Call appropriate function to obtain string replacement.
      String functionValue = value.substring(start + scannedLength, end).trim();
      try {
        String replacement = func.apply(functionValue);
        result.append(replacement);
      } catch (IllegalStateException ise) {
        reporter.report(ise.getMessage());
        return value;
      }

      restart = end + 1;
    }

    return result.toString();
  }

  @VisibleForTesting
  static final class LocationFunction implements Function<String, String> {
    private static final int MAX_PATHS_SHOWN = 5;

    private final Label root;
    private final Supplier<Map<Label, Collection<Artifact>>> locationMapSupplier;
    private final boolean execPaths;
    private final boolean multiple;

    LocationFunction(
        Label root,
        Supplier<Map<Label, Collection<Artifact>>> locationMapSupplier,
        boolean execPaths,
        boolean multiple) {
      this.root = root;
      this.locationMapSupplier = locationMapSupplier;
      this.execPaths = execPaths;
      this.multiple = multiple;
    }

    @Override
    public String apply(String arg) {
      Label label;
      try {
        label = root.getRelative(arg);
      } catch (LabelSyntaxException e) {
        throw new IllegalStateException(
            String.format(
                "invalid label in %s expression: %s", functionName(), e.getMessage()), e);
      }
      Collection<String> paths = resolveLabel(label);
      return joinPaths(paths);
    }

    /**
     * Returns all target location(s) of the given label.
     */
    private Collection<String> resolveLabel(Label unresolved) throws IllegalStateException {
      Collection<Artifact> artifacts = locationMapSupplier.get().get(unresolved);

      if (artifacts == null) {
        throw new IllegalStateException(
            String.format(
                "label '%s' in %s expression is not a declared prerequisite of this rule",
                unresolved, functionName()));
      }

      Set<String> paths = getPaths(artifacts, execPaths);
      if (paths.isEmpty()) {
        throw new IllegalStateException(
            String.format(
                "label '%s' in %s expression expands to no files",
                unresolved, functionName()));
      }

      if (!multiple && paths.size() > 1) {
        throw new IllegalStateException(
            String.format(
                "label '%s' in $(location) expression expands to more than one file, "
                    + "please use $(locations %s) instead.  Files (at most %d shown) are: %s",
                unresolved,
                unresolved,
                MAX_PATHS_SHOWN,
                Iterables.limit(paths, MAX_PATHS_SHOWN)));
      }
      return paths;
    }

    /**
     * Extracts list of all executables associated with given collection of label
     * artifacts.
     *
     * @param artifacts to get the paths of
     * @param takeExecPath if false, the root relative path will be taken
     * @return all associated executable paths
     */
    private Set<String> getPaths(Collection<Artifact> artifacts, boolean takeExecPath) {
      TreeSet<String> paths = Sets.newTreeSet();
      for (Artifact artifact : artifacts) {
        PathFragment execPath =
            takeExecPath ? artifact.getExecPath() : artifact.getRootRelativePath();
        if (execPath != null) {  // omit middlemen etc
          paths.add(execPath.getCallablePathString());
        }
      }
      return paths;
    }

    private String joinPaths(Collection<String> paths) {
      return paths.stream().map(LocationFunction::quotePath).collect(joining(" "));
    }

    private static String quotePath(String path) {
      // TODO(ulfjack): Use existing ShellEscaper instead.
      if (path.contains(" ")) {
        path = "'" + path + "'";
      }
      return path;
    }

    private String functionName() {
      return multiple ? "$(locations)" : "$(location)";
    }
  }

  /**
   * Extracts all possible target locations from target specification.
   *
   * @param ruleContext BUILD target object
   * @param labelMap map of labels to build artifacts
   * @return map of all possible target locations
   */
  static Map<Label, Collection<Artifact>> buildLocationMap(
      RuleContext ruleContext,
      Map<Label, ? extends Collection<Artifact>> labelMap,
      boolean allowDataAttributeEntriesInLabel) {
    Map<Label, Collection<Artifact>> locationMap = Maps.newHashMap();
    if (labelMap != null) {
      for (Map.Entry<Label, ? extends Collection<Artifact>> entry : labelMap.entrySet()) {
        mapGet(locationMap, entry.getKey()).addAll(entry.getValue());
      }
    }

    // Add all destination locations.
    for (OutputFile out : ruleContext.getRule().getOutputFiles()) {
      mapGet(locationMap, out.getLabel()).add(ruleContext.createOutputArtifact(out));
    }

    if (ruleContext.getRule().isAttrDefined("srcs", BuildType.LABEL_LIST)) {
      for (TransitiveInfoCollection src : ruleContext
          .getPrerequisitesIf("srcs", Mode.TARGET, FileProvider.class)) {
        Iterables.addAll(mapGet(locationMap, AliasProvider.getDependencyLabel(src)),
            src.getProvider(FileProvider.class).getFilesToBuild());
      }
    }

    // Add all locations associated with dependencies and tools
    List<TransitiveInfoCollection> depsDataAndTools = new ArrayList<>();
    if (ruleContext.getRule().isAttrDefined("deps", BuildType.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisitesIf("deps", Mode.DONT_CHECK, FilesToRunProvider.class));
    }
    if (allowDataAttributeEntriesInLabel
        && ruleContext.getRule().isAttrDefined("data", BuildType.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisitesIf("data", Mode.DATA, FilesToRunProvider.class));
    }
    if (ruleContext.getRule().isAttrDefined("tools", BuildType.LABEL_LIST)) {
      Iterables.addAll(depsDataAndTools,
          ruleContext.getPrerequisitesIf("tools", Mode.HOST, FilesToRunProvider.class));
    }

    for (TransitiveInfoCollection dep : depsDataAndTools) {
      Label label = AliasProvider.getDependencyLabel(dep);
      FilesToRunProvider filesToRun = dep.getProvider(FilesToRunProvider.class);
      Artifact executableArtifact = filesToRun.getExecutable();

      // If the label has an executable artifact add that to the multimaps.
      if (executableArtifact != null) {
        mapGet(locationMap, label).add(executableArtifact);
      } else {
        Iterables.addAll(mapGet(locationMap, label), filesToRun.getFilesToRun());
      }
    }
    return locationMap;
  }

  /**
   * Returns the value in the specified map corresponding to 'key', creating and
   * inserting an empty container if absent. We use Map not Multimap because
   * we need to distinguish the cases of "empty value" and "absent key".
   *
   * @return the value in the specified map corresponding to 'key'
   */
  private static <K, V> Collection<V> mapGet(Map<K, Collection<V>> map, K key) {
    Collection<V> values = map.get(key);
    if (values == null) {
      // We use sets not lists, because it's conceivable that the same label
      // could appear twice, in "srcs" and "deps".
      values = Sets.newHashSet();
      map.put(key, values);
    }
    return values;
  }

  private static interface ErrorReporter {
    void report(String error);
  }

  private static final class AttributeErrorReporter implements ErrorReporter {
    private final RuleErrorConsumer delegate;
    private final String attrName;

    public AttributeErrorReporter(RuleErrorConsumer delegate, String attrName) {
      this.delegate = delegate;
      this.attrName = attrName;
    }

    @Override
    public void report(String error) {
      delegate.attributeError(attrName, error);
    }
  }

  private static final class RuleErrorReporter implements ErrorReporter {
    private final RuleErrorConsumer delegate;

    public RuleErrorReporter(RuleErrorConsumer delegate) {
      this.delegate = delegate;
    }

    @Override
    public void report(String error) {
      delegate.ruleError(error);
    }
  }
}
