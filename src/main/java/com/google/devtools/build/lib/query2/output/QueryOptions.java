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
package com.google.devtools.build.lib.query2.output;

import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Command-line options for the Blaze query language, revision 2.
 */
public class QueryOptions extends OptionsBase {
  /** An enum converter for {@code  AspectResolver.Mode} . Should be used internally only. */
  public static class AspectResolutionModeConverter extends EnumConverter<AspectResolver.Mode> {
    public AspectResolutionModeConverter() {
      super(AspectResolver.Mode.class, "Aspect resolution mode");
    }
  }

  /** An enum converter for {@code OrderOutput} . Should be used internally only. */
  public static class OrderOutputConverter extends EnumConverter<OrderOutput> {
    public OrderOutputConverter() {
      super(OrderOutput.class, "Order output setting");
    }
  }

  @Option(
    name = "output",
    defaultValue = "label",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "The format in which the query results should be printed. Allowed values are: "
            + "label, label_kind, minrank, maxrank, package, location, graph, xml, proto."
  )
  public String outputFormat;

  @Option(
    name = "null",
    defaultValue = "null",
    category = "query",
    expansion = {"--line_terminator_null=true"},
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help = "Whether each format is terminated with \0 instead of newline."
  )
  public Void isNull;

  @Option(
    name = "line_terminator_null",
    defaultValue = "false",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help = "Whether each format is terminated with \0 instead of newline."
  )
  public boolean lineTerminatorNull;

  @Option(
    name = "order_results",
    defaultValue = "null",
    category = "query",
    deprecationWarning = "Please use --order_output=auto or --order_output=no instead of this flag",
    expansion = {"--order_output=auto"},
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "Output the results in dependency-ordered (default) or unordered fashion. The "
            + "unordered output is faster but only supported when --output is not minrank, "
            + "maxrank, or graph."
  )
  public Void orderResults;

  @Option(
    name = "noorder_results",
    defaultValue = "null",
    category = "query",
    deprecationWarning = "Please use --order_output=no or --order_output=auto instead of this flag",
    expansion = {"--order_output=no"},
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "Output the results in dependency-ordered (default) or unordered fashion. The "
            + "unordered output is faster but only supported when --output is not minrank, "
            + "maxrank, or graph."
  )
  public Void noOrderResults;

  /** Whether and how output should be ordered. */
  public enum OrderOutput {
    NO, /** Make no effort to order output besides that required by output formatter. */
    DEPS, /** Output in dependency order when compatible with output formatter. */
    AUTO, /** Same as full unless formatter is proto, minrank, maxrank, or graph, then deps. */
    FULL /** Output in dependency order, breaking ties with alphabetical order when needed. */
  }

  @Option(
    name = "order_output",
    converter = OrderOutputConverter.class,
    defaultValue = "auto",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "Output the results unordered (no), dependency-ordered (deps), or fully ordered (full). "
            + "The default is 'auto', meaning that results are output either dependency-ordered or "
            + "fully ordered, depending on the output formatter (dependency-ordered for proto, "
            + "minrank, maxrank, and graph, fully ordered for all others). When output is fully "
            + "ordered, nodes that would otherwise be unordered by the output formatter are "
            + "alphabetized before output."
  )
  public OrderOutput orderOutput;

  @Option(
    name = "keep_going",
    abbrev = 'k',
    defaultValue = "false",
    category = "strategy",
    documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
    effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT},
    help =
        "Continue as much as possible after an error. While the target that failed, and those "
            + "that depend on it, cannot be analyzed, other prerequisites of these targets can be."
  )
  public boolean keepGoing;

  @Option(
    name = "loading_phase_threads",
    defaultValue = "200",
    documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
    effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
    help = "Number of parallel threads to use for the loading phase."
  )
  public int loadingPhaseThreads;

  @Option(
    name = "host_deps",
    defaultValue = "true",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "If enabled, dependencies on 'host configuration' targets will be included in the "
            + "dependency graph over which the query operates.  A 'host configuration' dependency "
            + "edge, such as the one from any 'proto_library' rule to the Protocol Compiler, "
            + "usually points to a tool executed during the build (on the host machine) rather "
            + "than a part of the same 'target' program.  Queries whose purpose is to discover "
            + "the set of things needed during a build will typically enable this option; queries "
            + "aimed at revealing the structure of a single program will typically disable this "
            + "option."
  )
  public boolean includeHostDeps;

  @Option(
    name = "implicit_deps",
    defaultValue = "true",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "If enabled, implicit dependencies will be included in the dependency graph over "
            + "which the query operates. An implicit dependency is one that is not explicitly "
            + "specified in the BUILD file but added by blaze."
  )
  public boolean includeImplicitDeps;

  @Option(
    name = "graph:node_limit",
    defaultValue = "512",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "The maximum length of the label string for a graph node in the output.  Longer labels"
            + " will be truncated; -1 means no truncation.  This option is only applicable to"
            + " --output=graph."
  )
  public int graphNodeStringLimit;

  @Option(
    name = "graph:factored",
    defaultValue = "true",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "If true, then the graph will be emitted 'factored', i.e. topologically-equivalent nodes "
            + "will be merged together and their labels concatenated. This option is only "
            + "applicable to --output=graph."
  )
  public boolean graphFactored;

  @Option(
    name = "proto:default_values",
    defaultValue = "true",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "If true, attributes whose value is not explicitly specified in the BUILD file are "
            + "included; otherwise they are omitted. This option is applicable to --output=proto"
  )
  public boolean protoIncludeDefaultValues;

  @Option(
    name = "xml:line_numbers",
    defaultValue = "true",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "If true, XML output contains line numbers. Disabling this option may make diffs easier "
            + "to read.  This option is only applicable to --output=xml."
  )
  public boolean xmlLineNumbers;

  @Option(
    name = "xml:default_values",
    defaultValue = "false",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "If true, rule attributes whose value is not explicitly specified in the BUILD file are "
            + "printed; otherwise they are omitted."
  )
  public boolean xmlShowDefaultValues;

  @Option(
    name = "strict_test_suite",
    defaultValue = "false",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "If true, the tests() expression gives an error if it encounters a test_suite containing "
            + "non-test targets."
  )
  public boolean strictTestSuite;

  @Option(
    name = "universe_scope",
    converter = Converters.CommaSeparatedOptionListConverter.class,
    defaultValue = "",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.CHANGES_INPUTS},
    help =
        "A comma-separated set of target patterns (additive and subtractive). The query may "
            + "be performed in the universe defined by the transitive closure of the specified "
            + "targets."
  )
  public List<String> universeScope;

  @Option(
    name = "relative_locations",
    defaultValue = "false",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "If true, the location of BUILD files in xml and proto outputs will be relative. "
            + "By default, the location output is an absolute path and will not be consistent "
            + "across machines. You can set this option to true to have a consistent result "
            + "across machines."
  )
  public boolean relativeLocations;

  @Option(
    name = "aspect_deps",
    converter = AspectResolutionModeConverter.class,
    defaultValue = "conservative",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "How to resolve aspect dependencies when the output format is one of {xml,proto,record}. "
            + "'off' means no aspect dependencies are resolved, 'conservative' (the default) means "
            + "all declared aspect dependencies are added regardless of whether they are viable "
            + "given the rule class of direct dependencies, 'precise' means that only those "
            + "aspects are added that are possibly active given the rule class of the direct "
            + "dependencies. Note that precise mode requires loading other packages to evaluate "
            + "a single target thus making it slower than the other modes. Also note that even "
            + "precise mode is not completely precise: the decision whether to compute an aspect "
            + "is decided in the analysis phase, which is not run during 'blaze query'."
  )
  public AspectResolver.Mode aspectDeps;

  @Option(
    name = "query_file",
    defaultValue = "",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.CHANGES_INPUTS},
    help =
        "If set, query will read the query from the file named here, rather than on the command "
            + "line. It is an error to specify a file here as well as a command-line query."
  )
  public String queryFile;

  /** Ugly workaround since line terminator option default has to be constant expression. */
  public String getLineTerminator() {
    if (lineTerminatorNull) {
      return "\0";
    }

    return System.lineSeparator();
  }

  /** Return the current options as a set of QueryEnvironment settings. */
  public Set<Setting> toSettings() {
    Set<Setting> settings = EnumSet.noneOf(Setting.class);
    if (strictTestSuite) {
      settings.add(Setting.TESTS_EXPRESSION_STRICT);
    }
    if (!includeHostDeps) {
      settings.add(Setting.NO_HOST_DEPS);
    }
    if (!includeImplicitDeps) {
      settings.add(Setting.NO_IMPLICIT_DEPS);
    }
    return settings;
  }
}
