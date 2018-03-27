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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.CommonQueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import java.util.List;
import java.util.Set;

/** Command-line options for the Blaze query language, revision 2. */
public class QueryOptions extends CommonQueryOptions {
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
    name = "proto:output_rule_attrs",
    converter = CommaSeparatedOptionListConverter.class,
    defaultValue = "all",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
    help =
        "Comma separated list of attributes to include in output. Defaults to all attributes. "
            + "Set to empty string to not output any attribute. "
            + "This option is applicable to --output=proto."
  )
  public List<String> protoOutputRuleAttributes = ImmutableList.of("all");

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
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS, OptionEffectTag.EAGERNESS_TO_EXIT},
    help =
        "If true, the tests() expression gives an error if it encounters a test_suite containing "
            + "non-test targets."
  )
  public boolean strictTestSuite;

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

  @Option(
    name = "proto:flatten_selects",
    defaultValue = "true",
    category = "query",
    documentationCategory = OptionDocumentationCategory.QUERY,
    effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
    help =
        "If enabled, configurable attributes created by select() are flattened. For list types "
            + "the flattened representation is a list containing each value of the select map "
            + "exactly once. Scalar types are flattened to null."
  )
  public boolean protoFlattenSelects;

  /** Return the current options as a set of QueryEnvironment settings. */
  @Override
  public Set<Setting> toSettings() {
    Set<Setting> settings = super.toSettings();
    if (strictTestSuite) {
      settings.add(Setting.TESTS_EXPRESSION_STRICT);
    }
    return settings;
  }
}
