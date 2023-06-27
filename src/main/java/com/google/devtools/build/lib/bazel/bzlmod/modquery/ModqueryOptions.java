// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.bzlmod.modquery;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.bzlmod.Version;
import com.google.devtools.build.lib.bazel.bzlmod.Version.ParseException;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.CommaSeparatedNonEmptyOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.List;
import javax.annotation.Nullable;

/** Options for ModqueryCommand */
public class ModqueryOptions extends OptionsBase {

  @Option(
      name = "from",
      defaultValue = "root",
      converter = TargetModuleListConverter.class,
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "The module(s) starting from which the dependency graph query will be displayed. Check"
              + " each query’s description for the exact semantic. Defaults to root.\n")
  public ImmutableList<TargetModule> modulesFrom;

  @Option(
      name = "extra",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "The queries will also display the reason why modules were resolved to their current"
              + " version (if changed). Defaults to true only for the explain query.")
  public boolean extra;

  @Option(
      name = "include_unused",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "The queries will also take into account and display the unused modules, which are not"
              + " present in the module resolution graph after selection (due to the"
              + " Minimal-Version Selection or override rules). This can have different effects for"
              + " each of the query types i.e. include new paths in the all_paths command, or extra"
              + " dependants in the explain command.\n")
  public boolean includeUnused;

  @Option(
      name = "depth",
      defaultValue = "-1",
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Maximum display depth of the dependency tree. A depth of 1 displays the direct"
              + " dependencies, for example. For tree, path and all_paths it defaults to"
              + " Integer.MAX_VALUE, while for deps and explain it defaults to 1 (only displays"
              + " direct deps of the root besides the target leaves and their parents).\n")
  public int depth;

  @Option(
      name = "cycles",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "Points out dependency cycles inside the displayed tree, which are normally ignored by"
              + " default.")
  public boolean cycles;

  @Option(
      name = "charset",
      defaultValue = "utf8",
      converter = CharsetConverter.class,
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Chooses the character set to use for the tree. Only affects text output. Valid values"
              + " are \"utf8\" or \"ascii\". Default is \"utf8\"")
  public Charset charset;

  @Option(
      name = "output",
      defaultValue = "text",
      converter = OutputFormatConverter.class,
      documentationCategory = OptionDocumentationCategory.MODQUERY,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "The format in which the query results should be printed. Allowed values for query are: "
              + "text, json, graph")
  public OutputFormat outputFormat;

  /**
   * Possible subcommands that can be specified for {@link
   * com.google.devtools.build.lib.bazel.commands.ModqueryCommand}
   */
  public enum QueryType {
    DEPS(1),
    TREE(0),
    ALL_PATHS(1),
    PATH(1),
    EXPLAIN(1),
    SHOW(1);

    /* Store the number of arguments that it accepts for easy pre-check */
    private final int argNumber;

    QueryType(int argNumber) {
      this.argNumber = argNumber;
    }

    @Override
    public String toString() {
      return Ascii.toLowerCase(this.name());
    }

    public int getArgNumber() {
      return argNumber;
    }

    public static String printValues() {
      return "(" + stream(values()).map(QueryType::toString).collect(joining(", ")) + ")";
    }
  }

  /** Converts a query type option string to a properly typed {@link QueryType} */
  public static class QueryTypeConverter extends EnumConverter<QueryType> {
    public QueryTypeConverter() {
      super(QueryType.class, "query type");
    }
  }

  /**
   * Charset to be used in outputting the {@link
   * com.google.devtools.build.lib.bazel.commands.ModqueryCommand} result.
   */
  public enum Charset {
    UTF8,
    ASCII
  }

  /** Converts a charset option string to a properly typed {@link Charset} */
  public static class CharsetConverter extends EnumConverter<Charset> {
    public CharsetConverter() {
      super(Charset.class, "output charset");
    }
  }

  /**
   * Possible formats of the {@link com.google.devtools.build.lib.bazel.commands.ModqueryCommand}
   * result.
   */
  public enum OutputFormat {
    TEXT,
    JSON,
    GRAPH
  }

  /** Converts an output format option string to a properly typed {@link OutputFormat} */
  public static class OutputFormatConverter extends EnumConverter<OutputFormat> {
    public OutputFormatConverter() {
      super(OutputFormat.class, "output format");
    }
  }

  /** Argument of a modquery converted from the form name@version or name. */
  @AutoValue
  public abstract static class TargetModule {
    static TargetModule create(String name, Version version) {
      return new AutoValue_ModqueryOptions_TargetModule(name, version);
    }

    public abstract String getName();

    /**
     * If it is null, it represents any (one or multiple) present versions of the module in the dep
     * graph, which is different from the empty version
     */
    @Nullable
    public abstract Version getVersion();
  }

  /** Converts a module target argument string to a properly typed {@link TargetModule} */
  static class TargetModuleConverter extends Converter.Contextless<TargetModule> {

    @Override
    public TargetModule convert(String input) throws OptionsParsingException {
      String errorMessage = String.format("Cannot parse the given module argument: %s.", input);
      Preconditions.checkArgument(input != null);
      // The keyword root takes priority if any module is named the same it can only be referenced
      // using the full key.
      if (Ascii.equalsIgnoreCase(input, "root")) {
        return TargetModule.create("", Version.EMPTY);
      } else {
        List<String> splits = Splitter.on('@').splitToList(input);
        if (splits.isEmpty() || splits.get(0).isEmpty()) {
          throw new OptionsParsingException(errorMessage);
        }

        if (splits.size() == 2) {
          if (splits.get(1).equals("_")) {
            return TargetModule.create(splits.get(0), Version.EMPTY);
          }
          if (splits.get(1).isEmpty()) {
            throw new OptionsParsingException(errorMessage);
          }
          try {
            return TargetModule.create(splits.get(0), Version.parse(splits.get(1)));
          } catch (ParseException e) {
            throw new OptionsParsingException(errorMessage, e);
          }

        } else if (splits.size() == 1) {
          return TargetModule.create(splits.get(0), null);
        } else {
          throw new OptionsParsingException(errorMessage);
        }
      }
    }

    @Override
    public String getTypeDescription() {
      return "root, <module>@<version> or <module>";
    }
  }

  /** Converts a comma-separated module list argument (i.e. A@1.0,B@2.0) */
  public static class TargetModuleListConverter
      extends Converter.Contextless<ImmutableList<TargetModule>> {

    @Override
    public ImmutableList<TargetModule> convert(String input) throws OptionsParsingException {
      CommaSeparatedNonEmptyOptionListConverter listConverter =
          new CommaSeparatedNonEmptyOptionListConverter();
      TargetModuleConverter targetModuleConverter = new TargetModuleConverter();
      ImmutableList<String> targetStrings =
          listConverter.convert(input, /*conversionContext=*/ null);
      ImmutableList.Builder<TargetModule> targetModules = new ImmutableList.Builder<>();
      for (String targetInput : targetStrings) {
        targetModules.add(targetModuleConverter.convert(targetInput, /*conversionContext=*/ null));
      }
      return targetModules.build();
    }

    @Override
    public String getTypeDescription() {
      return "a list of <module>s separated by comma";
    }
  }

  static ModqueryOptions getDefaultOptions() {
    ModqueryOptions options = new ModqueryOptions();
    options.depth = Integer.MAX_VALUE;
    options.cycles = false;
    options.includeUnused = false;
    options.extra = false;
    options.modulesFrom = ImmutableList.of(TargetModule.create("", Version.EMPTY));
    options.charset = Charset.UTF8;
    options.outputFormat = OutputFormat.TEXT;
    return options;
  }
}
