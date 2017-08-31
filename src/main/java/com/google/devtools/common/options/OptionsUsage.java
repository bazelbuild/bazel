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
package com.google.devtools.common.options;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.escape.Escaper;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/** A renderer for usage messages for any combination of options classes. */
class OptionsUsage {

  private static final Splitter NEWLINE_SPLITTER = Splitter.on('\n');
  private static final Joiner COMMA_JOINER = Joiner.on(",");

  /**
   * Given an options class, render the usage string into the usage, which is passed in as an
   * argument. This will not include information about expansions for options using expansion
   * functions (it would be unsafe to report this as we cannot know what options from other {@link
   * OptionsBase} subclasses they depend on until a complete parser is constructed).
   */
  static void getUsage(Class<? extends OptionsBase> optionsClass, StringBuilder usage) {
    OptionsData data = OptionsParser.getOptionsDataInternal(optionsClass);
    List<OptionDefinition> optionDefinitions =
        new ArrayList<>(OptionsData.getAllOptionDefinitionsForClass(optionsClass));
    optionDefinitions.sort(OptionDefinition.BY_OPTION_NAME);
    for (OptionDefinition optionDefinition : optionDefinitions) {
      getUsage(optionDefinition, usage, OptionsParser.HelpVerbosity.LONG, data);
    }
  }

  /**
   * Paragraph-fill the specified input text, indenting lines to 'indent' and
   * wrapping lines at 'width'.  Returns the formatted result.
   */
  static String paragraphFill(String in, int indent, int width) {
    String indentString = Strings.repeat(" ", indent);
    StringBuilder out = new StringBuilder();
    String sep = "";
    for (String paragraph : NEWLINE_SPLITTER.split(in)) {
      // TODO(ccalvarin) break iterators expect hyphenated words to be line-breakable, which looks
      // funny for --flag
      BreakIterator boundary = BreakIterator.getLineInstance(); // (factory)
      boundary.setText(paragraph);
      out.append(sep).append(indentString);
      int cursor = indent;
      for (int start = boundary.first(), end = boundary.next();
           end != BreakIterator.DONE;
           start = end, end = boundary.next()) {
        String word =
            paragraph.substring(start, end); // (may include trailing space)
        if (word.length() + cursor > width) {
          out.append('\n').append(indentString);
          cursor = indent;
        }
        out.append(word);
        cursor += word.length();
      }
      sep = "\n";
    }
    return out.toString();
  }

  /**
   * Returns the expansion for an option, if any, regardless of if the expansion is from a function
   * or is statically declared in the annotation.
   */
  private static @Nullable ImmutableList<String> getExpansionIfKnown(
      OptionDefinition optionDefinition, OptionsData optionsData) {
    Preconditions.checkNotNull(optionDefinition);
    try {
      return optionsData.getEvaluatedExpansion(optionDefinition, null);
    } catch (ExpansionNeedsValueException e) {
      return null;
    } catch (OptionsParsingException e) {
      throw new IllegalStateException("Error expanding void expansion function: ", e);
    }

  }

  /** Appends the usage message for a single option-field message to 'usage'. */
  static void getUsage(
      OptionDefinition optionDefinition,
      StringBuilder usage,
      OptionsParser.HelpVerbosity helpVerbosity,
      OptionsData optionsData) {
    String flagName = getFlagName(optionDefinition);
    String typeDescription = getTypeDescription(optionDefinition);
    usage.append("  --").append(flagName);
    if (helpVerbosity == OptionsParser.HelpVerbosity.SHORT) { // just the name
      usage.append('\n');
      return;
    }
    if (optionDefinition.getAbbreviation() != '\0') {
      usage.append(" [-").append(optionDefinition.getAbbreviation()).append(']');
    }
    if (!typeDescription.equals("")) {
      usage.append(" (").append(typeDescription).append("; ");
      if (optionDefinition.allowsMultiple()) {
        usage.append("may be used multiple times");
      } else {
        // Don't call the annotation directly (we must allow overrides to certain defaults)
        String defaultValueString = optionDefinition.getUnparsedDefaultValue();
        if (optionDefinition.isSpecialNullDefault()) {
          usage.append("default: see description");
        } else {
          usage.append("default: \"").append(defaultValueString).append("\"");
        }
      }
      usage.append(")");
    }
    usage.append("\n");
    if (helpVerbosity == OptionsParser.HelpVerbosity.MEDIUM) { // just the name and type.
      return;
    }
    if (!optionDefinition.getHelpText().isEmpty()) {
      usage.append(paragraphFill(optionDefinition.getHelpText(), /*indent=*/ 4, /*width=*/ 80));
      usage.append('\n');
    }
    ImmutableList<String> expansion = getExpansionIfKnown(optionDefinition, optionsData);
    if (expansion == null) {
      usage.append(paragraphFill("Expands to unknown options.", /*indent=*/ 6, /*width=*/ 80));
      usage.append('\n');
    } else if (!expansion.isEmpty()) {
      StringBuilder expandsMsg = new StringBuilder("Expands to: ");
      for (String exp : expansion) {
        expandsMsg.append(exp).append(" ");
      }
      usage.append(paragraphFill(expandsMsg.toString(), /*indent=*/ 6, /*width=*/ 80));
      usage.append('\n');
    }
    if (optionDefinition.getImplicitRequirements().length > 0) {
      StringBuilder requiredMsg = new StringBuilder("Using this option will also add: ");
      for (String req : optionDefinition.getImplicitRequirements()) {
        requiredMsg.append(req).append(" ");
      }
      usage.append(paragraphFill(requiredMsg.toString(), /*indent=*/ 6, /*width=*/ 80));
      usage.append('\n');
    }
  }

  /** Append the usage message for a single option-field message to 'usage'. */
  static void getUsageHtml(
      OptionDefinition optionDefinition,
      StringBuilder usage,
      Escaper escaper,
      OptionsData optionsData) {
    String plainFlagName = optionDefinition.getOptionName();
    String flagName = getFlagName(optionDefinition);
    String valueDescription = optionDefinition.getValueTypeHelpText();
    String typeDescription = getTypeDescription(optionDefinition);
    usage.append("<dt><code><a name=\"flag--").append(plainFlagName).append("\"></a>--");
    usage.append(flagName);
    if (optionDefinition.isBooleanField() || optionDefinition.isVoidField()) {
      // Nothing for boolean, tristate, boolean_or_enum, or void options.
    } else if (!valueDescription.isEmpty()) {
      usage.append("=").append(escaper.escape(valueDescription));
    } else if (!typeDescription.isEmpty()) {
      // Generic fallback, which isn't very good.
      usage.append("=&lt;").append(escaper.escape(typeDescription)).append("&gt");
    }
    usage.append("</code>");
    if (optionDefinition.getAbbreviation() != '\0') {
      usage.append(" [<code>-").append(optionDefinition.getAbbreviation()).append("</code>]");
    }
    if (optionDefinition.allowsMultiple()) {
      // Allow-multiple options can't have a default value.
      usage.append(" multiple uses are accumulated");
    } else {
      // Don't call the annotation directly (we must allow overrides to certain defaults).
      String defaultValueString = optionDefinition.getUnparsedDefaultValue();
      if (optionDefinition.isVoidField()) {
        // Void options don't have a default.
      } else if (optionDefinition.isSpecialNullDefault()) {
        usage.append(" default: see description");
      } else {
        usage.append(" default: \"").append(escaper.escape(defaultValueString)).append("\"");
      }
    }
    usage.append("</dt>\n");
    usage.append("<dd>\n");
    if (!optionDefinition.getHelpText().isEmpty()) {
      usage.append(
          paragraphFill(
              escaper.escape(optionDefinition.getHelpText()), /*indent=*/ 0, /*width=*/ 80));
      usage.append('\n');
    }

    if (!optionsData.getExpansionDataForField(optionDefinition).isEmpty()) {
      // If this is an expansion option, list the expansion if known, or at least specify that we
      // don't know.
      usage.append("<br/>\n");
      ImmutableList<String> expansion = getExpansionIfKnown(optionDefinition, optionsData);
      StringBuilder expandsMsg;
      if (expansion == null) {
        expandsMsg = new StringBuilder("Expands to unknown options.<br/>\n");
      } else {
        Preconditions.checkArgument(!expansion.isEmpty());
        expandsMsg = new StringBuilder("Expands to:<br/>\n");
        for (String exp : expansion) {
          // TODO(ulfjack): Can we link to the expanded flags here?
          expandsMsg
              .append("&nbsp;&nbsp;<code>")
              .append(escaper.escape(exp))
              .append("</code><br/>\n");
        }
      }
      usage.append(expandsMsg.toString());
    }

    usage.append("</dd>\n");
  }

  /**
   * Returns the available completion for the given option field. The completions are the exact
   * command line option (with the prepending '--') that one should pass. It is suitable for
   * completion script to use. If the option expect an argument, the kind of argument is given
   * after the equals. If the kind is a enum, the various enum values are given inside an accolade
   * in a comma separated list. For other special kind, the type is given as a name (e.g.,
   * <code>label</code>, <code>float</ode>, <code>path</code>...). Example outputs of this
   * function are for, respectively, a tristate flag <code>tristate_flag</code>, a enum
   * flag <code>enum_flag</code> which can take <code>value1</code>, <code>value2</code> and
   * <code>value3</code>, a path fragment flag <code>path_flag</code>, a string flag
   * <code>string_flag</code> and a void flag <code>void_flag</code>:
   * <pre>
   *   --tristate_flag={auto,yes,no}
   *   --notristate_flag
   *   --enum_flag={value1,value2,value3}
   *   --path_flag=path
   *   --string_flag=
   *   --void_flag
   * </pre>
   *
   * @param optionDefinition The field to return completion for
   * @param builder the string builder to store the completion values
   */
  static void getCompletion(OptionDefinition optionDefinition, StringBuilder builder) {
    // Return the list of possible completions for this option
    String flagName = optionDefinition.getOptionName();
    Class<?> fieldType = optionDefinition.getType();
    builder.append("--").append(flagName);
    if (fieldType.equals(boolean.class)) {
      builder.append("\n");
      builder.append("--no").append(flagName).append("\n");
    } else if (fieldType.equals(TriState.class)) {
      builder.append("={auto,yes,no}\n");
      builder.append("--no").append(flagName).append("\n");
    } else if (fieldType.isEnum()) {
      builder.append("={")
          .append(COMMA_JOINER.join(fieldType.getEnumConstants()).toLowerCase()).append("}\n");
    } else if (fieldType.getSimpleName().equals("Label")) {
      // String comparison so we don't introduce a dependency to com.google.devtools.build.lib.
      builder.append("=label\n");
    } else if (fieldType.getSimpleName().equals("PathFragment")) {
      builder.append("=path\n");
    } else if (Void.class.isAssignableFrom(fieldType)) {
      builder.append("\n");
    } else {
      // TODO(bazel-team): add more types. Maybe even move the completion type
      // to the @Option annotation?
      builder.append("=\n");
    }
  }

  private static String getTypeDescription(OptionDefinition optionsDefinition) {
    return optionsDefinition.getConverter().getTypeDescription();
  }

  static String getFlagName(OptionDefinition optionDefinition) {
    String name = optionDefinition.getOptionName();
    return optionDefinition.isBooleanField() ? "[no]" + name : name;
  }
}
