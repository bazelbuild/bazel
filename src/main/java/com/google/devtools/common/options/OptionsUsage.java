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

import com.google.common.base.Ascii;
import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import java.text.BreakIterator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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
      getUsage(optionDefinition, usage, OptionsParser.HelpVerbosity.LONG, data, false);
    }
  }

  /** Appends the usage message for a single option-field message to 'usage'. */
  static void getUsage(
      OptionDefinition optionDefinition,
      StringBuilder usage,
      OptionsParser.HelpVerbosity helpVerbosity,
      OptionsData optionsData,
      boolean includeTags) {
    String flagName = getFlagName(optionDefinition);
    String typeDescription = getTypeDescription(optionDefinition);
    usage.append("  --").append(flagName);
    if (helpVerbosity == OptionsParser.HelpVerbosity.SHORT) {
      usage.append('\n');
      return;
    }

    // Add the option's type and default information. Stop there for "medium" verbosity.
    if (optionDefinition.getAbbreviation() != '\0') {
      usage.append(" [-").append(optionDefinition.getAbbreviation()).append(']');
    }
    if (!typeDescription.isEmpty()) {
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
    if (helpVerbosity == OptionsParser.HelpVerbosity.MEDIUM) {
      return;
    }

    // For verbosity "long," add the full description and expansion, along with the tag
    // information if requested.
    if (!optionDefinition.getHelpText().isEmpty()) {
      usage.append(paragraphFill(optionDefinition.getHelpText(), /*indent=*/ 4, /*width=*/ 80));
      usage.append('\n');
    }
    ImmutableList<String> expansion = optionsData.getEvaluatedExpansion(optionDefinition);
    if (!expansion.isEmpty()) {
      StringBuilder expandsMsg = new StringBuilder("Expands to: ");
      for (String exp : expansion) {
        expandsMsg.append(exp).append(" ");
      }
      usage.append(paragraphFill(expandsMsg.toString(), /*indent=*/ 6, /*width=*/ 80));
      usage.append('\n');
    }
    if (optionDefinition.hasImplicitRequirements()) {
      StringBuilder requiredMsg = new StringBuilder("Using this option will also add: ");
      for (String req : optionDefinition.getImplicitRequirements()) {
        requiredMsg.append(req).append(" ");
      }
      usage.append(paragraphFill(requiredMsg.toString(), 6, 80)); // (indent, width)
      usage.append('\n');
    }
    if (!includeTags) {
      return;
    }

    // If we are expected to include the tags, add them for high verbosity.
    Stream<OptionEffectTag> effectTagStream =
        Arrays.stream(optionDefinition.getOptionEffectTags())
            .filter(OptionsUsage::shouldEffectTagBeListed);
    Stream<OptionMetadataTag> metadataTagStream =
        Arrays.stream(optionDefinition.getOptionMetadataTags())
            .filter(OptionsUsage::shouldMetadataTagBeListed);
    String tagList =
        Stream.concat(effectTagStream, metadataTagStream)
            .map(tag -> Ascii.toLowerCase(tag.toString()))
            .collect(Collectors.joining(", "));
    if (!tagList.isEmpty()) {
      usage.append(paragraphFill("Tags: " + tagList, 6, 80)); // (indent, width)
      usage.append("\n");
    }
  }

  /**
   * Paragraph-fill the specified input text, indenting lines to 'indent' and
   * wrapping lines at 'width'.  Returns the formatted result.
   */
  static String paragraphFill(String in, int indent, int width) {
    String indentString = " ".repeat(indent);
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

  // Placeholder tag "UNKNOWN" is ignored.
  static boolean shouldEffectTagBeListed(OptionEffectTag effectTag) {
    return !effectTag.equals(OptionEffectTag.UNKNOWN);
  }

  // Tags that only apply to undocumented options are excluded.
  static boolean shouldMetadataTagBeListed(OptionMetadataTag metadataTag) {
    return !metadataTag.equals(OptionMetadataTag.HIDDEN)
        && !metadataTag.equals(OptionMetadataTag.INTERNAL);
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
      builder
          .append("={")
          .append(COMMA_JOINER.join(fieldType.getEnumConstants()).toLowerCase(Locale.ENGLISH))
          .append("}\n");
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

  static String getTypeDescription(OptionDefinition optionsDefinition) {
    return optionsDefinition.getConverter().getTypeDescription();
  }

  static String getFlagName(OptionDefinition optionDefinition) {
    String name = optionDefinition.getOptionName();
    return optionDefinition.usesBooleanValueSyntax() ? "[no]" + name : name;
  }
}
