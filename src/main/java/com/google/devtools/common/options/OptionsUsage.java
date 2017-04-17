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
import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;
import com.google.common.escape.Escaper;
import java.lang.reflect.Field;
import java.text.BreakIterator;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A renderer for usage messages. For now this is very simple.
 */
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
    List<Field> optionFields =
        Lists.newArrayList(OptionsParser.getAllAnnotatedFields(optionsClass));
    Collections.sort(optionFields, BY_NAME);
    for (Field optionField : optionFields) {
      getUsage(optionField, usage, OptionsParser.HelpVerbosity.LONG, null);
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
   * Returns the expansion for an option, to the extent known. Precisely, if an {@link OptionsData}
   * object is supplied, the expansion is read from that. Otherwise, the annotation is inspected: If
   * the annotation uses {@link Option#expansion} it is returned, and if it uses {@link
   * Option#expansionFunction} null is returned, indicating a lack of definite information. In all
   * cases, when the option is not an expansion option, an empty array is returned.
   */
  private static @Nullable String[] getExpansionIfKnown(
      Field optionField, Option annotation, @Nullable OptionsData optionsData) {
    if (optionsData != null) {
      return optionsData.getEvaluatedExpansion(optionField);
    } else {
      if (OptionsData.usesExpansionFunction(annotation)) {
        return null;
      } else {
        // Empty array if it's not an expansion option.
        return annotation.expansion();
      }
    }
  }

  /**
   * Appends the usage message for a single option-field message to 'usage'. If {@code optionsData}
   * is not supplied, options that use expansion functions won't be fully described.
   */
  static void getUsage(
      Field optionField,
      StringBuilder usage,
      OptionsParser.HelpVerbosity helpVerbosity,
      @Nullable OptionsData optionsData) {
    String flagName = getFlagName(optionField);
    String typeDescription = getTypeDescription(optionField);
    Option annotation = optionField.getAnnotation(Option.class);
    usage.append("  --" + flagName);
    if (helpVerbosity == OptionsParser.HelpVerbosity.SHORT) { // just the name
      usage.append('\n');
      return;
    }
    if (annotation.abbrev() != '\0') {
      usage.append(" [-").append(annotation.abbrev()).append(']');
    }
    if (!typeDescription.equals("")) {
      usage.append(" (" + typeDescription + "; ");
      if (annotation.allowMultiple()) {
        usage.append("may be used multiple times");
      } else {
        // Don't call the annotation directly (we must allow overrides to certain defaults)
        String defaultValueString = OptionsParserImpl.getDefaultOptionString(optionField);
        if (OptionsParserImpl.isSpecialNullDefault(defaultValueString, optionField)) {
          usage.append("default: see description");
        } else {
          usage.append("default: \"" + defaultValueString + "\"");
        }
      }
      usage.append(")");
    }
    usage.append("\n");
    if (helpVerbosity == OptionsParser.HelpVerbosity.MEDIUM) { // just the name and type.
      return;
    }
    if (!annotation.help().equals("")) {
      usage.append(paragraphFill(annotation.help(), 4, 80)); // (indent, width)
      usage.append('\n');
    }
    String[] expansion = getExpansionIfKnown(optionField, annotation, optionsData);
    if (expansion == null) {
      usage.append("    Expands to unknown options.\n");
    } else if (expansion.length > 0) {
      StringBuilder expandsMsg = new StringBuilder("Expands to: ");
      for (String exp : expansion) {
        expandsMsg.append(exp).append(" ");
      }
      usage.append(paragraphFill(expandsMsg.toString(), 4, 80)); // (indent, width)
      usage.append('\n');
    }
  }

  /**
   * Append the usage message for a single option-field message to 'usage'. If {@code optionsData}
   * is not supplied, options that use expansion functions won't be fully described.
   */
  static void getUsageHtml(
      Field optionField, StringBuilder usage, Escaper escaper, @Nullable OptionsData optionsData) {
    String plainFlagName = optionField.getAnnotation(Option.class).name();
    String flagName = getFlagName(optionField);
    String valueDescription = optionField.getAnnotation(Option.class).valueHelp();
    String typeDescription = getTypeDescription(optionField);
    Option annotation = optionField.getAnnotation(Option.class);
    usage.append("<dt><code><a name=\"flag--").append(plainFlagName).append("\"></a>--");
    usage.append(flagName);
    if (OptionsData.isBooleanField(optionField) || OptionsData.isVoidField(optionField)) {
      // Nothing for boolean, tristate, boolean_or_enum, or void options.
    } else if (!valueDescription.isEmpty()) {
      usage.append("=").append(escaper.escape(valueDescription));
    } else if (!typeDescription.isEmpty()) {
      // Generic fallback, which isn't very good.
      usage.append("=&lt;").append(escaper.escape(typeDescription)).append("&gt");
    }
    usage.append("</code>");
    if (annotation.abbrev() != '\0') {
      usage.append(" [<code>-").append(annotation.abbrev()).append("</code>]");
    }
    if (annotation.allowMultiple()) {
      // Allow-multiple options can't have a default value.
      usage.append(" multiple uses are accumulated");
    } else {
      // Don't call the annotation directly (we must allow overrides to certain defaults).
      String defaultValueString = OptionsParserImpl.getDefaultOptionString(optionField);
      if (OptionsData.isVoidField(optionField)) {
        // Void options don't have a default.
      } else if (OptionsParserImpl.isSpecialNullDefault(defaultValueString, optionField)) {
        usage.append(" default: see description");
      } else {
        usage.append(" default: \"").append(escaper.escape(defaultValueString)).append("\"");
      }
    }
    usage.append("</dt>\n");
    usage.append("<dd>\n");
    if (!annotation.help().isEmpty()) {
      usage.append(paragraphFill(escaper.escape(annotation.help()), 0, 80)); // (indent, width)
      usage.append('\n');
    }
    String[] expansion = getExpansionIfKnown(optionField, annotation, optionsData);
    if (expansion == null) {
      usage.append("    Expands to unknown options.<br>\n");
    } else if (expansion.length > 0) {
      usage.append("<br/>\n");
      StringBuilder expandsMsg = new StringBuilder("Expands to:<br/>\n");
      for (String exp : expansion) {
        // TODO(ulfjack): Can we link to the expanded flags here?
        expandsMsg
            .append("&nbsp;&nbsp;<code>")
            .append(escaper.escape(exp))
            .append("</code><br/>\n");
      }
      usage.append(expandsMsg.toString()); // (indent, width)
      usage.append('\n');
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
   * @param field The field to return completion for
   * @param builder the string builder to store the completion values
   */
  static void getCompletion(Field field, StringBuilder builder) {
    // Return the list of possible completions for this option
    String flagName = field.getAnnotation(Option.class).name();
    Class<?> fieldType = field.getType();
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

  // TODO(brandjon): Should this use sorting by option name instead of field name?
  private static final Comparator<Field> BY_NAME = new Comparator<Field>() {
    @Override
    public int compare(Field left, Field right) {
      return left.getName().compareTo(right.getName());
    }
  };

  /**
   * An ordering relation for option-field fields that first groups together
   * options of the same category, then sorts by name within the category.
   */
  static final Comparator<Field> BY_CATEGORY = new Comparator<Field>() {
    @Override
    public int compare(Field left, Field right) {
      int r = left.getAnnotation(Option.class).category().compareTo(
              right.getAnnotation(Option.class).category());
      return r == 0 ? BY_NAME.compare(left, right) : r;
    }
  };

  private static String getTypeDescription(Field optionsField) {
    return OptionsData.findConverter(optionsField).getTypeDescription();
  }

  static String getFlagName(Field field) {
    String name = field.getAnnotation(Option.class).name();
    return OptionsData.isBooleanField(field) ? "[no]" + name : name;
  }

}
