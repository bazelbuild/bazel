// Copyright 2014 Google Inc. All rights reserved.
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

import static com.google.devtools.common.options.OptionsParserImpl.findConverter;

import com.google.common.base.Splitter;
import com.google.common.base.Strings;
import com.google.common.collect.Lists;

import java.lang.reflect.Field;
import java.text.BreakIterator;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * A renderer for usage messages. For now this is very simple.
 */
class OptionsUsage {

  private static final Splitter NEWLINE_SPLITTER = Splitter.on('\n');

  /**
   * Given an options class, render the usage string into the usage,
   * which is passed in as an argument.
   */
  static void getUsage(Class<? extends OptionsBase> optionsClass, StringBuilder usage) {
    List<Field> optionFields =
        Lists.newArrayList(OptionsParser.getAllAnnotatedFields(optionsClass));
    Collections.sort(optionFields, BY_NAME);
    for (Field optionField : optionFields) {
      getUsage(optionField, usage, OptionsParser.HelpVerbosity.LONG);
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
   * Append the usage message for a single option-field message to 'usage'.
   */
  static void getUsage(Field optionField, StringBuilder usage,
                       OptionsParser.HelpVerbosity helpVerbosity) {
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
    if (annotation.expansion().length > 0) {
      StringBuilder expandsMsg = new StringBuilder("Expands to: ");
      for (String exp : annotation.expansion()) {
        expandsMsg.append(exp).append(" ");
      }
      usage.append(paragraphFill(expandsMsg.toString(), 4, 80)); // (indent, width)
      usage.append('\n');
    }
  }

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
    return findConverter(optionsField).getTypeDescription();
  }

  static String getFlagName(Field field) {
    String name = field.getAnnotation(Option.class).name();
    return OptionsParserImpl.isBooleanField(field) ? "[no]" + name : name;
  }

}
