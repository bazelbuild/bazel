// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.skylark.skylint;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ExpressionStatement;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** Utilities to extract and parse docstrings. */
public final class DocstringUtils {
  private DocstringUtils() {}

  @Nullable
  static StringLiteral extractDocstring(List<Statement> statements) {
    if (statements.isEmpty()) {
      return null;
    }
    Statement statement = statements.get(0);
    if (statement instanceof ExpressionStatement) {
      Expression expr = ((ExpressionStatement) statement).getExpression();
      if (expr instanceof StringLiteral) {
        return (StringLiteral) expr;
      }
    }
    return null;
  }

  /**
   * Parses a docstring.
   *
   * <p>The format of the docstring is as follows
   *
   * <pre>{@code
   * """One-line summary: must be followed and may be preceded by a blank line.
   *
   * Optional additional description like this.
   *
   * If it's a function docstring and the function has more than one argument, the docstring has
   * to document these parameters as follows:
   *
   * Args:
   *   parameter1: description of the first parameter
   *   parameter2: description of the second
   *     parameter that spans two lines. Each additional line
   *     must be indented by (at least) two spaces
   *   another_parameter (unused, mutable): a parameter may be followed
   *     by additional attributes in parentheses
   * """
   * }</pre>
   *
   * @param docstring a docstring of the format described above
   * @param parseErrors a list to which parsing error messages are written
   * @return the parsed docstring information
   */
  static DocstringInfo parseDocString(String docstring, List<DocstringParseError> parseErrors) {
    DocstringParser parser = new DocstringParser(docstring);
    DocstringInfo result = parser.parse();
    parseErrors.addAll(parser.errors);
    return result;
  }

  static class DocstringInfo {
    final String summary;
    final List<ParameterDoc> parameters;
    final String longDescription;

    public DocstringInfo(String summary, List<ParameterDoc> parameters, String longDescription) {
      this.summary = summary;
      this.parameters = ImmutableList.copyOf(parameters);
      this.longDescription = longDescription;
    }
  }

  static class ParameterDoc {
    final String parameterName;
    final List<String> attributes; // e.g. a type annotation, "unused", "mutable"
    final String description;

    public ParameterDoc(String parameterName, List<String> attributes, String description) {
      this.parameterName = parameterName;
      this.attributes = ImmutableList.copyOf(attributes);
      this.description = description;
    }
  }

  private static class DocstringParser {
    private final String docstring;
    private int startOfLineOffset = 0;
    private int endOfLineOffset = -1;
    private int lineNumber = 0;
    private int expectedIndentation = 0;
    private String line = "";
    private final List<DocstringParseError> errors = new ArrayList<>();

    DocstringParser(String docstring) {
      this.docstring = docstring;
      nextLine();
    }

    boolean nextLine() {
      if (startOfLineOffset >= docstring.length()) {
        return false;
      }
      lineNumber++;
      startOfLineOffset = endOfLineOffset + 1;
      if (startOfLineOffset >= docstring.length()) {
        line = "";
        return false;
      }
      endOfLineOffset = docstring.indexOf('\n', startOfLineOffset);
      if (endOfLineOffset < 0) {
        endOfLineOffset = docstring.length();
      }
      line = docstring.substring(startOfLineOffset, endOfLineOffset);
      int indentation = getIndentation(line);
      if (!line.isEmpty()) {
        if (indentation < expectedIndentation) {
          error(
              "line indented too little (here: "
                  + indentation
                  + " spaces; before: "
                  + expectedIndentation
                  + " spaces)");
          expectedIndentation = indentation;
        }
        startOfLineOffset += expectedIndentation;
      }
      line = docstring.substring(startOfLineOffset, endOfLineOffset);
      return true;
    }

    private static int getIndentation(String line) {
      int index = 0;
      while (index < line.length() && line.charAt(index) == ' ') {
        index++;
      }
      return index;
    }

    void error(String message) {
      errors.add(new DocstringParseError(message, lineNumber));
    }

    DocstringInfo parse() {
      String summary = line;
      if (!nextLine()) {
        return new DocstringInfo(summary, Collections.emptyList(), "");
      }
      if (!line.isEmpty()) {
        error("the one-line summary should be followed by a blank line");
      } else {
        nextLine();
      }
      expectedIndentation = getIndentation(line);
      line = line.substring(expectedIndentation);
      List<String> longDescriptionLines = new ArrayList<>();
      List<ParameterDoc> params = new ArrayList<>();
      boolean sectionStart = true;
      do {
        if (line.startsWith(" ")) {
          error(
              "line indented too much (here: "
                  + (expectedIndentation + getIndentation(line))
                  + " spaces; expected: "
                  + expectedIndentation
                  + " spaces)");
        }
        if (line.equals("Args:")) {
          if (!sectionStart) {
            error("section should be preceded by a blank line");
          }
          if (!params.isEmpty()) {
            error("parameters were already documented before");
          }
          params.addAll(parseParameters());
        } else {
          longDescriptionLines.add(line);
        }
        sectionStart = line.isEmpty();
      } while (nextLine());
      return new DocstringInfo(summary, params, String.join("\n", longDescriptionLines));
    }

    private static final Pattern paramLineMatcher =
        Pattern.compile(
            "\\s*(?<name>[*\\w]+)( \\(\\s*(?<attributes>.*)\\s*\\))?: (?<description>.*)");

    private static final Pattern attributesSeparator = Pattern.compile("\\s*,\\s*");

    private List<ParameterDoc> parseParameters() {
      nextLine();
      List<ParameterDoc> params = new ArrayList<>();
      while (!line.isEmpty()) {
        if (getIndentation(line) != 2) {
          error("parameter lines have to be indented by two spaces");
        } else {
          line = line.substring(2);
        }
        Matcher matcher = paramLineMatcher.matcher(line);
        if (!matcher.matches()) {
          error("invalid parameter documentation");
          nextLine();
          continue;
        }
        String parameterName = Preconditions.checkNotNull(matcher.group("name"));
        String attributesString = matcher.group("attributes");
        StringBuilder description = new StringBuilder(matcher.group("description"));
        List<String> attributes =
            attributesString == null
                ? Collections.emptyList()
                : Arrays.asList(attributesSeparator.split(attributesString));
        while (nextLine() && getIndentation(line) > 2) {
          description.append('\n');
          description.append(line, getIndentation(line), line.length());
        }
        params.add(new ParameterDoc(parameterName, attributes, description.toString()));
      }
      return params;
    }
  }

  static class DocstringParseError {
    final String message;
    final int lineNumber;

    public DocstringParseError(String message, int lineNumber) {
      this.message = message;
      this.lineNumber = lineNumber;
    }

    @Override
    public String toString() {
      return ":" + lineNumber + ": " + message;
    }
  }
}
