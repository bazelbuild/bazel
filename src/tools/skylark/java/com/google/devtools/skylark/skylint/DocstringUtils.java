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
   *   parameter1: description of the first parameter. Each parameter line
   *     should be indented by one, preferably two, spaces (as here).
   *   parameter2: description of the second
   *     parameter that spans two lines. Each additional line should have a
   *     hanging indentation of at least one, preferably two, additional spaces (as here).
   *   another_parameter (unused, mutable): a parameter may be followed
   *     by additional attributes in parentheses
   *
   * Returns:
   *   Description of the return value.
   *   Should be indented by at least one, preferably two spaces (as here)
   *   Can span multiple lines.
   * """
   * }</pre>
   *
   * @param docstring a docstring of the format described above
   * @param indentation the indentation level (number of spaces) of the docstring
   * @param parseErrors a list to which parsing error messages are written
   * @return the parsed docstring information
   */
  static DocstringInfo parseDocstring(
      String docstring, int indentation, List<DocstringParseError> parseErrors) {
    DocstringParser parser = new DocstringParser(docstring, indentation);
    DocstringInfo result = parser.parse();
    parseErrors.addAll(parser.errors);
    return result;
  }

  static class DocstringInfo {
    final String summary;
    final List<ParameterDoc> parameters;
    final String returns;
    final String longDescription;

    public DocstringInfo(
        String summary, List<ParameterDoc> parameters, String returns, String longDescription) {
      this.summary = summary;
      this.parameters = ImmutableList.copyOf(parameters);
      this.returns = returns;
      this.longDescription = longDescription;
    }

    public boolean isSingleLineDocstring() {
      return longDescription.isEmpty() && parameters.isEmpty() && returns.isEmpty();
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
    private int baselineIndentation = 0;
    private boolean blankLineBefore = false;
    private String line = "";
    private final List<DocstringParseError> errors = new ArrayList<>();

    DocstringParser(String docstring, int indentation) {
      this.docstring = docstring;
      nextLine();
      // the indentation is only relevant for the following lines, not the first one:
      this.baselineIndentation = indentation;
    }

    boolean nextLine() {
      if (startOfLineOffset >= docstring.length()) {
        return false;
      }
      blankLineBefore = line.isEmpty();
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
        if (indentation < baselineIndentation) {
          error(
              "line indented too little (here: "
                  + indentation
                  + " spaces; expected: "
                  + baselineIndentation
                  + " spaces)");
          startOfLineOffset += indentation;
        } else {
          startOfLineOffset += baselineIndentation;
        }
      }
      line = docstring.substring(startOfLineOffset, endOfLineOffset);
      return true;
    }

    private boolean eof() {
      return startOfLineOffset >= docstring.length();
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
        return new DocstringInfo(summary, Collections.emptyList(), "", "");
      }
      if (!line.isEmpty()) {
        error("the one-line summary should be followed by a blank line");
      } else {
        nextLine();
      }
      List<String> longDescriptionLines = new ArrayList<>();
      List<ParameterDoc> params = new ArrayList<>();
      String returns = "";
      while (!eof()) {
        switch (line) {
          case "Args:":
            if (!blankLineBefore) {
              error("section should be preceded by a blank line");
            }
            if (!params.isEmpty()) {
              error("parameters were already documented before");
            }
            if (!returns.isEmpty()) {
              error("parameters should be documented before the return value");
            }
            params.addAll(parseParameters());
            break;
          case "Returns:":
            if (!blankLineBefore) {
              error("section should be preceded by a blank line");
            }
            if (!returns.isEmpty()) {
              error("return value was already documented before");
            }
            returns = parseSectionAfterHeading();
            break;
          default:
            longDescriptionLines.add(line);
            nextLine();
        }
      }
      return new DocstringInfo(summary, params, returns, String.join("\n", longDescriptionLines));
    }

    private static final Pattern paramLineMatcher =
        Pattern.compile(
            "\\s*(?<name>[*\\w]+)( \\(\\s*(?<attributes>.*)\\s*\\))?: (?<description>.*)");

    private static final Pattern attributesSeparator = Pattern.compile("\\s*,\\s*");

    private List<ParameterDoc> parseParameters() {
      nextLine();
      List<ParameterDoc> params = new ArrayList<>();
      int expectedParamLineIndentation = -1;
      while (!eof()) {
        if (line.isEmpty()) {
          nextLine();
          continue;
        }
        int actualIndentation = getIndentation(line);
        if (actualIndentation == 0) {
          if (!blankLineBefore) {
            error("end of 'Args' section without blank line");
          }
          break;
        }
        String trimmedLine;
        if (expectedParamLineIndentation == -1) {
          expectedParamLineIndentation = actualIndentation;
        }
        if (expectedParamLineIndentation != actualIndentation) {
          error(
              "inconsistent indentation of parameter lines (before: "
                  + expectedParamLineIndentation
                  + "; here: "
                  + actualIndentation
                  + " spaces)");
        }
        trimmedLine = line.substring(actualIndentation);
        Matcher matcher = paramLineMatcher.matcher(trimmedLine);
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
        parseContinuedParamDescription(actualIndentation, description);
        params.add(new ParameterDoc(parameterName, attributes, description.toString().trim()));
      }
      return params;
    }

    /** Parses additional lines that can come after "param: foo" in an 'Args' section. */
    private void parseContinuedParamDescription(
        int baselineIndentation, StringBuilder description) {
      while (nextLine()) {
        if (line.isEmpty()) {
          description.append('\n');
          continue;
        }
        if (getIndentation(line) <= baselineIndentation) {
          break;
        }
        String trimmedLine = line.substring(baselineIndentation);
        description.append('\n');
        description.append(trimmedLine);
      }
    }

    private String parseSectionAfterHeading() {
      nextLine();
      StringBuilder returns = new StringBuilder();
      boolean firstLine = true;
      while (!eof()) {
        String trimmedLine;
        if (line.isEmpty()) {
          trimmedLine = line;
        } else if (getIndentation(line) == 0) {
          if (!blankLineBefore) {
            error("end of section without blank line");
          }
          break;
        } else {
          if (getIndentation(line) < 2) {
            error(
                "text in a section has to be indented by two spaces"
                    + " (relative to the left margin of the docstring)");
            trimmedLine = line.substring(getIndentation(line));
          } else {
            trimmedLine = line.substring(2);
          }
        }
        if (!firstLine) {
          returns.append('\n');
        }
        returns.append(trimmedLine);
        nextLine();
        firstLine = false;
      }
      return returns.toString().trim();
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
