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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.AssignmentStatement;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.Expression;
import com.google.devtools.build.lib.syntax.ExpressionStatement;
import com.google.devtools.build.lib.syntax.FunctionDefStatement;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.Statement;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.skylark.skylint.LocationRange.Location;
import java.util.AbstractMap;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** Utilities to extract and parse docstrings. */
public final class DocstringUtils {
  private DocstringUtils() {}

  /**
   * Collect all docstrings in the AST and store them in a map: name -> docstring.
   *
   * <p>Note that local variables can't have docstrings.
   *
   * @param ast the AST to traverse
   * @return a map from identifier names to their docstring; if there is a file-level docstring, its
   *     key is "".
   */
  static ImmutableMap<String, StringLiteral> collectDocstringLiterals(BuildFileAST ast) {
    ImmutableMap.Builder<String, StringLiteral> nameToDocstringLiteral = ImmutableMap.builder();
    Statement previousStatement = null;
    for (Statement currentStatement : ast.getStatements()) {
      Map.Entry<String, StringLiteral> entry =
          getNameAndDocstring(previousStatement, currentStatement);
      if (entry != null) {
        nameToDocstringLiteral.put(entry);
      }
      previousStatement = currentStatement;
    }
    return nameToDocstringLiteral.build();
  }

  @Nullable
  private static Map.Entry<String, StringLiteral> getNameAndDocstring(
      @Nullable Statement previousStatement, Statement currentStatement) {
    // function docstring:
    if (currentStatement instanceof FunctionDefStatement) {
      StringLiteral docstring =
          extractDocstring(((FunctionDefStatement) currentStatement).getStatements());
      if (docstring != null) {
        return new AbstractMap.SimpleEntry<>(
            ((FunctionDefStatement) currentStatement).getIdentifier().getName(), docstring);
      }
    } else {
      StringLiteral docstring = getStringLiteral(currentStatement);
      if (docstring != null) {
        if (previousStatement == null) {
          // file docstring:
          return new SimpleEntry<>("", docstring);
        } else {
          // variable docstring:
          String variable = getAssignedVariableName(previousStatement);
          if (variable != null) {
            return new SimpleEntry<>(variable, docstring);
          }
        }
      }
    }
    return null;
  }

  /** If the statement is an assignment to one variable, returns its name, or otherwise null. */
  @Nullable
  static String getAssignedVariableName(@Nullable Statement stmt) {
    if (stmt instanceof AssignmentStatement) {
      Expression lhs = ((AssignmentStatement) stmt).getLValue().getExpression();
      if (lhs instanceof Identifier) {
        return ((Identifier) lhs).getName();
      }
    }
    return null;
  }

  /** If the statement is a string literal, returns it, or otherwise null. */
  @Nullable
  static StringLiteral getStringLiteral(Statement stmt) {
    if (stmt instanceof ExpressionStatement) {
      Expression expr = ((ExpressionStatement) stmt).getExpression();
      if (expr instanceof StringLiteral) {
        return (StringLiteral) expr;
      }
    }
    return null;
  }

  /** Takes a function body and returns the docstring literal, if present. */
  @Nullable
  public static StringLiteral extractDocstring(List<Statement> statements) {
    if (statements.isEmpty()) {
      return null;
    }
    return getStringLiteral(statements.get(0));
  }

  /** Parses a docstring from a string literal and appends any new errors to the given list. */
  public static DocstringInfo parseDocstring(
      StringLiteral docstring, List<DocstringParseError> errors) {
    int indentation = docstring.getLocation().getStartLineAndColumn().getColumn() - 1;
    return parseDocstring(docstring.getValue(), indentation, errors);
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

  /** Encapsulates information about a Starlark function docstring. */
  public static class DocstringInfo {

    /** Returns the one-line summary of the docstring. */
    public String getSummary() {
      return summary;
    }

    /**
     * Returns a list containing information about parameter documentation for the parameters of the
     * documented function.
     */
    public List<ParameterDoc> getParameters() {
      return parameters;
    }

    /**
     * Returns the long-form description of the docstring. (Everything after the one-line summary
     * and before special sections such as "Args:".
     */
    public String getLongDescription() {
      return longDescription;
    }

    /** The one-line summary at the start of the docstring. */
    final String summary;
    /** Documentation of function parameters from the 'Args:' section. */
    final List<ParameterDoc> parameters;
    /** Documentation of the return value from the 'Returns:' section, or empty if there is none. */
    final String returns;
    /** Deprecation warning from the 'Deprecated:' section, or empty if there is none. */
    final String deprecated;
    /** Rest of the docstring that is not part of any of the special sections above. */
    final String longDescription;
    /** The texual location of the 'Arguments:' (not 'Args:') section in the function. */
    final LocationRange argumentsLocation;

    public DocstringInfo(
        String summary,
        List<ParameterDoc> parameters,
        String returns,
        String deprecated,
        String longDescription,
        LocationRange argumentsLocation) {
      this.summary = summary;
      this.parameters = ImmutableList.copyOf(parameters);
      this.returns = returns;
      this.deprecated = deprecated;
      this.longDescription = longDescription;
      this.argumentsLocation = argumentsLocation;
    }

    public boolean isSingleLineDocstring() {
      return longDescription.isEmpty() && parameters.isEmpty() && returns.isEmpty();
    }
  }

  /**
   * Contains information about the documentation for function parameters of a Starlark function.
   */
  public static class ParameterDoc {
    final String parameterName;
    final List<String> attributes; // e.g. a type annotation, "unused", "mutable"
    final String description;

    public ParameterDoc(String parameterName, List<String> attributes, String description) {
      this.parameterName = parameterName;
      this.attributes = ImmutableList.copyOf(attributes);
      this.description = description;
    }

    public String getParameterName() {
      return parameterName;
    }

    public List<String> getAttributes() {
      return attributes;
    }

    public String getDescription() {
      return description;
    }
  }

  private static class DocstringParser {
    private final String docstring;
    /** Start offset of the current line. */
    private int startOfLineOffset = 0;
    /** End offset of the current line. */
    private int endOfLineOffset = 0;
    /** Current line number within the docstring. */
    private int lineNumber = 0;
    /**
     * The indentation of the doctring literal in the source file.
     *
     * <p>Every line except the first one must be indented by at least that many spaces.
     */
    private int baselineIndentation = 0;
    /** Whether there was a blank line before the current line. */
    private boolean blankLineBefore = false;
    /** Whether we've seen a special section, e.g. 'Args:', already. */
    private boolean specialSectionsStarted = false;
    /** List of all parsed lines in the docstring so far, including all indentation. */
    private ArrayList<String> originalLines = new ArrayList<>();
    /**
     * The current line in the docstring with the baseline indentation removed.
     *
     * <p>If the indentation of a docstring line is less than the expected {@link
     * #baselineIndentation}, only the existing indentation is stripped; none of the remaining
     * characters are cut off.
     */
    private String line = "";
    /** Errors that occurred so far. */
    private final List<DocstringParseError> errors = new ArrayList<>();

    DocstringParser(String docstring, int indentation) {
      this.docstring = docstring;
      nextLine();
      // the indentation is only relevant for the following lines, not the first one:
      this.baselineIndentation = indentation;
    }

    /**
     * Move on to the next line and update the parser's internal state accordingly.
     *
     * @return whether there are lines remaining to be parsed
     */
    private boolean nextLine() {
      if (startOfLineOffset >= docstring.length()) {
        return false;
      }
      blankLineBefore = line.trim().isEmpty();
      startOfLineOffset = endOfLineOffset;
      if (startOfLineOffset >= docstring.length()) {
        // Previous line was the last; previous line had no trailing newline character.
        line = "";
        return false;
      }
      // If not the first line, advance start past the newline character. In the case where there is
      // no more content, then the previous line was the second-to-last line and this last line is
      // empty.
      if (docstring.charAt(startOfLineOffset) == '\n') {
        startOfLineOffset += 1;
      }
      lineNumber++;
      endOfLineOffset = docstring.indexOf('\n', startOfLineOffset);
      if (endOfLineOffset < 0) {
        endOfLineOffset = docstring.length();
      }
      String originalLine = docstring.substring(startOfLineOffset, endOfLineOffset);
      originalLines.add(originalLine);
      int indentation = getIndentation(originalLine);
      if (endOfLineOffset == docstring.length() && startOfLineOffset != 0) {
        if (!originalLine.trim().isEmpty()) {
          error("closing docstring quote should be on its own line, indented the same as the "
              + "opening quote");
        } else if (indentation != baselineIndentation) {
          error("closing docstring quote should be indented the same as the opening quote");
        }
      }
      if (originalLine.trim().isEmpty()) {
        line = "";
      } else {
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
        line = docstring.substring(startOfLineOffset, endOfLineOffset);
      }
      return true;
    }

    /**
     * Returns whether the current line is the last one in the docstring.
     *
     * <p>It is possible for both this function and {@link #eof} to return true if all content has
     * been exhausted, or if the last line is empty.
     */
    private boolean onLastLine() {
      return endOfLineOffset >= docstring.length();
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

    private void error(String message) {
      error(this.lineNumber, message);
    }

    private void error(int lineNumber, String message) {
      errors.add(new DocstringParseError(message, lineNumber, originalLines.get(lineNumber - 1)));
    }

    private void parseArgumentSection(
        List<ParameterDoc> params, String returns, String deprecated) {
      checkSectionStart(!params.isEmpty());
      if (!returns.isEmpty()) {
        error("'Args:' section should go before the 'Returns:' section");
      }
      if (!deprecated.isEmpty()) {
        error("'Args:' section should go before the 'Deprecated:' section");
      }
      params.addAll(parseParameters());
    }

    DocstringInfo parse() {
      String summary = line;
      String nonStandardDeprecation = checkForNonStandardDeprecation(line);
      if (!nextLine()) {
        return new DocstringInfo(
            summary, Collections.emptyList(), "", nonStandardDeprecation, "", null);
      }
      if (!line.isEmpty()) {
        error("the one-line summary should be followed by a blank line");
      } else {
        nextLine();
      }
      List<String> longDescriptionLines = new ArrayList<>();
      List<ParameterDoc> params = new ArrayList<>();
      String returns = "";
      String deprecated = "";
      boolean descriptionBodyAfterSpecialSectionsReported = false;
      LocationRange argumentsLocation = null;
      while (!eof()) {
        switch (line) {
          case "Args:":
            parseArgumentSection(params, returns, deprecated);
            break;
          case "Arguments:":
            // Setting the location indicates an issue will be reported.
            argumentsLocation =
                new LocationRange(
                    new Location(lineNumber, baselineIndentation + 1),
                    // 10 is the length of "Arguments:".
                    // The 1 is for the character after the base indentation.
                    new Location(lineNumber, baselineIndentation + 1 + 10));
            parseArgumentSection(params, returns, deprecated);
            break;
          case "Returns:":
            checkSectionStart(!returns.isEmpty());
            if (!deprecated.isEmpty()) {
              error("'Returns:' section should go before the 'Deprecated:' section");
            }
            returns = parseSectionAfterHeading();
            break;
          case "Deprecated:":
            checkSectionStart(!deprecated.isEmpty());
            deprecated = parseSectionAfterHeading();
            break;
          default:
            if (specialSectionsStarted && !descriptionBodyAfterSpecialSectionsReported) {
              error("description body should go before the special sections");
              descriptionBodyAfterSpecialSectionsReported = true;
            }
            if (deprecated.isEmpty() && nonStandardDeprecation.isEmpty()) {
              nonStandardDeprecation = checkForNonStandardDeprecation(line);
            }
            if (line.startsWith("Returns: ")) {
              error(
                  "the return value should be documented in a section, like this:\n\n"
                      + "Returns:\n"
                      + "  <documentation here>\n\n"
                      + "For more details, please have a look at the documentation.");
            }
            if (!(onLastLine() && line.trim().isEmpty())) {
              longDescriptionLines.add(line);
            }
            nextLine();
        }
      }
      if (deprecated.isEmpty()) {
        deprecated = nonStandardDeprecation;
      }
      return new DocstringInfo(
          summary,
          params,
          returns,
          deprecated,
          String.join("\n", longDescriptionLines),
          argumentsLocation);
    }

    private void checkSectionStart(boolean duplicateSection) {
      specialSectionsStarted = true;
      if (!blankLineBefore) {
        error("section should be preceded by a blank line");
      }
      if (duplicateSection) {
        error("duplicate '" + line + "' section");
      }
    }

    private String checkForNonStandardDeprecation(String line) {
      if (line.toLowerCase().startsWith("deprecated:") || line.contains("DEPRECATED")) {
        error(
            "use a 'Deprecated:' section for deprecations, similar to a 'Returns:' section:\n\n"
                + "Deprecated:\n"
                + "  <reason and alternative>\n\n"
                + "For more details, please have a look at the documentation.");
        return line;
      }
      return "";
    }

    private static final Pattern paramLineMatcher =
        Pattern.compile(
            "\\s*(?<name>[*\\w]+)( \\(\\s*(?<attributes>.*)\\s*\\))?: (?<description>.*)");

    private static final Pattern attributesSeparator = Pattern.compile("\\s*,\\s*");

    private List<ParameterDoc> parseParameters() {
      int sectionLineNumber = lineNumber;
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
        int paramLineNumber = lineNumber;
        trimmedLine = line.substring(actualIndentation);
        Matcher matcher = paramLineMatcher.matcher(trimmedLine);
        if (!matcher.matches()) {
          error(
              "invalid parameter documentation"
                  + " (expected format: \"parameter_name: documentation\")."
                  + " For more details, please have a look at the documentation.");
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
        String parameterDescription = description.toString().trim();
        if (parameterDescription.isEmpty()) {
          error(paramLineNumber, "empty parameter description for '" + parameterName + "'");
        }
        params.add(new ParameterDoc(parameterName, attributes, parameterDescription));
      }
      if (params.isEmpty()) {
        error(sectionLineNumber, "section is empty or badly formatted");
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
      int sectionLineNumber = lineNumber;
      nextLine();
      StringBuilder contents = new StringBuilder();
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
          contents.append('\n');
        }
        contents.append(trimmedLine);
        nextLine();
        firstLine = false;
      }
      String result = contents.toString().trim();
      if (result.isEmpty()) {
        error(sectionLineNumber, "section is empty");
      }
      return result;
    }
  }

  /** Contains error information to reflect a docstring parse error. */
  public static class DocstringParseError {
    final String message;
    final int lineNumber;
    final String line;

    public DocstringParseError(String message, int lineNumber, String line) {
      this.message = message;
      this.lineNumber = lineNumber;
      this.line = line;
    }

    @Override
    public String toString() {
      return lineNumber + ": " + message;
    }

    /** Returns a descriptive method about the error which occurred. */
    public String getMessage() {
      return message;
    }

    /** Returns the line number in the containing Starlark file which contains this error. */
    public int getLineNumber() {
      return lineNumber;
    }
  }
}
