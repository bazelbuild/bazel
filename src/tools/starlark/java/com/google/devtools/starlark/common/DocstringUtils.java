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

package com.google.devtools.starlark.common;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Utilities to extract and parse docstrings. */
public final class DocstringUtils {

  private DocstringUtils() {} // uninstantiable

  /**
   * Parses a trimmed docstring.
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
   * <p>We expect the docstring to already be trimmed and dedented to a minimal common indentation
   * level by {@link Starlark#trimDocString} or an equivalent PEP-257 style trim() implementation;
   * note that {@link StarlarkFunction#getDocumentation} returns a correctly trimmed and dedented
   * doc string.
   *
   * @param doc a docstring of the format described above
   * @param parseErrors a list to which parsing error messages are written
   * @return the parsed docstring information
   */
  public static DocstringInfo parseDocstring(String doc, List<DocstringParseError> parseErrors) {
    DocstringParser parser = new DocstringParser(doc);
    DocstringInfo result = parser.parse();
    parseErrors.addAll(parser.errors);
    return result;
  }

  /** Encapsulates information about a Starlark function docstring. */
  public static class DocstringInfo {

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

    public DocstringInfo(
        String summary,
        List<ParameterDoc> parameters,
        String returns,
        String deprecated,
        String longDescription) {
      this.summary = summary;
      this.parameters = ImmutableList.copyOf(parameters);
      this.returns = returns;
      this.deprecated = deprecated;
      this.longDescription = longDescription;
    }


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

    /**
     * Returns the deprecation warning from the 'Deprecated:' section, or empty if there is none.
     */
    public String getDeprecated() {
      return deprecated;
    }

    public boolean isSingleLineDocstring() {
      return longDescription.isEmpty() && parameters.isEmpty() && returns.isEmpty();
    }

    /**
     * Returns the documentation of the return value from the 'Returns:' section, or empty if
     * there is none.
     */
    public String getReturns() {
      return returns;
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
    /**
     * Current line number within the docstring, 1-based; 0 indicates that parsing has not started;
     * {@code lineNumber > lines.size()} indicates EOF.
     */
    private int lineNumber = 0;

    /** Whether there was a blank line before the current line. */
    private boolean blankLineBefore = false;

    /** Whether we've seen a special section, e.g. 'Args:', already. */
    private boolean specialSectionsStarted = false;

    /** List of all parsed lines in the docstring. */
    private final ImmutableList<String> lines;

    /** Iterator over lines. */
    private final Iterator<String> linesIter;

    /** The current line in the docstring. */
    private String line = "";

    /** Errors that occurred so far. */
    private final List<DocstringParseError> errors = new ArrayList<>();

    private DocstringParser(String docstring) {
      this.lines = ImmutableList.copyOf(Splitter.on("\n").split(docstring));
      this.linesIter = lines.iterator();
      // Load the summary line
      nextLine();
    }

    /**
     * Move on to the next line and update the parser's internal state accordingly.
     *
     * @return whether there are lines remaining to be parsed
     */
    private boolean nextLine() {
      if (linesIter.hasNext()) {
        blankLineBefore = line.trim().isEmpty();
        line = linesIter.next();
        lineNumber++;
        return true;
      } else {
        line = "";
        lineNumber = lines.size() + 1;
        return false;
      }
    }

    private boolean eof() {
      return lineNumber > lines.size();
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
      errors.add(new DocstringParseError(message, lineNumber, lines.get(lineNumber - 1)));
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
        return new DocstringInfo(summary, Collections.emptyList(), "", nonStandardDeprecation, "");
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
      while (!eof()) {
        switch (line) {
          case "Args:":
            parseArgumentSection(params, returns, deprecated);
            break;
          case "Arguments:":
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
            longDescriptionLines.add(line);
            nextLine();
        }
      }
      if (deprecated.isEmpty()) {
        deprecated = nonStandardDeprecation;
      }
      return new DocstringInfo(
          summary, params, returns, deprecated, String.join("\n", longDescriptionLines));
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
      if (line.toLowerCase(Locale.ROOT).startsWith("deprecated:") || line.contains("DEPRECATED")) {
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
      // Two iterations: first buffer lines and find the minimal indent, then trim to the min
      List<String> buffer = new ArrayList<>();
      int continuationIndentation = Integer.MAX_VALUE;
      while (nextLine()) {
        if (!line.isEmpty()) {
          if (getIndentation(line) <= baselineIndentation) {
            break;
          }
          continuationIndentation = Math.min(getIndentation(line), continuationIndentation);
        }
        buffer.add(line);
      }

      for (String bufLine : buffer) {
        description.append('\n');
        if (!bufLine.isEmpty()) {
          String trimmedLine = bufLine.substring(continuationIndentation);
          description.append(trimmedLine);
        }
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

    /**
     * Returns the line number (skipping leading blank lines, if any) in the original doc string
     * which contains this error.
     */
    public int getLineNumber() {
      return lineNumber;
    }

    /** Returns the contents of the original line that caused the parse error. */
    public String getLine() {
      return line;
    }
  }
}
