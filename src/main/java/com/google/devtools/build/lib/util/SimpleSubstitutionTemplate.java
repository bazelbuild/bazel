package com.google.devtools.build.lib.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * A compiled template that substitutes variables in a string.
 */
public final class SimpleSubstitutionTemplate {

  private static final Pattern SUBSTITUTION_VARIABLE = Pattern.compile("\\{\\{([^}]*)\\}\\}");

  /**
   * A template consists of a sequence of {@link Part} objects that may be either variables or
   * literals.
   */
  private interface Part {

    String evaluate(VariableEvaluator evaluator);
  }

  private final ImmutableList<Part> parts;

  /**
   * @param templateText
   * @return A compiled template ready for evaluation.
   */
  public static SimpleSubstitutionTemplate parse(String templateText) {
    try {
      return parse(templateText, (ignored) -> {
      });
    } catch (InvalidVariableNameException e) {
      throw new RuntimeException(e); // Should never occur.
    }
  }

  public static SimpleSubstitutionTemplate parse(String templateText,
      VariableNameValidator validator)
      throws InvalidVariableNameException {
    ImmutableList.Builder<Part> partsBuilder = ImmutableList.builder();
    Matcher matcher = SUBSTITUTION_VARIABLE.matcher(templateText);
    int startOfNonVariablePart = 0;
    while (matcher.find()) {
      int endOfNonVariablePart = matcher.start();
      if (endOfNonVariablePart != startOfNonVariablePart) {
        String literal = templateText.substring(startOfNonVariablePart, endOfNonVariablePart);
        partsBuilder.add((ignored) -> literal);
      }
      String variableName = matcher.group(1);
      // Let validateVariable throw an exception if desired.
      validator.validateVariableName(variableName);
      partsBuilder.add((evaluator) -> evaluator.evaluatedVariable(variableName));
      startOfNonVariablePart = matcher.end();
    }
    int endOfNonVariablePart = templateText.length();
    if (endOfNonVariablePart != startOfNonVariablePart) {
      String literal = templateText.substring(startOfNonVariablePart, endOfNonVariablePart);
      partsBuilder.add((ignored) -> literal);
    }
    return new SimpleSubstitutionTemplate(partsBuilder.build());
  }

  /**
   * Evaluates the template, calling evaluator to replace each variable in the template.
   */
  public String evaluate(VariableEvaluator evaluator) {
    return parts.stream().map(p -> p.evaluate(evaluator)).collect(Collectors.joining());
  }

  /**
   * Interface for an object that checks variable names in a template.
   */
  public interface VariableNameValidator {

    /**
     * Validates the variable name, throwing an exception if the variable name is unexpected.
     */
    void validateVariableName(String name) throws InvalidVariableNameException;
  }

  /**
   * Used to validate that each template variable is from a known set. Each variable name has a
   * definition that can be used to assist the template writer.
   */
  public static final class VariableDefinitionSet {

    private VariableDefinitionSet(ImmutableMap<String, Definition> definitions) {
      this.definitions = definitions;
    }

    /**
     * Constructs a  {@link VariableDefinitionSet} from a set of definitions.
     */
    public static VariableDefinitionSet of(Definition... definitions) {
      //definitions

      return new VariableDefinitionSet(
          Arrays.stream(definitions).collect(ImmutableMap.toImmutableMap(
              definition -> definition.variableName,
              definition -> definition)));
    }

    /**
     * Returns a {@link VariableNameValidator} that will use this set of known variables.
     */
    public VariableNameValidator validator() {
      return name -> {
        if (!definitions.containsKey(name)) {
          String validNames = definitions.keySet().stream().sorted()
              .collect(Collectors.joining(", "));
          throw new InvalidVariableNameException(name,
              String.format("must be one of %s", validNames));

        }
      };
    }


    /**
     * A variable name and its meaning.
     */
    public static final class Definition {

      private final String variableName;
      private final String meaning;

      public Definition(String variableName, String meaning) {
        this.variableName = variableName;
        this.meaning = meaning;
      }
    }

    private final ImmutableMap<String, Definition> definitions;

  }

  /**
   * A function called to substitute a variable for another value at template execution time.
   */
  public interface VariableEvaluator {

    /**
     * Returns the value of the given variable in the evaluation context.
     */
    String evaluatedVariable(String variableName);
  }

  /**
   * Thrown when an invalid variable name is used in a template.
   */
  public static class InvalidVariableNameException extends Exception {

    public InvalidVariableNameException(String variable, String message) {
      super(String.format("invalid variable name %s: %s", variable, message));
    }
  }

  /**
   * A class to use for the value of the --experimental_workspace_rules_log_file flag.
   */
  public static final class LogPathFragmentTemplate {
    private static final String COMMAND_ID_VAR = "command_id";

    private static final VariableDefinitionSet VARIABLE_DEFINITION_SET =
        VariableDefinitionSet.of(
            new SimpleSubstitutionTemplate.VariableDefinitionSet.Definition(
                COMMAND_ID_VAR,
                "The UUID that Blaze uses to identify everything logged from the current "
                    + "build command."));

    private final SimpleSubstitutionTemplate stringTemplate;

    /**
     * Parses the template.
     */
    public static LogPathFragmentTemplate parse(String template)
        throws InvalidVariableNameException {
      return new LogPathFragmentTemplate(
          SimpleSubstitutionTemplate.parse(template, VARIABLE_DEFINITION_SET.validator()));
    }

    /**
     * Evaluate the template in the context of a given command.
     */
    public PathFragment evaluate(UUID commandId) {
      return PathFragment.create(stringTemplate.evaluate(variableName -> {
        if (COMMAND_ID_VAR.equals(variableName)) {
          return commandId.toString();
        }
        throw new RuntimeException("got unknown template variable; programming problem");
      }));
    }

    private LogPathFragmentTemplate(SimpleSubstitutionTemplate stringTemplate) {
      this.stringTemplate = stringTemplate;
    }
  }

  private SimpleSubstitutionTemplate(ImmutableList<Part> parts) {
    this.parts = parts;
  }
}
