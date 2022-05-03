package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.util.SimpleSubstitutionTemplate.InvalidVariableNameException;
import com.google.devtools.build.lib.util.SimpleSubstitutionTemplate.VariableDefinitionSet;
import com.google.devtools.build.lib.util.SimpleSubstitutionTemplate.VariableEvaluator;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SimpleSubstitutionTemplateTest {

  @Test
  public void parse() {
    assertThat(SimpleSubstitutionTemplate.parse(
            "my {{relation}}, {{friend}} {{BAD}}!")
        .evaluate((variableName -> {
          if (variableName.equals("relation")) {
            return "dog";
          }
          if (variableName.equals("friend")) {
            return "Sam";
          }
          return "INVALID";
        }))).isEqualTo("my dog, Sam INVALID!");
  }

  @Test
  public void parseWithValidator() throws Exception {
    VariableDefinitionSet set = VariableDefinitionSet.of(
        new VariableDefinitionSet.Definition("name", "name is the name of the person"),
        new VariableDefinitionSet.Definition("environment", "a place of some kind"));

    ImmutableMap<String, String> substitutions = ImmutableMap.of(
        "name", "Sally");
    assertThat(SimpleSubstitutionTemplate.parse("hello, {{name}}.", set.validator())
        .evaluate(evaluatorForMap(substitutions)))
        .isEqualTo("hello, Sally.");

    InvalidVariableNameException exception = assertThrows(
        InvalidVariableNameException.class,
        () -> SimpleSubstitutionTemplate.parse("hello, {{name}} {{ahem}}.", set.validator()));
    assertThat(exception.getMessage()).isEqualTo(
        "invalid variable name ahem: must be one of environment, name");
  }

  private static VariableEvaluator evaluatorForMap(Map<String, String> map) {
    return variableName -> map.getOrDefault(variableName, "MISSING");
  }
}
