// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkinterface.processor;

import static com.google.common.truth.Truth.assertAbout;
import static com.google.testing.compile.JavaSourceSubjectFactory.javaSource;

import com.google.common.io.Resources;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.testing.compile.JavaFileObjects;
import javax.tools.JavaFileObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for SkylarkCallableProcessor.
 */
@RunWith(JUnit4.class)
public final class SkylarkCallableProcessorTest {

  private static JavaFileObject getFile(String pathToFile) {
    return JavaFileObjects.forResource(Resources.getResource(
        SkylarkCallableProcessorTest.class, "testsources/" + pathToFile));
  }

  @Test
  public void testGoldenCase() throws Exception {
    assertAbout(javaSource())
        .that(getFile("GoldenCase.java"))
        .processedWith(new SkylarkCallableProcessor())
        .compilesWithoutError();
  }

  @Test
  public void testPrivateMethod() throws Exception {
    assertAbout(javaSource())
        .that(getFile("PrivateMethod.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkCallable annotated methods must be public.");
  }

  @Test
  public void testStaticMethod() throws Exception {
    assertAbout(javaSource())
        .that(getFile("StaticMethod.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkCallable annotated methods cannot be static.");
  }


  @Test
  public void testStructFieldWithArguments() throws Exception {
    assertAbout(javaSource())
        .that(getFile("StructFieldWithArguments.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable annotated methods with structField=true must have 0 user-supplied "
                + "parameters. Expected 0 extra interpreter parameters, "
                + "but found 1 total parameters.");
  }

  @Test
  public void testStructFieldWithInvalidInfo() throws Exception {
    assertAbout(javaSource())
        .that(getFile("StructFieldWithInvalidInfo.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable-annotated methods with structField=true may not also specify "
                + "useStarlarkThread, extraPositionals, or extraKeywords");
  }

  @Test
  public void testStructFieldWithExtraArgs() throws Exception {
    assertAbout(javaSource())
        .that(getFile("StructFieldWithExtraArgs.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable-annotated methods with structField=true may not also specify "
                + "useStarlarkThread, extraPositionals, or extraKeywords");
  }

  @Test
  public void testStructFieldWithExtraKeywords() throws Exception {
    assertAbout(javaSource())
        .that(getFile("StructFieldWithExtraKeywords.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable-annotated methods with structField=true may not also specify "
                + "useStarlarkThread, extraPositionals, or extraKeywords");
  }

  @Test
  public void testDocumentationMissing() throws Exception {
    assertAbout(javaSource())
        .that(getFile("DocumentationMissing.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("The 'doc' string must be non-empty if 'documented' is true.");
  }

  @Test
  public void testArgumentMissing() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ArgumentMissing.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable annotated method has 0 parameters, "
                + "but annotation declared 1 user-supplied parameters "
                + "and 0 extra interpreter parameters.");
  }

  @Test
  public void testStarlarkThreadMissing() throws Exception {
    assertAbout(javaSource())
        .that(getFile("StarlarkThreadMissing.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Expected parameter index 2 to be the "
                + StarlarkThread.class.getCanonicalName()
                + " type, matching useStarlarkThread, but was java.lang.String");
  }

  @Test
  public void testLocationMissing() throws Exception {
    assertAbout(javaSource())
        .that(getFile("LocationMissing.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Expected parameter index 2 to be the "
                + Location.class.getCanonicalName()
                + " type, matching useLocation, but was java.lang.String");
  }

  @Test
  public void testSkylarkInfoBeforeParams() throws Exception {
    assertAbout(javaSource())
        .that(getFile("SkylarkInfoBeforeParams.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Expected parameter index 3 to be the "
                + Location.class.getCanonicalName()
                + " type, matching useLocation, but was java.lang.Integer");
  }

  @Test
  public void testSkylarkInfoParamsWrongOrder() throws Exception {
    assertAbout(javaSource())
        .that(getFile("SkylarkInfoParamsWrongOrder.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Expected parameter index 1 to be the "
                + Location.class.getCanonicalName()
                + " type, matching useLocation, but was "
                + StarlarkThread.class.getCanonicalName());
  }

  @Test
  public void testTooManyArguments() throws Exception {
    assertAbout(javaSource())
        .that(getFile("TooManyArguments.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable annotated method has 2 parameters, "
                + "but annotation declared 1 user-supplied parameters "
                + "and 0 extra interpreter parameters.");
  }

  @Test
  public void testInvalidParamNoneDefault() throws Exception {
    assertAbout(javaSource())
        .that(getFile("InvalidParamNoneDefault.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Parameter 'a_parameter' has 'None' default value but is not noneable.");
  }

  @Test
  public void testParamTypeConflict() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ParamTypeConflict.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Parameter 'a_parameter' has both 'type' and 'allowedTypes' specified."
                + " Only one may be specified.");
  }

  @Test
  public void testParamNeitherNamedNorPositional() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ParamNeitherNamedNorPositional.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Parameter 'a_parameter' must be either positional or named");
  }

  @Test
  public void testNonDefaultParamAfterDefault() throws Exception {
    assertAbout(javaSource())
        .that(getFile("NonDefaultParamAfterDefault.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Positional parameter 'two' has no default value but is specified "
                + "after one or more positional parameters with default values");
  }

  @Test
  public void testPositionalParamAfterNonPositional() throws Exception {
    assertAbout(javaSource())
        .that(getFile("PositionalParamAfterNonPositional.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Positional parameter 'two' is specified after one or more non-positonal parameters");
  }

  @Test
  public void testPositionalOnlyParamAfterNamed() throws Exception {
    assertAbout(javaSource())
        .that(getFile("PositionalOnlyParamAfterNamed.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Positional-only parameter 'two' is specified after one or more named parameters");
  }

  @Test
  public void testExtraKeywordsOutOfOrder() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ExtraKeywordsOutOfOrder.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Expected parameter index 1 to be the "
                + "com.google.devtools.build.lib.syntax.Dict<?,?> type, matching "
                + "extraKeywords, but was java.lang.String");
  }

  @Test
  public void testExtraPositionalsMissing() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ExtraPositionalsMissing.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable annotated method has 3 parameters, but annotation declared "
                + "1 user-supplied parameters and 3 extra interpreter parameters.");
  }

  @Test
  public void testSelfCallWithNoName() throws Exception {
    assertAbout(javaSource())
        .that(getFile("SelfCallWithNoName.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("@SkylarkCallable.name must be non-empty.");
  }

  @Test
  public void testSelfCallWithStructField() throws Exception {
    assertAbout(javaSource())
        .that(getFile("SelfCallWithStructField.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "@SkylarkCallable-annotated methods with selfCall=true must have structField=false");
  }

  @Test
  public void testMultipleSelfCallMethods() throws Exception {
    assertAbout(javaSource())
        .that(getFile("MultipleSelfCallMethods.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Containing class has more than one selfCall method defined.");
  }

  @Test
  public void testEnablingAndDisablingFlag() throws Exception {
    assertAbout(javaSource())
        .that(getFile("EnablingAndDisablingFlag.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Only one of @SkylarkCallable.enablingFlag and @SkylarkCallable.disablingFlag may be "
                + "specified.");
  }

  @Test
  public void testEnablingAndDisablingFlag_param() throws Exception {
    assertAbout(javaSource())
        .that(getFile("EnablingAndDisablingFlagParam.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Parameter 'two' has enableOnlyWithFlag and disableWithFlag set. "
                + "At most one may be set");
  }

  @Test
  public void testConflictingMethodNames() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ConflictingMethodNames.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("Containing class has more than one method with name "
            + "'conflicting_method' defined");
  }

  @Test
  public void testDisabledValueParamNoToggle() throws Exception {
    assertAbout(javaSource())
        .that(getFile("DisabledValueParamNoToggle.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("Parameter 'two' has valueWhenDisabled set, but is always enabled");
  }

  @Test
  public void testToggledKwargsParam() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ToggledKwargsParam.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining("The extraKeywords parameter may not be toggled by semantic flag");
  }

  @Test
  public void testToggledParamNoDisabledValue() throws Exception {
    assertAbout(javaSource())
        .that(getFile("ToggledParamNoDisabledValue.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Parameter 'two' may be disabled by semantic flag, "
                + "thus valueWhenDisabled must be set");
  }

  @Test
  public void testSpecifiedGenericType() throws Exception {
    assertAbout(javaSource())
        .that(getFile("SpecifiedGenericType.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Parameter one has generic type "
                + "com.google.devtools.build.lib.syntax.Sequence<java.lang.String>");
  }

  @Test
  public void testInvalidNoneableParameter() throws Exception {
    assertAbout(javaSource())
        .that(getFile("InvalidNoneableParameter.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "Expected type 'Object' but got type 'java.lang.String' "
                + "for noneable parameter 'aParameter'.");
  }

  @Test
  public void testDoesntImplementStarlarkValue() throws Exception {
    assertAbout(javaSource())
        .that(getFile("DoesntImplementSkylarkValue.java"))
        .processedWith(new SkylarkCallableProcessor())
        .failsToCompile()
        .withErrorContaining(
            "method x has @SkylarkCallable annotation but enclosing class"
                + " DoesntImplementSkylarkValue does not implement StarlarkValue nor has"
                + " @SkylarkGlobalLibrary annotation");
  }
}
