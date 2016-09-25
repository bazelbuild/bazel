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

package com.google.devtools.build.lib.skylark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.PredicateWithMessage;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkClassObjectConstructor;
import com.google.devtools.build.lib.rules.SkylarkAttr;
import com.google.devtools.build.lib.rules.SkylarkFileType;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.RuleFunction;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.Collection;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for SkylarkRuleClassFunctions.
 */
@RunWith(JUnit4.class)
public class SkylarkRuleClassFunctionsTest extends SkylarkTestCase {

  @Before
  public final void createBuildFile() throws Exception  {
    scratch.file(
        "foo/BUILD",
        "genrule(name = 'foo',",
        "  cmd = 'dummy_cmd',",
        "  srcs = ['a.txt', 'b.img'],",
        "  tools = ['t.exe'],",
        "  outs = ['c.txt'])",
        "genrule(name = 'bar',",
        "  cmd = 'dummy_cmd',",
        "  srcs = [':jl', ':gl'],",
        "  outs = ['d.txt'])",
        "java_library(name = 'jl',",
        "  srcs = ['a.java'])",
        "genrule(name = 'gl',",
        "  cmd = 'touch $(OUTS)',",
        "  srcs = ['a.go'],",
        "  outs = [ 'gl.a', 'gl.gcgox', ],",
        "  output_to_bindir = 1,",
        ")");
  }

  @Test
  public void testCannotOverrideBuiltInAttribute() throws Exception {
    ev.setFailFast(true);
    try {
      evalAndExport(
          "def impl(ctx): return", "r = rule(impl, attrs = {'tags': attr.string_list()})");
      Assert.fail("Expected error '"
          + "There is already a built-in attribute 'tags' which cannot be overridden"
          + "' but got no error");
    } catch (IllegalArgumentException | EvalException e) {
      assertThat(e).hasMessage(
          "There is already a built-in attribute 'tags' which cannot be overridden");
    }
  }

  @Test
  public void testImplicitArgsAttribute() throws Exception {
    evalAndExport(
        "def _impl(ctx):",
        "  pass",
        "exec_rule = rule(implementation = _impl, executable = True)",
        "non_exec_rule = rule(implementation = _impl)");
    assertTrue(getRuleClass("exec_rule").hasAttr("args", Type.STRING_LIST));
    assertFalse(getRuleClass("non_exec_rule").hasAttr("args", Type.STRING_LIST));
  }

  private RuleClass getRuleClass(String name) throws Exception {
    return ((RuleFunction) lookup(name)).getRuleClass();
  }

  private void registerDummyUserDefinedFunction() throws Exception {
    eval("def impl():\n" + "  return 0\n");
  }

  private Attribute.Builder<?> evalAttributeDefinition(String... lines) throws Exception {
    return ((SkylarkAttr.Descriptor) evalRuleClassCode(lines)).getAttributeBuilder();
  }

  @Test
  public void testAttrWithOnlyType() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string_list()").build("a1");
    assertEquals(Type.STRING_LIST, attr.getType());
  }

  @Test
  public void testOutputListAttr() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.output_list()").build("a1");
    assertEquals(BuildType.OUTPUT_LIST, attr.getType());
  }

  @Test
  public void testIntListAttr() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.int_list()").build("a1");
    assertEquals(Type.INTEGER_LIST, attr.getType());
  }

  @Test
  public void testOutputAttr() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.output()").build("a1");
    assertEquals(BuildType.OUTPUT, attr.getType());
  }

  @Test
  public void testStringDictAttr() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string_dict(default = {'a': 'b'})").build("a1");
    assertEquals(Type.STRING_DICT, attr.getType());
  }

  @Test
  public void testStringListDictAttr() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string_list_dict(default = {'a': ['b', 'c']})")
        .build("a1");
    assertEquals(Type.STRING_LIST_DICT, attr.getType());
  }

  @Test
  public void testAttrAllowedFileTypesAnyFile() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.label_list(allow_files = True)").build("a1");
    assertEquals(FileTypeSet.ANY_FILE, attr.getAllowedFileTypesPredicate());
  }

  @Test
  public void testAttrAllowedFileTypesWrongType() throws Exception {
    checkErrorContains(
        "allow_files should be a boolean or a string list",
        "attr.label_list(allow_files = 18)");
  }

  @Test
  public void testAttrAllowedSingleFileTypesWrongType() throws Exception {
    checkErrorContains(
        "allow_single_file should be a boolean or a string list",
        "attr.label(allow_single_file = 18)");
  }

  @Test
  public void testAttrWithList() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.label_list(allow_files = ['.xml'])")
        .build("a1");
    assertTrue(attr.getAllowedFileTypesPredicate().apply("a.xml"));
    assertFalse(attr.getAllowedFileTypesPredicate().apply("a.txt"));
    assertFalse(attr.isSingleArtifact());
  }

  @Test
  public void testAttrSingleFileWithList() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.label(allow_single_file = ['.xml'])")
        .build("a1");
    assertTrue(attr.getAllowedFileTypesPredicate().apply("a.xml"));
    assertFalse(attr.getAllowedFileTypesPredicate().apply("a.txt"));
    assertTrue(attr.isSingleArtifact());
  }

  @Test
  public void testAttrWithSkylarkFileType() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.label_list(allow_files = FileType(['.xml']))")
        .build("a1");
    assertTrue(attr.getAllowedFileTypesPredicate().apply("a.xml"));
    assertFalse(attr.getAllowedFileTypesPredicate().apply("a.txt"));
  }

  @Test
  public void testAttrWithProviders() throws Exception {
    Attribute attr =
        evalAttributeDefinition("attr.label_list(allow_files = True, providers = ['a', 'b'])")
            .build("a1");
    assertThat(attr.getMandatoryProvidersList()).containsExactly(ImmutableSet.of("a", "b"));
  }

  @Test
  public void testAttrWithProvidersList() throws Exception {
    Attribute attr =
        evalAttributeDefinition("attr.label_list(allow_files = True,"
            + " providers = [['a', 'b'], ['c']])")
            .build("a1");
    assertThat(attr.getMandatoryProvidersList()).containsExactly(ImmutableSet.of("a", "b"),
        ImmutableSet.of("c"));
  }

  @Test
  public void testAttrWithWrongProvidersList() throws Exception {
    checkErrorContains("Illegal argument: element in 'providers' is of unexpected type."
            + " Should be list of string, but got list with an element of type int.",
        "attr.label_list(allow_files = True,  providers = [['a', 1], ['c']])");

    checkErrorContains("Illegal argument: element in 'providers' is of unexpected type."
            + " Should be list of string, but got string.",
        "attr.label_list(allow_files = True,  providers = [['a', 'b'], 'c'])");
  }

  @Test
  public void testLabelListWithAspects() throws Exception {
    SkylarkAttr.Descriptor attr =
        (SkylarkAttr.Descriptor) evalRuleClassCode(
            "def _impl(target, ctx):",
            "   pass",
            "my_aspect = aspect(implementation = _impl)",
            "attr.label_list(aspects = [my_aspect])");
    Object aspect = ev.lookup("my_aspect");
    assertThat(aspect).isNotNull();
    assertThat(attr.getAspects()).containsExactly(aspect);
  }

  @Test
  public void testLabelListWithAspectsError() throws Exception {
    checkErrorContains(
        "Illegal argument: expected type Aspect for 'aspects' element but got type int instead",
        "def _impl(target, ctx):",
        "   pass",
        "my_aspect = aspect(implementation = _impl)",
        "attr.label_list(aspects = [my_aspect, 123])"
    );
  }

  @Test
  public void testAspectExtraDeps() throws Exception {
    evalAndExport(
        "def _impl(target, ctx):",
        "   pass",
        "my_aspect = aspect(_impl,",
        "   attrs = { '_extra_deps' : attr.label(default = Label('//foo/bar:baz')) }",
        ")");
    SkylarkAspect aspect = (SkylarkAspect) ev.lookup("my_aspect");
    Attribute attribute = Iterables.getOnlyElement(aspect.getAttributes());
    assertThat(attribute.getName()).isEqualTo("$extra_deps");
    assertThat(attribute.getDefaultValue(null))
        .isEqualTo(Label.parseAbsolute("//foo/bar:baz", false));
  }

  @Test
  public void testAspectParameter() throws Exception {
    evalAndExport(
        "def _impl(target, ctx):",
        "   pass",
        "my_aspect = aspect(_impl,",
        "   attrs = { 'param' : attr.string(values=['a', 'b']) }",
        ")");
    SkylarkAspect aspect = (SkylarkAspect) ev.lookup("my_aspect");
    Attribute attribute = Iterables.getOnlyElement(aspect.getAttributes());
    assertThat(attribute.getName()).isEqualTo("param");
  }

  @Test
  public void testAspectParameterRequiresValues() throws Exception {
    checkErrorContains(
        "Aspect parameter attribute 'param' must have type 'string' and use the 'values' "
        + "restriction.",
        "def _impl(target, ctx):",
        "   pass",
        "my_aspect = aspect(_impl,",
        "   attrs = { 'param' : attr.string(default = 'c') }",
        ")");
  }

  @Test
  public void testAspectParameterBadType() throws Exception {
    checkErrorContains(
        "Aspect parameter attribute 'param' must have type 'string' and use the 'values' "
        + "restriction.",
        "def _impl(target, ctx):",
        "   pass",
        "my_aspect = aspect(_impl,",
        "   attrs = { 'param' : attr.label(default = Label('//foo/bar:baz')) }",
        ")");
  }

  @Test
  public void testAspectParameterAndExtraDeps() throws Exception {
    evalAndExport(
        "def _impl(target, ctx):",
        "   pass",
        "my_aspect = aspect(_impl,",
        "   attrs = { 'param' : attr.string(values=['a', 'b']),",
        "             '_extra' : attr.label(default = Label('//foo/bar:baz')) }",
        ")");
    SkylarkAspect aspect = (SkylarkAspect) ev.lookup("my_aspect");
    assertThat(aspect.getAttributes()).hasSize(2);
    assertThat(aspect.getParamAttributes()).containsExactly("param");
  }

  @Test
  public void testAspectNoDefaultValueAttribute() throws Exception {
    checkErrorContains(
        "Aspect attribute '_extra_deps' has no default value",
        "def _impl(target, ctx):",
        "   pass",
        "my_aspect = aspect(_impl,",
        "   attrs = { '_extra_deps' : attr.label() }",
        ")");
  }

  @Test
  public void testNonLabelAttrWithProviders() throws Exception {
    checkErrorContains(
        "unexpected keyword 'providers' in call to string", "attr.string(providers = ['a'])");
  }

  private static final RuleClass.ConfiguredTargetFactory<Object, Object>
      DUMMY_CONFIGURED_TARGET_FACTORY =
      new RuleClass.ConfiguredTargetFactory<Object, Object>() {
        @Override
        public Object create(Object ruleContext) throws InterruptedException {
          throw new IllegalStateException();
        }
      };

  private RuleClass ruleClass(String name) {
    return new RuleClass.Builder(name, RuleClassType.NORMAL, false)
        .factory(DUMMY_CONFIGURED_TARGET_FACTORY)
        .add(Attribute.attr("tags", Type.STRING_LIST))
        .build();
  }
  @Test
  public void testAttrAllowedRuleClassesSpecificRuleClasses() throws Exception {
    Attribute attr = evalAttributeDefinition(
        "attr.label_list(allow_rules = ['java_binary'], allow_files = True)").build("a");
    assertTrue(attr.getAllowedRuleClassesPredicate().apply(ruleClass("java_binary")));
    assertFalse(attr.getAllowedRuleClassesPredicate().apply(ruleClass("genrule")));
  }
  @Test
  public void testAttrDefaultValue() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string(default = 'some value')").build("a1");
    assertEquals("some value", attr.getDefaultValueForTesting());
  }

  @Test
  public void testAttrDefaultValueBadType() throws Exception {
    checkErrorContains(
        "Method attr.string(*, default: string, mandatory: bool, values: sequence of strings) "
            + "is not applicable for arguments (int, bool, list): 'default' is int, "
            + "but should be string",
        "attr.string(default = 1)");
  }

  @Test
  public void testAttrMandatory() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string(mandatory=True)").build("a1");
    assertTrue(attr.isMandatory());
    assertFalse(attr.isNonEmpty());
  }

  @Test
  public void testAttrNonEmpty() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string_list(non_empty=True)").build("a1");
    assertTrue(attr.isNonEmpty());
    assertFalse(attr.isMandatory());
  }

  @Test
  public void testAttrAllowEmpty() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string_list(allow_empty=False)").build("a1");
    assertTrue(attr.isNonEmpty());
    assertFalse(attr.isMandatory());
  }

  @Test
  public void testAttrBadKeywordArguments() throws Exception {
    checkErrorContains(
        "unexpected keyword 'bad_keyword' in call to string", "attr.string(bad_keyword = '')");
  }

  @Test
  public void testAttrCfg() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.label(cfg = 'host', allow_files = True)")
        .build("a1");
    assertEquals(ConfigurationTransition.HOST, attr.getConfigurationTransition());
  }

  @Test
  public void testAttrCfgData() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.label(cfg = 'data', allow_files = True)")
        .build("a1");
    assertEquals(ConfigurationTransition.DATA, attr.getConfigurationTransition());
  }

  @Test
  public void testAttrValues() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.string(values = ['ab', 'cd'])").build("a1");
    PredicateWithMessage<Object> predicate = attr.getAllowedValues();
    assertThat(predicate.apply("ab")).isTrue();
    assertThat(predicate.apply("xy")).isFalse();
  }

  @Test
  public void testAttrIntValues() throws Exception {
    Attribute attr = evalAttributeDefinition("attr.int(values = [1, 2])").build("a1");
    PredicateWithMessage<Object> predicate = attr.getAllowedValues();
    assertThat(predicate.apply(2)).isTrue();
    assertThat(predicate.apply(3)).isFalse();
  }

  @Test
  public void testRuleImplementation() throws Exception {
    evalAndExport("def impl(ctx): return None", "rule1 = rule(impl)");
    RuleClass c = ((RuleFunction) lookup("rule1")).getRuleClass();
    assertEquals("impl", c.getConfiguredTargetFunction().getName());
  }

  @Test
  public void testLateBoundAttrWorksWithOnlyLabel() throws Exception {
    checkEvalError(
        "Method attr.string(*, default: string, mandatory: bool, values: sequence of strings) "
            + "is not applicable for arguments (function, bool, list): 'default' is function, "
            + "but should be string",
        "def attr_value(cfg): return 'a'",
        "attr.string(default=attr_value)");
  }

  private static final Label FAKE_LABEL = Label.parseAbsoluteUnchecked("//fake/label.bzl");

  @Test
  public void testRuleAddAttribute() throws Exception {
    evalAndExport("def impl(ctx): return None", "r1 = rule(impl, attrs={'a1': attr.string()})");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    assertTrue(c.hasAttr("a1", Type.STRING));
  }

  protected void evalAndExport(String... lines) throws Exception {
    eval(lines);
    SkylarkRuleClassFunctions.exportRuleFunctionsAndAspects(ev.getEnvironment(), FAKE_LABEL);
  }

  @Test
  public void testOutputToGenfiles() throws Exception {
    evalAndExport("def impl(ctx): pass", "r1 = rule(impl, output_to_genfiles=True)");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    assertFalse(c.hasBinaryOutput());
  }

  @Test
  public void testRuleAddMultipleAttributes() throws Exception {
    evalAndExport(
        "def impl(ctx): return None",
        "r1 = rule(impl,",
        "     attrs = {",
        "            'a1': attr.label_list(allow_files=True),",
        "            'a2': attr.int()",
        "})");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    assertTrue(c.hasAttr("a1", BuildType.LABEL_LIST));
    assertTrue(c.hasAttr("a2", Type.INTEGER));
  }
  @Test
  public void testRuleAttributeFlag() throws Exception {
    evalAndExport(
        "def impl(ctx): return None",
        "r1 = rule(impl, attrs = {'a1': attr.string(mandatory=True)})");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    assertTrue(c.getAttributeByName("a1").isMandatory());
  }

  @Test
  public void testRuleOutputs() throws Exception {
    evalAndExport(
        "def impl(ctx): return None",
        "r1 = rule(impl, outputs = {'a': 'a.txt'})");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    ImplicitOutputsFunction function = c.getDefaultImplicitOutputsFunction();
    assertEquals("a.txt", Iterables.getOnlyElement(function.getImplicitOutputs(null)));
  }

  @Test
  public void testRuleUnknownKeyword() throws Exception {
    registerDummyUserDefinedFunction();
    checkErrorContains(
        "unexpected keyword 'bad_keyword' in call to " + "rule(implementation: function, ",
        "rule(impl, bad_keyword = 'some text')");
  }

  @Test
  public void testRuleImplementationMissing() throws Exception {
    checkErrorContains(
        "missing mandatory positional argument 'implementation' while calling "
            + "rule(implementation",
        "rule(attrs = {})");
  }

  @Test
  public void testRuleBadTypeForAdd() throws Exception {
    registerDummyUserDefinedFunction();
    checkErrorContains(
        "expected dict or NoneType for 'attrs' while calling rule but got string instead: "
            + "some text",
        "rule(impl, attrs = 'some text')");
  }

  @Test
  public void testRuleBadTypeInAdd() throws Exception {
    registerDummyUserDefinedFunction();
    checkErrorContains(
        "Illegal argument: "
            + "expected <String, Descriptor> type for 'attrs' but got <string, string> instead",
        "rule(impl, attrs = {'a1': 'some text'})");
  }

  @Test
  public void testLabel() throws Exception {
    Object result = evalRuleClassCode("Label('//foo/foo:foo')");
    assertThat(result).isInstanceOf(Label.class);
    assertEquals("//foo/foo:foo", result.toString());
  }

  @Test
  public void testLabelSameInstance() throws Exception {
    Object l1 = evalRuleClassCode("Label('//foo/foo:foo')");
    // Implicitly creates a new pkgContext and environment, yet labels should be the same.
    Object l2 = evalRuleClassCode("Label('//foo/foo:foo')");
    assertSame(l2, l1);
  }

  @Test
  public void testLabelNameAndPackage() throws Exception {
    Object result = evalRuleClassCode("Label('//foo/bar:baz').name");
    assertEquals("baz", result);
    // NB: implicitly creates a new pkgContext and environments, yet labels should be the same.
    result = evalRuleClassCode("Label('//foo/bar:baz').package");
    assertEquals("foo/bar", result);
  }

  @Test
  public void testRuleLabelDefaultValue() throws Exception {
    evalAndExport(
        "def impl(ctx): return None\n"
            + "r1 = rule(impl, attrs = {'a1': "
            + "attr.label(default = Label('//foo:foo'), allow_files=True)})");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    Attribute a = c.getAttributeByName("a1");
    assertThat(a.getDefaultValueForTesting()).isInstanceOf(Label.class);
    assertEquals("//foo:foo", a.getDefaultValueForTesting().toString());
  }

  @Test
  public void testIntDefaultValue() throws Exception {
    evalAndExport(
        "def impl(ctx): return None",
        "r1 = rule(impl, attrs = {'a1': attr.int(default = 40+2)})");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    Attribute a = c.getAttributeByName("a1");
    assertEquals(42, a.getDefaultValueForTesting());
  }

  @Test
  public void testFileType() throws Exception {
    Object result = evalRuleClassCode("FileType(['.css'])");
    SkylarkFileType fts = (SkylarkFileType) result;
    assertEquals(ImmutableList.of(".css"), fts.getExtensions());
  }

  @Test
  public void testRuleInheritsBaseRuleAttributes() throws Exception {
    evalAndExport("def impl(ctx): return None", "r1 = rule(impl)");
    RuleClass c = ((RuleFunction) lookup("r1")).getRuleClass();
    assertTrue(c.hasAttr("tags", Type.STRING_LIST));
    assertTrue(c.hasAttr("visibility", BuildType.NODEP_LABEL_LIST));
    assertTrue(c.hasAttr("deprecation", Type.STRING));
    assertTrue(c.hasAttr(":action_listener", BuildType.LABEL_LIST)); // required for extra actions
  }

  private void checkTextMessage(String from, String... lines) throws Exception {
    Object result = evalRuleClassCode(from);
    assertEquals(Joiner.on("\n").join(lines) + "\n", result);
  }

  @Test
  public void testSimpleTextMessagesBooleanFields() throws Exception {
    checkTextMessage("struct(name=True).to_proto()", "name: true");
    checkTextMessage("struct(name=False).to_proto()", "name: false");
  }

  @Test
  public void testSimpleTextMessages() throws Exception {
    checkTextMessage("struct(name='value').to_proto()", "name: \"value\"");
    checkTextMessage("struct(name=['a', 'b']).to_proto()", "name: \"a\"", "name: \"b\"");
    checkTextMessage("struct(name=123).to_proto()", "name: 123");
    checkTextMessage("struct(name=[1, 2, 3]).to_proto()", "name: 1", "name: 2", "name: 3");
    checkTextMessage("struct(a=struct(b='b')).to_proto()", "a {", "  b: \"b\"", "}");
    checkTextMessage(
        "struct(a=[struct(b='x'), struct(b='y')]).to_proto()",
        "a {",
        "  b: \"x\"",
        "}",
        "a {",
        "  b: \"y\"",
        "}");
    checkTextMessage(
        "struct(a=struct(b=struct(c='c'))).to_proto()", "a {", "  b {", "    c: \"c\"", "  }", "}");
  }

  @Test
  public void testTextMessageEscapes() throws Exception {
    checkTextMessage("struct(name='a\"b').to_proto()", "name: \"a\\\"b\"");
    checkTextMessage("struct(name='a\\'b').to_proto()", "name: \"a'b\"");
    checkTextMessage("struct(name='a\\nb').to_proto()", "name: \"a\\nb\"");

    // struct(name="a\\\"b") -> name: "a\\\"b"
    checkTextMessage("struct(name='a\\\\\\\"b').to_proto()", "name: \"a\\\\\\\"b\"");
  }

  @Test
  public void testTextMessageInvalidElementInListStructure() throws Exception {
    checkErrorContains(
        "Invalid text format, expected a struct, a string, a bool, or "
            + "an int but got a list for list element in struct field 'a'",
        "struct(a=[['b']]).to_proto()");
  }

  @Test
  public void testTextMessageInvalidStructure() throws Exception {
    checkErrorContains(
        "Invalid text format, expected a struct, a string, a bool, or an int "
            + "but got a ConfigurationTransition for struct field 'a'",
        "struct(a=DATA_CFG).to_proto()");
  }

  private void checkJson(String from, String expected) throws Exception {
    Object result = evalRuleClassCode(from);
    assertEquals(expected, result);
  }

  @Test
  public void testJsonBooleanFields() throws Exception {
    checkJson("struct(name=True).to_json()", "{\"name\":true}");
    checkJson("struct(name=False).to_json()", "{\"name\":false}");
  }

  @Test
  public void testJsonEncoding() throws Exception {
    checkJson("struct(name='value').to_json()", "{\"name\":\"value\"}");
    checkJson("struct(name=['a', 'b']).to_json()", "{\"name\":[\"a\",\"b\"]}");
    checkJson("struct(name=123).to_json()", "{\"name\":123}");
    checkJson("struct(name=[1, 2, 3]).to_json()", "{\"name\":[1,2,3]}");
    checkJson("struct(a=struct(b='b')).to_json()", "{\"a\":{\"b\":\"b\"}}");
    checkJson("struct(a=[struct(b='x'), struct(b='y')]).to_json()",
        "{\"a\":[{\"b\":\"x\"},{\"b\":\"y\"}]}");
    checkJson("struct(a=struct(b=struct(c='c'))).to_json()",
        "{\"a\":{\"b\":{\"c\":\"c\"}}}");
  }

  @Test
  public void testJsonEscapes() throws Exception {
    checkJson("struct(name='a\"b').to_json()", "{\"name\":\"a\\\"b\"}");
    checkJson("struct(name='a\\'b').to_json()", "{\"name\":\"a'b\"}");
    checkJson("struct(name='a\\\\b').to_json()", "{\"name\":\"a\\\\b\"}");
    checkJson("struct(name='a\\nb').to_json()", "{\"name\":\"a\\nb\"}");
    checkJson("struct(name='a\\rb').to_json()", "{\"name\":\"a\\rb\"}");
    checkJson("struct(name='a\\tb').to_json()", "{\"name\":\"a\\tb\"}");
  }

  @Test
  public void testJsonNestedListStructure() throws Exception {
    checkJson("struct(a=[['b']]).to_json()", "{\"a\":[[\"b\"]]}");
  }

  @Test
  public void testJsonInvalidStructure() throws Exception {
    checkErrorContains(
        "Invalid text format, expected a struct, a string, a bool, or an int but got a "
            + "ConfigurationTransition for struct field 'a'",
        "struct(a=DATA_CFG).to_json()");
  }

  @Test
  public void testLabelAttrWrongDefault() throws Exception {
    checkErrorContains(
        "expected Label or Label-returning function or NoneType for 'default' "
            + "while calling label but got string instead: //foo:bar",
        "attr.label(default = '//foo:bar')");
  }

  @Test
  public void testLabelGetRelative() throws Exception {
    assertEquals("//foo:baz", eval("Label('//foo:bar').relative('baz')").toString());
    assertEquals("//baz:qux", eval("Label('//foo:bar').relative('//baz:qux')").toString());
  }

  @Test
  public void testLabelGetRelativeSyntaxError() throws Exception {
    checkErrorContains(
        "invalid target name 'bad syntax': target names may not contain ' '",
        "Label('//foo:bar').relative('bad syntax')");
  }

  @Test
  public void testLicenseAttributesNonconfigurable() throws Exception {
    scratch.file("test/BUILD");
    scratch.file("test/rule.bzl",
        "def _impl(ctx):",
        "  return",
        "some_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'licenses': attr.license()",
        "  }",
        ")");
    scratch.file("third_party/foo/BUILD",
        "load('/test/rule', 'some_rule')",
        "some_rule(",
        "    name='r',",
        "    licenses = ['unencumbered']",
        ")");
    invalidatePackages();
    // Should succeed without a "licenses attribute is potentially configurable" loading error:
    createRuleContext("//third_party/foo:r");
  }

  @Test
  public void testStructCreation() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    eval("x = struct(a = 1, b = 2)");
    assertThat(lookup("x")).isInstanceOf(ClassObject.class);
  }

  @Test
  public void testStructFields() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    eval("x = struct(a = 1, b = 2)");
    ClassObject x = (ClassObject) lookup("x");
    assertEquals(1, x.getValue("a"));
    assertEquals(2, x.getValue("b"));
  }

  @Test
  public void testStructAccessingFieldsFromSkylark() throws Exception {
    eval("x = struct(a = 1, b = 2)", "x1 = x.a", "x2 = x.b");
    assertThat(lookup("x1")).isEqualTo(1);
    assertThat(lookup("x2")).isEqualTo(2);
  }

  @Test
  public void testStructAccessingUnknownField() throws Exception {
    checkErrorContains(
            "'struct' object has no attribute 'c'\n" + "Available attributes: a, b",
            "x = struct(a = 1, b = 2)",
            "y = x.c");
  }

  @Test
  public void testStructAccessingUnknownFieldWithArgs() throws Exception {
    checkErrorContains(
        "struct has no method 'c'", "x = struct(a = 1, b = 2)", "y = x.c()");
  }

  @Test
  public void testStructAccessingNonFunctionFieldWithArgs() throws Exception {
    checkErrorContains(
        "struct field 'a' is not a function", "x = struct(a = 1, b = 2)", "x1 = x.a(1)");
  }

  @Test
  public void testStructAccessingFunctionFieldWithArgs() throws Exception {
    eval("def f(x): return x+5", "x = struct(a = f, b = 2)", "x1 = x.a(1)");
    assertThat(lookup("x1")).isEqualTo(6);
  }

  @Test
  public void testStructPosArgs() throws Exception {
    checkErrorContains(
        "struct(**kwargs) does not accept positional arguments, but got 1", "x = struct(1, b = 2)");
  }

  @Test
  public void testStructConcatenationFieldNames() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    eval("x = struct(a = 1, b = 2)",
        "y = struct(c = 1, d = 2)",
        "z = x + y\n");
    SkylarkClassObject z = (SkylarkClassObject) lookup("z");
    assertEquals(ImmutableSet.of("a", "b", "c", "d"), z.getKeys());
  }

  @Test
  public void testStructConcatenationFieldValues() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    eval("x = struct(a = 1, b = 2)",
        "y = struct(c = 1, d = 2)",
        "z = x + y\n");
    SkylarkClassObject z = (SkylarkClassObject) lookup("z");
    assertEquals(1, z.getValue("a"));
    assertEquals(2, z.getValue("b"));
    assertEquals(1, z.getValue("c"));
    assertEquals(2, z.getValue("d"));
  }

  @Test
  public void testStructConcatenationCommonFields() throws Exception {
    checkErrorContains("Cannot concat structs with common field(s): a",
        "x = struct(a = 1, b = 2)", "y = struct(c = 1, a = 2)", "z = x + y\n");
  }

  @Test
  public void testConditionalStructConcatenation() throws Exception {
    // TODO(fwe): cannot be handled by current testing suite
    eval("def func():",
        "  x = struct(a = 1, b = 2)",
        "  if True:",
        "    x += struct(c = 1, d = 2)",
        "  return x",
        "x = func()");
    SkylarkClassObject x = (SkylarkClassObject) lookup("x");
    assertEquals(1, x.getValue("a"));
    assertEquals(2, x.getValue("b"));
    assertEquals(1, x.getValue("c"));
    assertEquals(2, x.getValue("d"));
  }

  @Test
  public void testGetattrNoAttr() throws Exception {
    checkErrorContains("Object of type 'struct' has no attribute \"b\"",
        "s = struct(a='val')", "getattr(s, 'b')");
  }

  @Test
  public void testGetattr() throws Exception {
    eval(
        "s = struct(a='val')",
        "x = getattr(s, 'a')",
        "y = getattr(s, 'b', 'def')",
        "z = getattr(s, 'b', default = 'def')",
        "w = getattr(s, 'a', default='ignored')");
    assertThat(lookup("x")).isEqualTo("val");
    assertThat(lookup("y")).isEqualTo("def");
    assertThat(lookup("z")).isEqualTo("def");
    assertThat(lookup("w")).isEqualTo("val");
  }

  @Test
  public void testHasattr() throws Exception {
    eval("s = struct(a=1)",
        "x = hasattr(s, 'a')",
        "y = hasattr(s, 'b')\n");
    assertThat(lookup("x")).isEqualTo(true);
    assertThat(lookup("y")).isEqualTo(false);
  }

  @Test
  public void testStructStr() throws Exception {
    assertThat(eval("str(struct(x = 2, y = 3, z = 4))"))
        .isEqualTo("struct(x = 2, y = 3, z = 4)");
  }

  @Test
  public void testStructsInSets() throws Exception {
    eval("set([struct(a='a')])");
  }

  @Test
  public void testStructMembersAreImmutable() throws Exception {
    checkErrorContains(
        "can only assign to variables and tuples, not to 's.x'",
        "s = struct(x = 'a')",
        "s.x = 'b'\n");
  }

  @Test
  public void testStructDictMembersAreMutable() throws Exception {
    eval(
        "s = struct(x = {'a' : 1})",
        "s.x['b'] = 2\n");
    assertThat(((SkylarkClassObject) lookup("s")).getValue("x"))
        .isEqualTo(ImmutableMap.of("a", 1, "b", 2));
  }

  @Test
  public void testNsetGoodCompositeItem() throws Exception {
    eval("def func():",
        "  return set([struct(a='a')])",
        "s = func()");
    Collection<Object> result = ((SkylarkNestedSet) lookup("s")).toCollection();
    assertThat(result).hasSize(1);
    assertThat(result.iterator().next()).isInstanceOf(SkylarkClassObject.class);
  }

  @Test
  public void testNsetBadMutableItem() throws Exception {
    checkEvalError("sets cannot contain mutable items", "set([([],)])");
    checkEvalError("sets cannot contain mutable items", "set([struct(a=[])])");
  }

  private static SkylarkClassObject makeStruct(String field, Object value) {
    return SkylarkClassObjectConstructor.STRUCT.create(
        ImmutableMap.of(field, value),
        "no field '%'");
  }

  private static SkylarkClassObject makeBigStruct(Environment env) {
    // struct(a=[struct(x={1:1}), ()], b=(), c={2:2})
    return SkylarkClassObjectConstructor.STRUCT.create(
        ImmutableMap.<String, Object>of(
            "a", MutableList.<Object>of(env,
                SkylarkClassObjectConstructor.STRUCT.create(ImmutableMap.<String, Object>of(
                    "x", SkylarkDict.<Object, Object>of(env, 1, 1)),
                    "no field '%s'"),
                Tuple.of()),
            "b", Tuple.of(),
            "c", SkylarkDict.<Object, Object>of(env, 2, 2)),
        "no field '%s'");
  }

  @Test
  public void testStructMutabilityShallow() throws Exception {
    assertTrue(EvalUtils.isImmutable(makeStruct("a", 1)));
  }

  private static MutableList<Object> makeList(Environment env) {
    return MutableList.<Object>of(env, 1, 2, 3);
  }

  @Test
  public void testStructMutabilityDeep() throws Exception {
    assertTrue(EvalUtils.isImmutable(Tuple.<Object>of(makeList(null))));
    assertTrue(EvalUtils.isImmutable(makeStruct("a", makeList(null))));
    assertTrue(EvalUtils.isImmutable(makeBigStruct(null)));

    assertFalse(EvalUtils.isImmutable(Tuple.<Object>of(makeList(ev.getEnvironment()))));
    assertFalse(EvalUtils.isImmutable(makeStruct("a", makeList(ev.getEnvironment()))));
    assertFalse(EvalUtils.isImmutable(makeBigStruct(ev.getEnvironment())));
  }

  @Test
  public void declaredProviders() throws Exception {
    evalAndExport(
        "data = provider()",
        "d = data(x = 1, y ='abc')",
        "d_x = d.x",
        "d_y = d.y"
    );
    assertThat(lookup("d_x")).isEqualTo(1);
    assertThat(lookup("d_y")).isEqualTo("abc");
    SkylarkClassObjectConstructor dataConstructor = (SkylarkClassObjectConstructor) lookup("data");
    SkylarkClassObject data = (SkylarkClassObject) lookup("d");
    assertThat(data.getConstructor()).isEqualTo(dataConstructor);
    assertThat(dataConstructor.isExported()).isTrue();
    assertThat(dataConstructor.getPrintableName()).isEqualTo("data");
    assertThat(dataConstructor.getKey()).isEqualTo(
        new SkylarkClassObjectConstructor.SkylarkKey(FAKE_LABEL, "data")
    );
  }

  @Test
  public void declaredProvidersConcatSuccess() throws Exception {
    evalAndExport(
        "data = provider()",
        "dx = data(x = 1)",
        "dy = data(y = 'abc')",
        "dxy = dx + dy",
        "x = dxy.x",
        "y = dxy.y"
    );
    assertThat(lookup("x")).isEqualTo(1);
    assertThat(lookup("y")).isEqualTo("abc");
    SkylarkClassObjectConstructor dataConstructor = (SkylarkClassObjectConstructor) lookup("data");
    SkylarkClassObject dx = (SkylarkClassObject) lookup("dx");
    assertThat(dx.getConstructor()).isEqualTo(dataConstructor);
    SkylarkClassObject dy = (SkylarkClassObject) lookup("dy");
    assertThat(dy.getConstructor()).isEqualTo(dataConstructor);
  }

  @Test
  public void declaredProvidersConcatError() throws Exception {
    evalAndExport(
        "data1 = provider()",
        "data2 = provider()"
    );

    checkEvalError("Cannot concat data1 with data2",
        "d1 = data1(x = 1)",
        "d2 = data2(y = 2)",
        "d = d1 + d2"
    );
  }

  @Test
  public void structsAsDeclaredProvidersTest() throws Exception {
    evalAndExport(
        "data = struct(x = 1)"
    );
    SkylarkClassObject data = (SkylarkClassObject) lookup("data");
    assertThat(SkylarkClassObjectConstructor.STRUCT.isExported()).isTrue();
    assertThat(data.getConstructor()).isEqualTo(SkylarkClassObjectConstructor.STRUCT);
    assertThat(data.getConstructor().getKey())
        .isEqualTo(SkylarkClassObjectConstructor.STRUCT.getKey());
  }
}
