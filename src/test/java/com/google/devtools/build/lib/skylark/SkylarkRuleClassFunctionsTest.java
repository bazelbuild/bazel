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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
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
import com.google.devtools.build.lib.rules.SkylarkFileType;
import com.google.devtools.build.lib.rules.SkylarkRuleClassFunctions.RuleFunction;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

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
  public void setUp() throws Exception {
    super.setUp();
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
    checkEvalError(
        "There is already a built-in attribute 'tags' which cannot be overridden",
        "def impl(ctx): return",
        "r = rule(impl, attrs = {'tags': attr.string_list()})");
  }

  @Test
  public void testImplicitArgsAttribute() throws Exception {
    eval(
        "def _impl(ctx):",
        "  pass",
        "exec_rule = rule(implementation = _impl, executable = True)",
        "non_exec_rule = rule(implementation = _impl)");
    assertTrue(getRuleClass("exec_rule").hasAttr("args", Type.STRING_LIST));
    assertFalse(getRuleClass("non_exec_rule").hasAttr("args", Type.STRING_LIST));
  }

  private RuleClass getRuleClass(String name) throws Exception {
    return ((RuleFunction) lookup(name)).getBuilder().build(name);
  }

  private void registerDummyUserDefinedFunction() throws Exception {
    eval("def impl():\n" + "  return 0\n");
  }

  @Test
  public void testAttrWithOnlyType() throws Exception {
    Object result = evalRuleClassCode("attr.string_list()");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(Type.STRING_LIST, attr.getType());
  }

  @Test
  public void testOutputListAttr() throws Exception {
    Object result = evalRuleClassCode("attr.output_list()");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(BuildType.OUTPUT_LIST, attr.getType());
  }

  @Test
  public void testIntListAttr() throws Exception {
    Object result = evalRuleClassCode("attr.int_list()");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(Type.INTEGER_LIST, attr.getType());
  }

  @Test
  public void testOutputAttr() throws Exception {
    Object result = evalRuleClassCode("attr.output()");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(BuildType.OUTPUT, attr.getType());
  }

  @Test
  public void testStringDictAttr() throws Exception {
    Object result = evalRuleClassCode("attr.string_dict(default = {'a': 'b'})");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(Type.STRING_DICT, attr.getType());
  }

  @Test
  public void testStringListDictAttr() throws Exception {
    Object result = evalRuleClassCode("attr.string_list_dict(default = {'a': ['b', 'c']})");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(Type.STRING_LIST_DICT, attr.getType());
  }

  @Test
  public void testAttrAllowedFileTypesAnyFile() throws Exception {
    Object result = evalRuleClassCode("attr.label_list(allow_files = True)");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(FileTypeSet.ANY_FILE, attr.getAllowedFileTypesPredicate());
  }

  @Test
  public void testAttrAllowedFileTypesWrongType() throws Exception {
    checkErrorContains(
        "allow_files should be a boolean or a filetype object.",
        "attr.label_list(allow_files = ['.xml'])");
  }

  @Test
  public void testAttrWithSkylarkFileType() throws Exception {
    Object result = evalRuleClassCode("attr.label_list(allow_files = FileType(['.xml']))");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertTrue(attr.getAllowedFileTypesPredicate().apply("a.xml"));
    assertFalse(attr.getAllowedFileTypesPredicate().apply("a.txt"));
  }

  @Test
  public void testAttrWithProviders() throws Exception {
    Object result =
        evalRuleClassCode("attr.label_list(allow_files = True, providers = ['a', 'b'])");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(ImmutableSet.of("a", "b"), attr.getMandatoryProviders());
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
    Object result =
        evalRuleClassCode("attr.label_list(allow_rules = ['java_binary'], allow_files = True)");
    Attribute attr = ((Attribute.Builder<?>) result).build("a");
    assertTrue(attr.getAllowedRuleClassesPredicate().apply(ruleClass("java_binary")));
    assertFalse(attr.getAllowedRuleClassesPredicate().apply(ruleClass("genrule")));
  }
  @Test
  public void testAttrDefaultValue() throws Exception {
    Object result = evalRuleClassCode("attr.string(default = 'some value')");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
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
    Object result = evalRuleClassCode("attr.string(mandatory=True)");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertTrue(attr.isMandatory());
    assertFalse(attr.isNonEmpty());
  }

  @Test
  public void testAttrNonEmpty() throws Exception {
    Object result = evalRuleClassCode("attr.string_list(non_empty=True)");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
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
    Object result = evalRuleClassCode("attr.label(cfg = HOST_CFG, allow_files = True)");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    assertEquals(ConfigurationTransition.HOST, attr.getConfigurationTransition());
  }

  @Test
  public void testAttrValues() throws Exception {
    Object result = evalRuleClassCode("attr.string(values = ['ab', 'cd'])");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    PredicateWithMessage<Object> predicate = attr.getAllowedValues();
    assertThat(predicate.apply("ab")).isTrue();
    assertThat(predicate.apply("xy")).isFalse();
  }

  @Test
  public void testAttrIntValues() throws Exception {
    Object result = evalRuleClassCode("attr.int(values = [1, 2])");
    Attribute attr = ((Attribute.Builder<?>) result).build("a1");
    PredicateWithMessage<Object> predicate = attr.getAllowedValues();
    assertThat(predicate.apply(2)).isTrue();
    assertThat(predicate.apply(3)).isFalse();
  }

  @Test
  public void testRuleImplementation() throws Exception {
    eval("def impl(ctx): return None", "rule1 = rule(impl)");
    RuleClass c = ((RuleFunction) lookup("rule1")).getBuilder().build("rule1");
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

  @Test
  public void testRuleAddAttribute() throws Exception {
    eval("def impl(ctx): return None", "r1 = rule(impl, attrs={'a1': attr.string()})");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
    assertTrue(c.hasAttr("a1", Type.STRING));
  }

  @Test
  public void testOutputToGenfiles() throws Exception {
    eval("def impl(ctx): pass", "r1 = rule(impl, output_to_genfiles=True)");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
    assertFalse(c.hasBinaryOutput());
  }

  @Test
  public void testRuleAddMultipleAttributes() throws Exception {
    eval(
        "def impl(ctx): return None",
        "r1 = rule(impl,",
        "     attrs = {",
        "            'a1': attr.label_list(allow_files=True),",
        "            'a2': attr.int()",
        "})");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
    assertTrue(c.hasAttr("a1", BuildType.LABEL_LIST));
    assertTrue(c.hasAttr("a2", Type.INTEGER));
  }
  @Test
  public void testRuleAttributeFlag() throws Exception {
    eval(
        "def impl(ctx): return None",
        "r1 = rule(impl, attrs = {'a1': attr.string(mandatory=True)})");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
    assertTrue(c.getAttributeByName("a1").isMandatory());
  }

  @Test
  public void testRuleOutputs() throws Exception {
    eval("def impl(ctx): return None", "r1 = rule(impl, outputs = {'a': 'a.txt'})");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
    ImplicitOutputsFunction function = c.getImplicitOutputsFunction();
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
            + "expected <String, Builder> type for 'attrs' but got <string, string> instead",
        "rule(impl, attrs = {'a1': 'some text'})");
  }

  @Test
  public void testLabel() throws Exception {
    Object result = evalRuleClassCode("Label('//foo/foo:foo')");
    assertEquals("//foo/foo:foo", ((Label) result).toString());
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
    eval(
        "def impl(ctx): return None\n"
            + "r1 = rule(impl, attrs = {'a1': "
            + "attr.label(default = Label('//foo:foo'), allow_files=True)})");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
    Attribute a = c.getAttributeByName("a1");
    assertEquals("//foo:foo", ((Label) a.getDefaultValueForTesting()).toString());
  }

  @Test
  public void testIntDefaultValue() throws Exception {
    eval("def impl(ctx): return None", "r1 = rule(impl, attrs = {'a1': attr.int(default = 40+2)})");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
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
    eval("def impl(ctx): return None", "r1 = rule(impl)");
    RuleClass c = ((RuleFunction) lookup("r1")).getBuilder().build("r1");
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
}
