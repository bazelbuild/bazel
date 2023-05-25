// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.starlarkdocextract;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.AttributeType;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionDeprecationInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionParamInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionReturnInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderFieldInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.ProviderNameGroup;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import java.util.function.Predicate;
import net.starlark.java.eval.Module;
import net.starlark.java.syntax.FileOptions;
import net.starlark.java.syntax.ParserInput;
import net.starlark.java.syntax.Program;
import net.starlark.java.syntax.StarlarkFile;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ModuleInfoExtractorTest {

  private static final Label FAKE_LABEL = Label.parseCanonicalUnchecked("//test:test.bzl");

  private Module exec(String... lines) throws Exception {
    BazelEvaluationTestCase ev = new BazelEvaluationTestCase();
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, FileOptions.DEFAULT);
    Program program = Program.compileFile(file, ev.getModule());
    BzlLoadFunction.execAndExport(
        program, FAKE_LABEL, ev.getEventHandler(), ev.getModule(), ev.getStarlarkThread());
    return ev.getModule();
  }

  private static ModuleInfoExtractor getExtractor() {
    return new ModuleInfoExtractor(name -> true, RepositoryMapping.ALWAYS_FALLBACK);
  }

  private static ModuleInfoExtractor getExtractor(Predicate<String> isWantedName) {
    return new ModuleInfoExtractor(isWantedName, RepositoryMapping.ALWAYS_FALLBACK);
  }

  private static ModuleInfoExtractor getExtractor(RepositoryMapping repositoryMapping) {
    return new ModuleInfoExtractor(name -> true, repositoryMapping);
  }

  @Test
  public void moduleDocstring() throws Exception {
    Module moduleWithDocstring = exec("'''This is my docstring'''", "foo = 1");
    assertThat(getExtractor().extractFrom(moduleWithDocstring).getModuleDocstring())
        .isEqualTo("This is my docstring");

    Module moduleWithoutDocstring = exec("foo = 1");
    assertThat(getExtractor().extractFrom(moduleWithoutDocstring).getModuleDocstring()).isEmpty();
  }

  @Test
  public void extractOnlyWantedExportableNames() throws Exception {
    Module module =
        exec(
            "def exported_unwanted():",
            "    pass",
            "def exported_wanted():",
            "    pass",
            "def _nonexported():",
            "    pass",
            "def _nonexported_matches_wanted_predicate():",
            "    pass");

    ModuleInfo moduleInfo = getExtractor(name -> name.contains("_wanted")).extractFrom(module);
    assertThat(moduleInfo.getFuncInfoList().stream().map(StarlarkFunctionInfo::getFunctionName))
        .containsExactly("exported_wanted");
  }

  @Test
  public void namespaces() throws Exception {
    Module module =
        exec(
            "def _my_func(**kwargs):",
            "    pass",
            "_my_binary = rule(implementation = _my_func)",
            "_my_aspect = aspect(implementation = _my_func)",
            "_MyInfo = provider()",
            "name = struct(",
            "    spaced = struct(",
            "        my_func = _my_func,",
            "        my_binary = _my_binary,",
            "        my_aspect = _my_aspect,",
            "        MyInfo = _MyInfo,",
            "    ),",
            ")");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getFuncInfoList().stream().map(StarlarkFunctionInfo::getFunctionName))
        .containsExactly("name.spaced.my_func");
    assertThat(moduleInfo.getRuleInfoList().stream().map(RuleInfo::getRuleName))
        .containsExactly("name.spaced.my_binary");
    assertThat(moduleInfo.getAspectInfoList().stream().map(AspectInfo::getAspectName))
        .containsExactly("name.spaced.my_aspect");
    assertThat(moduleInfo.getProviderInfoList().stream().map(ProviderInfo::getProviderName))
        .containsExactly("name.spaced.MyInfo");
  }

  @Test
  public void functionDocstring() throws Exception {
    Module module =
        exec(
            "def with_detailed_docstring():",
            "    '''My function",
            "    ",
            "    This function does things.",
            "    '''",
            "    pass",
            "def with_one_line_docstring():",
            "    '''My function'''",
            "    pass",
            "def without_docstring():",
            "    pass");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getFuncInfoList())
        .containsExactly(
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("with_detailed_docstring")
                .setDocString("My function\n\nThis function does things.")
                .build(),
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("with_one_line_docstring")
                .setDocString("My function")
                .build(),
            StarlarkFunctionInfo.newBuilder().setFunctionName("without_docstring").build());
  }

  @Test
  public void functionParams() throws Exception {
    Module module =
        exec(
            "def my_func(documented, undocumented, has_default = {'foo': 'bar'}, *args, **kwargs):",
            "    '''My function",
            "    ",
            "    Args:",
            "      documented: Documented param",
            "    '''",
            "    pass");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getFuncInfoList().get(0).getParameterList())
        .containsExactly(
            FunctionParamInfo.newBuilder()
                .setName("documented")
                .setDocString("Documented param")
                .setMandatory(true)
                .build(),
            FunctionParamInfo.newBuilder().setName("undocumented").setMandatory(true).build(),
            FunctionParamInfo.newBuilder()
                .setName("has_default")
                .setDefaultValue("{\"foo\": \"bar\"}")
                .build(),
            FunctionParamInfo.newBuilder().setName("args").build(),
            FunctionParamInfo.newBuilder().setName("kwargs").build());
  }

  @Test
  public void functionReturn() throws Exception {
    Module module =
        exec(
            "def with_return():",
            "    '''My doc",
            "    ",
            "    Returns:",
            "      None",
            "    '''",
            "    return None",
            "def without_return():",
            "    '''My doc'''",
            "    pass");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getFuncInfoList())
        .containsExactly(
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("with_return")
                .setDocString("My doc")
                .setReturn(FunctionReturnInfo.newBuilder().setDocString("None").build())
                .build(),
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("without_return")
                .setDocString("My doc")
                .build());
  }

  @Test
  public void functionDeprecated() throws Exception {
    Module module =
        exec(
            "def with_deprecated():",
            "    '''My doc",
            "    ",
            "    Deprecated:",
            "      This is deprecated",
            "    '''",
            "    pass",
            "def without_deprecated():",
            "    '''My doc'''",
            "    pass");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getFuncInfoList())
        .containsExactly(
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("with_deprecated")
                .setDocString("My doc")
                .setDeprecated(
                    FunctionDeprecationInfo.newBuilder().setDocString("This is deprecated").build())
                .build(),
            StarlarkFunctionInfo.newBuilder()
                .setFunctionName("without_deprecated")
                .setDocString("My doc")
                .build());
  }

  @Test
  public void providerDocstring() throws Exception {
    Module module =
        exec(
            "DocumentedInfo = provider(doc = 'My doc')", //
            "UndocumentedInfo = provider()");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getProviderInfoList())
        .containsExactly(
            ProviderInfo.newBuilder()
                .setProviderName("DocumentedInfo")
                .setDocString("My doc")
                .build(),
            ProviderInfo.newBuilder().setProviderName("UndocumentedInfo").build());
  }

  @Test
  public void providerFields() throws Exception {
    Module module =
        exec(
            // Note fields below are not alphabetized
            "DocumentedInfo = provider(fields = {'c': 'C', 'a': 'A', 'b': 'B', '_hidden':"
                + " 'Hidden'})",
            "UndocumentedInfo = provider(fields = ['c', 'a', 'b', '_hidden'])");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getProviderInfoList())
        .containsExactly(
            ProviderInfo.newBuilder()
                .setProviderName("DocumentedInfo")
                .addFieldInfo(ProviderFieldInfo.newBuilder().setName("c").setDocString("C"))
                .addFieldInfo(ProviderFieldInfo.newBuilder().setName("a").setDocString("A"))
                .addFieldInfo(ProviderFieldInfo.newBuilder().setName("b").setDocString("B"))
                .build(),
            ProviderInfo.newBuilder()
                .setProviderName("UndocumentedInfo")
                .addFieldInfo(ProviderFieldInfo.newBuilder().setName("c"))
                .addFieldInfo(ProviderFieldInfo.newBuilder().setName("a"))
                .addFieldInfo(ProviderFieldInfo.newBuilder().setName("b"))
                .build());
  }

  @Test
  public void ruleDocstring() throws Exception {
    Module module =
        exec(
            "def _my_impl(ctx):",
            "    pass",
            "documented_lib = rule(doc = 'My doc', implementation = _my_impl)",
            "undocumented_lib = rule(implementation = _my_impl)");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getRuleInfoList())
        .ignoringFields(RuleInfo.ATTRIBUTE_FIELD_NUMBER) // ignore implicit attributes
        .containsExactly(
            RuleInfo.newBuilder().setRuleName("documented_lib").setDocString("My doc").build(),
            RuleInfo.newBuilder().setRuleName("undocumented_lib").build());
  }

  @Test
  public void ruleAttributes() throws Exception {
    Module module =
        exec(
            "MyInfo1 = provider()",
            "MyInfo2 = provider()",
            "MyInfo3 = provider()",
            "def _my_impl(ctx):",
            "    pass",
            "my_lib = rule(",
            "    implementation = _my_impl,",
            "    attrs = {",
            "        'a': attr.string(doc = 'My doc', default = 'foo'),",
            "        'b': attr.string(mandatory = True),",
            "        'c': attr.label(providers = [MyInfo1, MyInfo2]),",
            "        'd': attr.label(providers = [[MyInfo1, MyInfo2], [MyInfo3]]),",
            "        '_e': attr.string(doc = 'Hidden attribute'),",
            "    }",
            ")");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getRuleInfoList().get(0).getAttributeList())
        .containsExactly(
            ModuleInfoExtractor.IMPLICIT_NAME_ATTRIBUTE_INFO,
            AttributeInfo.newBuilder()
                .setName("a")
                .setType(AttributeType.STRING)
                .setDocString("My doc")
                .setDefaultValue("\"foo\"")
                .build(),
            AttributeInfo.newBuilder()
                .setName("b")
                .setType(AttributeType.STRING)
                .setMandatory(true)
                .build(),
            AttributeInfo.newBuilder()
                .setName("c")
                .setType(AttributeType.LABEL)
                .setDefaultValue("None")
                .addProviderNameGroup(
                    ProviderNameGroup.newBuilder()
                        .addProviderName("MyInfo1")
                        .addProviderName("MyInfo2"))
                .build(),
            AttributeInfo.newBuilder()
                .setName("d")
                .setType(AttributeType.LABEL)
                .setDefaultValue("None")
                .addProviderNameGroup(
                    ProviderNameGroup.newBuilder()
                        .addProviderName("MyInfo1")
                        .addProviderName("MyInfo2"))
                .addProviderNameGroup(ProviderNameGroup.newBuilder().addProviderName("MyInfo3"))
                .build());
  }

  @Test
  public void attributeOrder() throws Exception {
    Module module =
        exec(
            "def _my_impl(ctx):",
            "    pass",
            "my_lib = rule(",
            "    implementation = _my_impl,",
            "    attrs = {",
            "        'foo': attr.int(),",
            "        'bar': attr.int(),",
            "        'baz': attr.int(),",
            "    }",
            ")");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(
            moduleInfo.getRuleInfoList().get(0).getAttributeList().stream()
                .map(AttributeInfo::getName))
        .containsExactly("name", "foo", "bar", "baz")
        .inOrder();
  }

  @Test
  public void attributeTypes() throws Exception {
    Module module =
        exec(
            "def _my_impl(ctx):",
            "    pass",
            "my_lib = rule(",
            "    implementation = _my_impl,",
            "    attrs = {",
            "        'a': attr.int(),",
            "        'b': attr.label(),",
            "        'c': attr.string(),",
            "        'd': attr.string_list(),",
            "        'e': attr.int_list(),",
            "        'f': attr.label_list(),",
            "        'g': attr.bool(),",
            "        'h': attr.label_keyed_string_dict(),",
            "        'i': attr.string_dict(),",
            "        'j': attr.string_list_dict(),",
            "        'k': attr.output(),",
            "        'l': attr.output_list(),",
            "    }",
            ")");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getRuleInfoList().get(0).getAttributeList())
        .containsExactly(
            ModuleInfoExtractor.IMPLICIT_NAME_ATTRIBUTE_INFO,
            AttributeInfo.newBuilder()
                .setName("a")
                .setType(AttributeType.INT)
                .setDefaultValue("0")
                .build(),
            AttributeInfo.newBuilder()
                .setName("b")
                .setType(AttributeType.LABEL)
                .setDefaultValue("None")
                .build(),
            AttributeInfo.newBuilder()
                .setName("c")
                .setType(AttributeType.STRING)
                .setDefaultValue("\"\"")
                .build(),
            AttributeInfo.newBuilder()
                .setName("d")
                .setType(AttributeType.STRING_LIST)
                .setDefaultValue("[]")
                .build(),
            AttributeInfo.newBuilder()
                .setName("e")
                .setType(AttributeType.INT_LIST)
                .setDefaultValue("[]")
                .build(),
            AttributeInfo.newBuilder()
                .setName("f")
                .setType(AttributeType.LABEL_LIST)
                .setDefaultValue("[]")
                .build(),
            AttributeInfo.newBuilder()
                .setName("g")
                .setType(AttributeType.BOOLEAN)
                .setDefaultValue("False")
                .build(),
            AttributeInfo.newBuilder()
                .setName("h")
                .setType(AttributeType.LABEL_STRING_DICT)
                .setDefaultValue("{}")
                .build(),
            AttributeInfo.newBuilder()
                .setName("i")
                .setType(AttributeType.STRING_DICT)
                .setDefaultValue("{}")
                .build(),
            AttributeInfo.newBuilder()
                .setName("j")
                .setType(AttributeType.STRING_LIST_DICT)
                .setDefaultValue("{}")
                .build(),
            AttributeInfo.newBuilder()
                .setName("k")
                .setType(AttributeType.OUTPUT)
                .setDefaultValue("None")
                .build(),
            AttributeInfo.newBuilder()
                .setName("l")
                .setType(AttributeType.OUTPUT_LIST)
                .setDefaultValue("[]")
                .build());
  }

  @Test
  public void labelStringification() throws Exception {
    Module module =
        exec(
            "def _my_impl(ctx):",
            "    pass",
            "my_lib = rule(",
            "    implementation = _my_impl,",
            "    attrs = {",
            "        'label': attr.label(default = '//test:foo'),",
            "        'label_list': attr.label_list(",
            "            default = ['//x', '@canonical//y', '@canonical//y:z'],",
            "        ),",
            "        'label_keyed_string_dict': attr.label_keyed_string_dict(",
            "           default = {'//x': 'label_in_main', '@canonical//y': 'label_in_dep'}",
            "         ),",
            "    }",
            ")");
    RepositoryName canonicalName = RepositoryName.create("canonical");
    RepositoryMapping repositoryMapping =
        RepositoryMapping.create(ImmutableMap.of("local", canonicalName), RepositoryName.MAIN);
    ModuleInfo moduleInfo = getExtractor(repositoryMapping).extractFrom(module);
    assertThat(
            moduleInfo.getRuleInfoList().get(0).getAttributeList().stream()
                .filter(attr -> !attr.equals(ModuleInfoExtractor.IMPLICIT_NAME_ATTRIBUTE_INFO))
                .map(AttributeInfo::getDefaultValue))
        .containsExactly(
            "\"//test:foo\"",
            "[\"//x\", \"@local//y\", \"@local//y:z\"]",
            "{\"//x\": \"label_in_main\", \"@local//y\": \"label_in_dep\"}");
  }

  @Test
  public void aspectDocstring() throws Exception {
    Module module =
        exec(
            "def _my_impl(target, ctx):",
            "    pass",
            "documented_aspect = aspect(doc = 'My doc', implementation = _my_impl)",
            "undocumented_aspect = aspect(implementation = _my_impl)");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getAspectInfoList())
        .ignoringFields(AspectInfo.ATTRIBUTE_FIELD_NUMBER) // ignore implicit attributes
        .containsExactly(
            AspectInfo.newBuilder()
                .setAspectName("documented_aspect")
                .setDocString("My doc")
                .build(),
            AspectInfo.newBuilder().setAspectName("undocumented_aspect").build());
  }

  @Test
  public void aspectAttributes() throws Exception {
    Module module =
        exec(
            "def _my_impl(target, ctx):",
            "    pass",
            "my_aspect = aspect(",
            "    implementation = _my_impl,",
            "    attr_aspects = ['deps', 'srcs'],",
            "    attrs = {",
            "        'a': attr.string(doc = 'My doc', default = 'foo'),",
            "        'b': attr.string(mandatory = True),",
            "        '_c': attr.string(doc = 'Hidden attribute'),",
            "    }",
            ")");
    ModuleInfo moduleInfo = getExtractor().extractFrom(module);
    assertThat(moduleInfo.getAspectInfoList())
        .containsExactly(
            AspectInfo.newBuilder()
                .setAspectName("my_aspect")
                .addAspectAttribute("deps")
                .addAspectAttribute("srcs")
                .addAttribute(ModuleInfoExtractor.IMPLICIT_NAME_ATTRIBUTE_INFO)
                .addAttribute(
                    AttributeInfo.newBuilder()
                        .setName("a")
                        .setType(AttributeType.STRING)
                        .setDocString("My doc")
                        .setDefaultValue("\"foo\""))
                .addAttribute(
                    AttributeInfo.newBuilder()
                        .setName("b")
                        .setType(AttributeType.STRING)
                        .setMandatory(true)
                        .build())
                .build());
  }
}
