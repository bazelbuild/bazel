package build.stack.devtools.build.constellate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import build.stack.devtools.build.constellate.rendering.ProtoRenderer;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.devtools.common.options.OptionsParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.ParserInput;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Constellate}. */
@RunWith(JUnit4.class)
public class ConstellateTest {

  private static final String TEST_DATA_DIR =
      "src/test/java/build/stack/devtools/build/constellate/testdata";

  private ModuleInfo evaluateFile(String filename) throws Exception {
    // Read the test file
    Path testFile = Paths.get(TEST_DATA_DIR, filename);
    assertTrue("Test file not found: " + testFile, Files.exists(testFile));

    // Create parser options
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(BuildLanguageOptions.class).build();
    BuildLanguageOptions semanticsOptions = parser.getOptions(BuildLanguageOptions.class);
    StarlarkSemantics semantics = semanticsOptions.toStarlarkSemantics();

    // Create constellate
    Constellate constellate =
        new Constellate(
            semantics,
            new FilesystemFileAccessor(),
            /* depRoots= */ ImmutableList.of());

    // Create label for the test file
    Label label = Label.parseCanonicalUnchecked("//test:" + filename);

    // Prepare data structures
    ImmutableMap.Builder<String, RuleInfo> ruleInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, ProviderInfo> providerInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, StarlarkFunction> userDefinedFunctions = ImmutableMap.builder();
    ImmutableMap.Builder<String, AspectInfo> aspectInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, RepositoryRuleInfo> repositoryRuleInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, ModuleExtensionInfo> moduleExtensionInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<String, MacroInfo> macroInfoMap = ImmutableMap.builder();
    ImmutableMap.Builder<Label, String> moduleDocMap = ImmutableMap.builder();
    ImmutableMap.Builder<Label, Map<String, Object>> globals = ImmutableMap.builder();

    // Parse and evaluate
    ParserInput input = ParserInput.fromLatin1(Files.readAllBytes(testFile), filename);

    build.stack.starlark.v1beta1.StarlarkProtos.Module.Builder moduleBuilder =
        build.stack.starlark.v1beta1.StarlarkProtos.Module.newBuilder();

    constellate.eval(
        input,
        label,
        ruleInfoMap,
        providerInfoMap,
        userDefinedFunctions,
        aspectInfoMap,
        repositoryRuleInfoMap,
        moduleExtensionInfoMap,
        macroInfoMap,
        moduleDocMap,
        moduleBuilder,
        globals);

    // Render to ModuleInfo
    ProtoRenderer renderer = new ProtoRenderer();
    renderer.appendRuleInfos(ruleInfoMap.build().values());
    renderer.appendProviderInfos(providerInfoMap.build().values());
    renderer.appendStarlarkFunctionInfos(userDefinedFunctions.build());
    renderer.appendAspectInfos(aspectInfoMap.build().values());
    renderer.setModuleDocstring(moduleDocMap.build().get(label));

    return renderer.getModuleInfo().build();
  }

  @Test
  public void testSimpleFile() throws Exception {
    ModuleInfo moduleInfo = evaluateFile("simple_test.bzl");

    assertNotNull(moduleInfo);

    // Should have at least 1 function (may also capture implementation functions starting with _)
    assertTrue(
        "Should have at least 1 function, got " + moduleInfo.getFuncInfoCount(),
        moduleInfo.getFuncInfoCount() >= 1);

    // Find simple_function
    StarlarkFunctionInfo funcInfo = null;
    for (StarlarkFunctionInfo func : moduleInfo.getFuncInfoList()) {
      if (func.getFunctionName().equals("simple_function")) {
        funcInfo = func;
        break;
      }
    }
    assertNotNull("simple_function should be extracted", funcInfo);
    assertTrue("OriginKey should be set", funcInfo.hasOriginKey());
    assertFalse("OriginKey name should be set", funcInfo.getOriginKey().getName().isEmpty());

    // Should have 1 provider
    assertEquals(1, moduleInfo.getProviderInfoCount());
    ProviderInfo providerInfo = moduleInfo.getProviderInfo(0);
    assertEquals("SimpleInfo", providerInfo.getProviderName());
    assertTrue("OriginKey should be set", providerInfo.hasOriginKey());
    assertFalse("OriginKey name should be set", providerInfo.getOriginKey().getName().isEmpty());

    // Should have 1 rule
    assertEquals(1, moduleInfo.getRuleInfoCount());
    RuleInfo ruleInfo = moduleInfo.getRuleInfo(0);
    assertEquals("simple_rule", ruleInfo.getRuleName());
    assertTrue("OriginKey should be set", ruleInfo.hasOriginKey());
    assertFalse("OriginKey name should be set", ruleInfo.getOriginKey().getName().isEmpty());
  }

  @Test
  public void testComprehensiveFile() throws Exception {
    ModuleInfo moduleInfo = evaluateFile("comprehensive_test.bzl");

    assertNotNull(moduleInfo);

    // ===== STARLARK FUNCTIONS =====
    assertTrue(
        "Should have at least 1 function, got " + moduleInfo.getFuncInfoCount(),
        moduleInfo.getFuncInfoCount() >= 1);

    // Check my_function - comprehensive function with all parameter types
    StarlarkFunctionInfo myFunc = null;
    for (StarlarkFunctionInfo func : moduleInfo.getFuncInfoList()) {
      if (func.getFunctionName().equals("my_function")) {
        myFunc = func;
        break;
      }
    }
    assertNotNull("my_function should be extracted", myFunc);

    // StarlarkFunctionInfo fields
    assertTrue("my_function should have OriginKey", myFunc.hasOriginKey());
    assertEquals("OriginKey.name should be my_function", "my_function", myFunc.getOriginKey().getName());
    assertFalse("OriginKey.file should be set", myFunc.getOriginKey().getFile().isEmpty());
    assertTrue("my_function should have docstring", myFunc.getDocString().length() > 0);
    assertTrue("my_function should have return info", myFunc.hasReturn());
    assertTrue("my_function should have deprecation info", myFunc.hasDeprecated());

    // FunctionParamInfo - should have param1, param2, *args, **kwargs
    assertEquals("Should have 4 parameters", 4, myFunc.getParameterCount());
    // Verify parameter roles are correctly identified
    boolean hasOrdinary = false, hasKwargs = false;
    for (com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamInfo param : myFunc.getParameterList()) {
      if (param.getRole() == com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_ORDINARY) {
        hasOrdinary = true;
      }
      if (param.getRole() == com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.FunctionParamRole.PARAM_ROLE_KWARGS) {
        hasKwargs = true;
      }
    }
    assertTrue("Should have ordinary parameters", hasOrdinary);
    assertTrue("Should have **kwargs parameter", hasKwargs);

    // ===== PROVIDERS =====
    assertTrue(
        "Should have at least 2 providers, got " + moduleInfo.getProviderInfoCount(),
        moduleInfo.getProviderInfoCount() >= 2);

    // Check MyInfoProvider - with init callback and field schema
    ProviderInfo myInfo = null;
    for (ProviderInfo provider : moduleInfo.getProviderInfoList()) {
      if (provider.getProviderName().equals("MyInfoProvider")) {
        myInfo = provider;
        break;
      }
    }
    assertNotNull("MyInfoProvider should be extracted", myInfo);

    // ProviderInfo fields
    assertTrue("MyInfoProvider should have OriginKey", myInfo.hasOriginKey());
    assertEquals("OriginKey.name should be MyInfoProvider", "MyInfoProvider", myInfo.getOriginKey().getName());
    assertFalse("OriginKey.file should be set", myInfo.getOriginKey().getFile().isEmpty());
    assertTrue("MyInfoProvider should have docstring", myInfo.getDocString().length() > 0);

    // ProviderFieldInfo - should have "value" and "count" fields
    assertTrue("MyInfoProvider should have field documentation", myInfo.getFieldInfoCount() >= 2);
    boolean hasValueField = false, hasCountField = false;
    for (com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderFieldInfo field : myInfo.getFieldInfoList()) {
      if (field.getName().equals("value")) {
        hasValueField = true;
        assertTrue("value field should have docstring", field.getDocString().length() > 0);
      }
      if (field.getName().equals("count")) {
        hasCountField = true;
        assertTrue("count field should have docstring", field.getDocString().length() > 0);
      }
    }
    assertTrue("Should have 'value' field", hasValueField);
    assertTrue("Should have 'count' field", hasCountField);

    // ProviderInfo.init - BLOCKED BY FAKE API
    // TODO: Provider init callback extraction requires real StarlarkProvider objects,
    // which are not available when using fake API. This is a known architectural limitation.
    // See STARDOC_PROTO_COVERAGE.md for details.
    // assertFalse("MyInfoProvider init should NOT be extracted (fake API limitation)", myInfo.hasInit());

    // Check SimpleProvider - without init
    ProviderInfo simpleProvider = null;
    for (ProviderInfo provider : moduleInfo.getProviderInfoList()) {
      if (provider.getProviderName().equals("SimpleProvider")) {
        simpleProvider = provider;
        break;
      }
    }
    assertNotNull("SimpleProvider should be extracted", simpleProvider);
    assertTrue("SimpleProvider should have OriginKey", simpleProvider.hasOriginKey());

    // ===== RULES =====
    assertTrue(
        "Should have at least 1 rule, got " + moduleInfo.getRuleInfoCount(),
        moduleInfo.getRuleInfoCount() >= 1);

    // Check my_rule
    RuleInfo myRule = null;
    for (RuleInfo rule : moduleInfo.getRuleInfoList()) {
      if (rule.getRuleName().equals("my_rule")) {
        myRule = rule;
        break;
      }
    }
    assertNotNull("my_rule should be extracted", myRule);

    // RuleInfo fields
    assertTrue("my_rule should have OriginKey", myRule.hasOriginKey());
    assertEquals("OriginKey.name should be my_rule", "my_rule", myRule.getOriginKey().getName());
    assertFalse("OriginKey.file should be set", myRule.getOriginKey().getFile().isEmpty());
    assertTrue("my_rule should have docstring", myRule.getDocString().length() > 0);

    // AttributeInfo - should have "value" and "deps" attributes
    assertTrue("my_rule should have attributes", myRule.getAttributeCount() >= 2);
    boolean hasValueAttr = false, hasDepsAttr = false;
    for (com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo attr : myRule.getAttributeList()) {
      if (attr.getName().equals("value")) {
        hasValueAttr = true;
        assertEquals("value should be STRING type",
            com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType.STRING,
            attr.getType());
        assertTrue("value should have docstring", attr.getDocString().length() > 0);
        assertFalse("value should not be mandatory", attr.getMandatory());
        assertTrue("value should have default value", attr.getDefaultValue().length() > 0);
      }
      if (attr.getName().equals("deps")) {
        hasDepsAttr = true;
        assertEquals("deps should be LABEL_LIST type",
            com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeType.LABEL_LIST,
            attr.getType());
        assertTrue("deps should have docstring", attr.getDocString().length() > 0);
        // TODO: AttributeInfo.provider_name_group extraction requires real Attribute objects,
        // which are not available when using fake API. This is a known limitation.
        // See STARDOC_PROTO_COVERAGE.md for details.
        // assertTrue("deps should have provider requirements", attr.getProviderNameGroupCount() > 0);
      }
    }
    assertTrue("Should have 'value' attribute", hasValueAttr);
    assertTrue("Should have 'deps' attribute", hasDepsAttr);

    // RuleInfo.advertised_providers - BLOCKED BY FAKE API
    // TODO: Advertised providers extraction requires real StarlarkRuleFunction objects,
    // which are not available when using fake API. This is a known architectural limitation.
    // See STARDOC_PROTO_COVERAGE.md for details.
    // assertFalse("my_rule advertised_providers should NOT be extracted (fake API limitation)",
    //     myRule.hasAdvertisedProviders());

    // ===== ASPECTS =====
    assertTrue(
        "Should have at least 1 aspect, got " + moduleInfo.getAspectInfoCount(),
        moduleInfo.getAspectInfoCount() >= 1);

    // Check my_aspect
    AspectInfo myAspect = null;
    for (AspectInfo aspect : moduleInfo.getAspectInfoList()) {
      if (aspect.getAspectName().equals("my_aspect")) {
        myAspect = aspect;
        break;
      }
    }
    assertNotNull("my_aspect should be extracted", myAspect);

    // AspectInfo fields
    assertTrue("my_aspect should have OriginKey", myAspect.hasOriginKey());
    assertEquals("OriginKey.name should be my_aspect", "my_aspect", myAspect.getOriginKey().getName());
    assertFalse("OriginKey.file should be set", myAspect.getOriginKey().getFile().isEmpty());
    assertTrue("my_aspect should have docstring", myAspect.getDocString().length() > 0);
    assertTrue("my_aspect should have aspect_attribute", myAspect.getAspectAttributeCount() > 0);
    assertEquals("my_aspect should propagate on 'deps'", "deps", myAspect.getAspectAttribute(0));
    assertTrue("my_aspect should have attributes", myAspect.getAttributeCount() > 0);

    // Verify aspect has aspect_param attribute
    boolean hasAspectParam = false;
    for (com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AttributeInfo attr : myAspect.getAttributeList()) {
      if (attr.getName().equals("aspect_param")) {
        hasAspectParam = true;
        assertTrue("aspect_param should have docstring", attr.getDocString().length() > 0);
      }
    }
    assertTrue("Should have 'aspect_param' attribute", hasAspectParam);

    // ===== MODULE INFO LEVEL FIELDS =====
    // Check module docstring
    assertTrue("Module docstring should not be empty", moduleInfo.getModuleDocstring().length() > 0);
  }
}
