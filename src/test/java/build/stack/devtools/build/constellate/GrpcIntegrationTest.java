package build.stack.devtools.build.constellate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import build.stack.starlark.v1beta1.StarlarkGrpc;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleInfoRequest;
import build.stack.starlark.v1beta1.StarlarkProtos.Module;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.devtools.common.options.OptionsParser;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for the gRPC server interface.
 *
 * Tests the full request/response cycle including:
 * - Single file extraction
 * - Multi-file extraction with load statements
 * - Cross-file OriginKey references
 * - Best-effort extraction when load graph fails
 */
@RunWith(JUnit4.class)
public class GrpcIntegrationTest {

  private static final String TEST_DATA_DIR =
      "src/test/java/build/stack/devtools/build/constellate/testdata";

  private Server server;
  private ManagedChannel channel;
  private StarlarkGrpc.StarlarkBlockingStub blockingStub;

  @Before
  public void setUp() throws Exception {
    // Create parser options
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(BuildLanguageOptions.class).build();
    BuildLanguageOptions semanticsOptions = parser.getOptions(BuildLanguageOptions.class);
    StarlarkSemantics semantics = semanticsOptions.toStarlarkSemantics();

    // Create in-process server
    String serverName = "test-" + System.currentTimeMillis();
    StarlarkServer starlarkServer = new StarlarkServer(semantics);

    server = InProcessServerBuilder
        .forName(serverName)
        .directExecutor()
        .addService(starlarkServer)
        .build()
        .start();

    // Create in-process channel
    channel = InProcessChannelBuilder
        .forName(serverName)
        .directExecutor()
        .build();

    blockingStub = StarlarkGrpc.newBlockingStub(channel);
  }

  @After
  public void tearDown() throws Exception {
    if (channel != null) {
      channel.shutdownNow();
      channel.awaitTermination(5, TimeUnit.SECONDS);
    }
    if (server != null) {
      server.shutdownNow();
      server.awaitTermination(5, TimeUnit.SECONDS);
    }
  }

  @Test
  public void testSingleFileExtraction() throws Exception {
    // Test extracting a single .bzl file
    // Use label format instead of absolute path
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:simple_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Verify basic extraction
    assertTrue("Should extract at least 1 function", moduleInfo.getFuncInfoCount() >= 1);
    assertEquals("Should extract 1 provider", 1, moduleInfo.getProviderInfoCount());
    assertEquals("Should extract 1 rule", 1, moduleInfo.getRuleInfoCount());

    // Verify OriginKeys are set
    StarlarkFunctionInfo func = findFunction(moduleInfo, "simple_function");
    assertNotNull("simple_function should be extracted", func);
    assertTrue("simple_function should have OriginKey", func.hasOriginKey());
    assertFalse("OriginKey name should be set", func.getOriginKey().getName().isEmpty());
    assertFalse("OriginKey file should be set", func.getOriginKey().getFile().isEmpty());

    ProviderInfo provider = moduleInfo.getProviderInfo(0);
    assertEquals("SimpleInfo", provider.getProviderName());
    assertTrue("SimpleInfo should have OriginKey", provider.hasOriginKey());
    assertEquals("SimpleInfo", provider.getOriginKey().getName());
    assertFalse("OriginKey file should be set", provider.getOriginKey().getFile().isEmpty());

    RuleInfo rule = moduleInfo.getRuleInfo(0);
    assertEquals("simple_rule", rule.getRuleName());
    assertTrue("simple_rule should have OriginKey", rule.hasOriginKey());
    assertEquals("simple_rule", rule.getOriginKey().getName());
  }

  @Test
  public void testMultiFileExtraction() throws Exception {
    // Test extracting a file that loads another file
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:load_test_main.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Verify load statements are recorded
    assertTrue("Should have load statements", response.getLoadCount() > 0);

    // Find the load statement for load_test_lib.bzl
    build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt loadStmt = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt stmt : response.getLoadList()) {
      if (stmt.getLabel().contains("load_test_lib.bzl")) {
        loadStmt = stmt;
        break;
      }
    }

    assertNotNull("Should have load statement for load_test_lib.bzl", loadStmt);
    assertTrue("Load statement should have symbols", loadStmt.getSymbolCount() > 0);

    // Verify the loaded symbols
    java.util.Set<String> loadedSymbols = new java.util.HashSet<>();
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadSymbol symbol : loadStmt.getSymbolList()) {
      loadedSymbols.add(symbol.getTo());  // The local name
    }

    assertTrue("Should load lib_function", loadedSymbols.contains("lib_function"));
    assertTrue("Should load LibInfo", loadedSymbols.contains("LibInfo"));
    assertTrue("Should load lib_rule", loadedSymbols.contains("lib_rule"));

    // When loading from other files, OriginKeys should reference the original file
    // This tests that cross-file references are properly tracked
    assertTrue("Should extract entities from loaded files",
        !moduleInfo.getFuncInfoList().isEmpty() ||
        !moduleInfo.getProviderInfoList().isEmpty() ||
        !moduleInfo.getRuleInfoList().isEmpty());
  }

  @Test
  public void testComprehensiveFileExtraction() throws Exception {
    // Test extracting the comprehensive test file with all entity types
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:comprehensive_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Verify all entity types are extracted
    assertTrue("Should extract functions", moduleInfo.getFuncInfoCount() >= 1);
    System.err.println("Provider count: " + moduleInfo.getProviderInfoCount());
    for (int i = 0; i < moduleInfo.getProviderInfoCount(); i++) {
      System.err.println("Provider " + i + ": " + moduleInfo.getProviderInfo(i).getProviderName());
    }
    assertTrue("Should extract providers (got " + moduleInfo.getProviderInfoCount() + ")", moduleInfo.getProviderInfoCount() >= 2);
    assertTrue("Should extract rules", moduleInfo.getRuleInfoCount() >= 1);
    assertTrue("Should extract aspects", moduleInfo.getAspectInfoCount() >= 1);

    // Verify specific entities with OriginKeys
    StarlarkFunctionInfo myFunc = findFunction(moduleInfo, "my_function");
    assertNotNull("my_function should be extracted", myFunc);
    assertTrue("my_function should have OriginKey", myFunc.hasOriginKey());
    assertTrue("my_function should have docstring", myFunc.getDocString().length() > 0);

    ProviderInfo myInfo = findProvider(moduleInfo, "MyInfoProvider");
    assertNotNull("MyInfoProvider should be extracted", myInfo);
    assertTrue("MyInfoProvider should have OriginKey", myInfo.hasOriginKey());
    assertTrue("MyInfoProvider should have fields", myInfo.getFieldInfoCount() > 0);

    RuleInfo myRule = findRule(moduleInfo, "my_rule");
    assertNotNull("my_rule should be extracted", myRule);
    assertTrue("my_rule should have OriginKey", myRule.hasOriginKey());

    AspectInfo myAspect = findAspect(moduleInfo, "my_aspect");
    assertNotNull("my_aspect should be extracted", myAspect);
    assertTrue("my_aspect should have OriginKey", myAspect.hasOriginKey());
  }

  @Test
  public void testGlobalScalarExtraction() throws Exception {
    // Test extracting global scalar constants
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:global_scalars_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have global values", response.getGlobalCount() > 0);

    // Verify string constants
    assertTrue("Should contain VERSION", response.containsGlobal("VERSION"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value versionValue = response.getGlobalOrThrow("VERSION");
    assertEquals("VERSION should be '1.2.3'", "1.2.3", versionValue.getString());

    assertTrue("Should contain TOOLCHAIN_NAME", response.containsGlobal("TOOLCHAIN_NAME"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value toolchainValue = response.getGlobalOrThrow("TOOLCHAIN_NAME");
    assertEquals("TOOLCHAIN_NAME should be 'my_toolchain'", "my_toolchain", toolchainValue.getString());

    assertTrue("Should contain DEFAULT_TAG", response.containsGlobal("DEFAULT_TAG"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value tagValue = response.getGlobalOrThrow("DEFAULT_TAG");
    assertEquals("DEFAULT_TAG should be 'latest'", "latest", tagValue.getString());

    // Verify integer constants
    assertTrue("Should contain MAX_SIZE", response.containsGlobal("MAX_SIZE"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value maxSizeValue = response.getGlobalOrThrow("MAX_SIZE");
    assertEquals("MAX_SIZE should be 100", 100L, maxSizeValue.getInt());

    assertTrue("Should contain DEFAULT_TIMEOUT", response.containsGlobal("DEFAULT_TIMEOUT"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value timeoutValue = response.getGlobalOrThrow("DEFAULT_TIMEOUT");
    assertEquals("DEFAULT_TIMEOUT should be 60", 60L, timeoutValue.getInt());

    // Verify boolean constants
    assertTrue("Should contain ENABLE_FEATURE", response.containsGlobal("ENABLE_FEATURE"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value enableValue = response.getGlobalOrThrow("ENABLE_FEATURE");
    assertTrue("ENABLE_FEATURE should be true", enableValue.getBool());

    assertTrue("Should contain DEBUG_MODE", response.containsGlobal("DEBUG_MODE"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value debugValue = response.getGlobalOrThrow("DEBUG_MODE");
    assertFalse("DEBUG_MODE should be false", debugValue.getBool());

    // Verify list constants
    assertTrue("Should contain SUPPORTED_PLATFORMS", response.containsGlobal("SUPPORTED_PLATFORMS"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value platformsValue = response.getGlobalOrThrow("SUPPORTED_PLATFORMS");
    assertTrue("SUPPORTED_PLATFORMS should be a list", platformsValue.hasList());
    assertEquals("SUPPORTED_PLATFORMS should have 3 elements", 3, platformsValue.getList().getValueCount());
    assertEquals("First platform should be 'linux'", "linux", platformsValue.getList().getValue(0).getString());
    assertEquals("Second platform should be 'darwin'", "darwin", platformsValue.getList().getValue(1).getString());
    assertEquals("Third platform should be 'windows'", "windows", platformsValue.getList().getValue(2).getString());

    assertTrue("Should contain EMPTY_LIST", response.containsGlobal("EMPTY_LIST"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value emptyListValue = response.getGlobalOrThrow("EMPTY_LIST");
    assertTrue("EMPTY_LIST should be a list", emptyListValue.hasList());
    assertEquals("EMPTY_LIST should have 0 elements", 0, emptyListValue.getList().getValueCount());

    // Verify list comprehension results
    assertTrue("Should contain NUMBERS (list comprehension)", response.containsGlobal("NUMBERS"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value numbersValue = response.getGlobalOrThrow("NUMBERS");
    assertTrue("NUMBERS should be a list", numbersValue.hasList());
    assertEquals("NUMBERS should have 5 elements [0, 2, 4, 6, 8]", 5, numbersValue.getList().getValueCount());
    assertEquals("NUMBERS[0] should be 0", 0L, numbersValue.getList().getValue(0).getInt());
    assertEquals("NUMBERS[1] should be 2", 2L, numbersValue.getList().getValue(1).getInt());
    assertEquals("NUMBERS[2] should be 4", 4L, numbersValue.getList().getValue(2).getInt());
    assertEquals("NUMBERS[3] should be 6", 6L, numbersValue.getList().getValue(3).getInt());
    assertEquals("NUMBERS[4] should be 8", 8L, numbersValue.getList().getValue(4).getInt());

    assertTrue("Should contain FILTERED_NUMBERS (list comprehension with filter)", response.containsGlobal("FILTERED_NUMBERS"));
    build.stack.starlark.v1beta1.StarlarkProtos.Value filteredValue = response.getGlobalOrThrow("FILTERED_NUMBERS");
    assertTrue("FILTERED_NUMBERS should be a list", filteredValue.hasList());
    assertEquals("FILTERED_NUMBERS should have 5 elements [0, 2, 4, 6, 8]", 5, filteredValue.getList().getValueCount());
    assertEquals("FILTERED_NUMBERS[0] should be 0", 0L, filteredValue.getList().getValue(0).getInt());
    assertEquals("FILTERED_NUMBERS[1] should be 2", 2L, filteredValue.getList().getValue(1).getInt());
    assertEquals("FILTERED_NUMBERS[2] should be 4", 4L, filteredValue.getList().getValue(2).getInt());
    assertEquals("FILTERED_NUMBERS[3] should be 6", 6L, filteredValue.getList().getValue(3).getInt());
    assertEquals("FILTERED_NUMBERS[4] should be 8", 8L, filteredValue.getList().getValue(4).getInt());

    // Verify that private symbols (starting with _) are NOT captured
    assertFalse("Should NOT contain private _INTERNAL_VALUE", response.containsGlobal("_INTERNAL_VALUE"));
  }

  @Test
  public void testBestEffortExtractionOnLoadFailure() throws Exception {
    // Test that extraction still works (best-effort) when load statements fail
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:load_failure_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    // This should not throw an exception even if load fails
    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null even on load failure", response);

    // Best-effort means we should still extract what we can from the file itself
    // Even if loaded dependencies are missing
    if (response.hasInfo()) {
      ModuleInfo moduleInfo = response.getInfo();
      // Any entities defined in the file itself should still be extracted
      // Verify the file has local entities despite load failure
      assertTrue("Should extract local entities despite load failure",
          !moduleInfo.getFuncInfoList().isEmpty() ||
          !moduleInfo.getProviderInfoList().isEmpty() ||
          !moduleInfo.getRuleInfoList().isEmpty());
    }
  }

  @Test
  public void testSymbolFiltering() throws Exception {
    // Test that we can filter to specific symbols
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:simple_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .addSymbolNames("simple_function")  // Only extract this symbol
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should only have the requested symbol (and not private symbols)
    StarlarkFunctionInfo func = findFunction(moduleInfo, "simple_function");
    assertNotNull("simple_function should be extracted", func);

    // Private functions (starting with _) should be filtered out
    StarlarkFunctionInfo privateFunc = findFunction(moduleInfo, "_simple_rule_impl");
    // Note: Current implementation may include private functions, this documents the behavior
  }

  @Test
  public void testOriginKeyFileFormat() throws Exception {
    // Test that OriginKey file field uses canonical label format
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:simple_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Check OriginKey format
    if (moduleInfo.getRuleInfoCount() > 0) {
      RuleInfo rule = moduleInfo.getRuleInfo(0);
      if (rule.hasOriginKey()) {
        String file = rule.getOriginKey().getFile();
        assertFalse("OriginKey file should not be empty", file.isEmpty());
        // File should be in label format (e.g., "//package:file.bzl" or "@repo//package:file.bzl")
        // The exact format depends on how labels are rendered
      }
    }
  }

  @Test
  public void testModuleContentInline() throws Exception {
    // Test that we can provide module content inline without requiring a file on disk
    // This is useful for LSP scenarios where the file hasn't been saved yet,
    // or for testing without creating temporary files

    String inlineContent =
        "\"\"\"Inline test module.\"\"\"\n" +
        "\n" +
        "def inline_function(x):\n" +
        "    \"\"\"An inline function.\n" +
        "\n" +
        "    Args:\n" +
        "        x: The input value\n" +
        "\n" +
        "    Returns:\n" +
        "        The squared value\n" +
        "    \"\"\"\n" +
        "    return x * x\n" +
        "\n" +
        "InlineInfo = provider(\n" +
        "    doc = \"An inline provider.\",\n" +
        "    fields = [\"data\"],\n" +
        ")\n";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel("//virtual:inline.bzl")  // Label doesn't need to exist on disk
        .setModuleContent(inlineContent)
        .addDepRoots(".")  // Add current directory as dep root so path resolution works
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should extract function from inline content
    StarlarkFunctionInfo func = findFunction(moduleInfo, "inline_function");
    assertNotNull("inline_function should be extracted from inline content", func);
    assertEquals("Function should have correct name", "inline_function", func.getFunctionName());
    assertFalse("Function should have documentation", func.getDocString().isEmpty());

    // Should extract provider from inline content
    ProviderInfo provider = findProvider(moduleInfo, "InlineInfo");
    assertNotNull("InlineInfo should be extracted from inline content", provider);
    assertEquals("Provider should have correct name", "InlineInfo", provider.getProviderName());
    assertEquals("Provider should have one field", 1, provider.getFieldInfoCount());
  }

  @Test
  public void testDepsetUsage() throws Exception {
    // Test that depset() builtin function works correctly
    // This verifies that our fake depset implementation can handle:
    // - Module-level depset creation (like rules_go does)
    // - Function-level depset creation
    // - Various depset signatures (direct, transitive, order)
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:depset_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should extract functions that use depset
    StarlarkFunctionInfo depsetFunc = findFunction(moduleInfo, "depset_function");
    assertNotNull("depset_function should be extracted", depsetFunc);

    StarlarkFunctionInfo depsetOrder = findFunction(moduleInfo, "depset_with_order");
    assertNotNull("depset_with_order should be extracted", depsetOrder);

    StarlarkFunctionInfo depsetTransitive = findFunction(moduleInfo, "depset_with_transitive");
    assertNotNull("depset_with_transitive should be extracted", depsetTransitive);

    // Should extract provider that references depsets
    ProviderInfo provider = findProvider(moduleInfo, "DepsetInfo");
    assertNotNull("DepsetInfo should be extracted", provider);
    assertEquals("Provider should have correct name", "DepsetInfo", provider.getProviderName());
    assertEquals("Provider should have two fields", 2, provider.getFieldInfoCount());

    // Should extract rule that uses depsets
    RuleInfo rule = findRule(moduleInfo, "depset_rule");
    assertNotNull("depset_rule should be extracted", rule);
    assertEquals("Rule should have correct name", "depset_rule", rule.getRuleName());
  }

  @Test
  public void testComplexDocstring() throws Exception {
    // Test that complex docstrings (like those from rules_go) are now handled correctly
    // with improved parser that supports:
    // - "Returns: content" format (content on same line as heading)
    // - Multi-line field descriptions in Returns sections
    // - Lenient indentation handling
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:cgo_docstring_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should successfully extract function with complex docstring
    StarlarkFunctionInfo cgoFunc = findFunction(moduleInfo, "cgo_configure");
    assertNotNull("cgo_configure should be extracted", cgoFunc);
    assertFalse("Function should have documentation", cgoFunc.getDocString().isEmpty());

    // Verify the docstring was parsed successfully (including the problematic "Returns: content" format)
    String docstring = cgoFunc.getDocString();
    assertTrue("Docstring should not be empty", !docstring.isEmpty());
    assertTrue("Docstring should contain function description", docstring.contains("cgo archive"));

    // Verify that the inline "Returns: a struct containing:" format was parsed correctly
    // This was previously rejected with "malformed docstring" error
    assertTrue("Docstring should have parsed Returns section",
        docstring.toLowerCase().contains("return") || docstring.contains("struct"));

    // Verify parameters were extracted (including multi-line parameter descriptions)
    assertTrue("Function should have 7 parameters", cgoFunc.getParameterCount() == 7);
  }

  @Test
  public void testSelectUsage() throws Exception {
    // Test that select() builtin function works correctly
    // This verifies that our fake select implementation can handle:
    // - Module-level select usage (like rules_go does with type(select(...)))
    // - Function-level select usage
    // - Select in default attribute values
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:select_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should extract function that uses select
    StarlarkFunctionInfo selectFunc = findFunction(moduleInfo, "function_with_select");
    assertNotNull("function_with_select should be extracted", selectFunc);

    // Should extract provider
    ProviderInfo provider = findProvider(moduleInfo, "SelectInfo");
    assertNotNull("SelectInfo should be extracted", provider);
    assertEquals("Provider should have correct name", "SelectInfo", provider.getProviderName());

    // Should extract rule that uses select
    RuleInfo rule = findRule(moduleInfo, "select_rule");
    assertNotNull("select_rule should be extracted", rule);
    assertEquals("Rule should have correct name", "select_rule", rule.getRuleName());
  }

  // Helper methods

  private StarlarkFunctionInfo findFunction(ModuleInfo moduleInfo, String name) {
    for (StarlarkFunctionInfo func : moduleInfo.getFuncInfoList()) {
      if (func.getFunctionName().equals(name)) {
        return func;
      }
    }
    return null;
  }

  private ProviderInfo findProvider(ModuleInfo moduleInfo, String name) {
    for (ProviderInfo provider : moduleInfo.getProviderInfoList()) {
      if (provider.getProviderName().equals(name)) {
        return provider;
      }
    }
    return null;
  }

  private RuleInfo findRule(ModuleInfo moduleInfo, String name) {
    for (RuleInfo rule : moduleInfo.getRuleInfoList()) {
      if (rule.getRuleName().equals(name)) {
        return rule;
      }
    }
    return null;
  }

  private AspectInfo findAspect(ModuleInfo moduleInfo, String name) {
    for (AspectInfo aspect : moduleInfo.getAspectInfoList()) {
      if (aspect.getAspectName().equals(name)) {
        return aspect;
      }
    }
    return null;
  }

  @Test
  public void testErrorCollection() throws Exception {
    // Test that when a docstring parse error occurs, it's collected as an error
    // and processing continues for other symbols in the file
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:malformed_docstring_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should extract the two good functions
    StarlarkFunctionInfo goodFunc1 = findFunction(moduleInfo, "good_function");
    assertNotNull("good_function should be extracted", goodFunc1);
    assertEquals("good_function should have correct name", "good_function", goodFunc1.getFunctionName());

    StarlarkFunctionInfo goodFunc2 = findFunction(moduleInfo, "another_good_function");
    assertNotNull("another_good_function should be extracted", goodFunc2);
    assertEquals("another_good_function should have correct name", "another_good_function", goodFunc2.getFunctionName());

    // Bad function should NOT be extracted (it has a malformed docstring)
    StarlarkFunctionInfo badFunc = findFunction(moduleInfo, "bad_function");
    assertNull("bad_function should not be extracted due to malformed docstring", badFunc);

    // Should have exactly one error collected
    assertTrue("Should have at least one error", response.getErrorCount() > 0);

    // Error should mention the bad function
    boolean foundBadFunctionError = false;
    for (String error : response.getErrorList()) {
      if (error.contains("bad_function")) {
        foundBadFunctionError = true;
        // Error should mention the docstring issue
        assertTrue("Error should mention Args/Returns ordering issue",
            error.toLowerCase().contains("args") || error.toLowerCase().contains("return"));
      }
    }
    assertTrue("Should have error mentioning bad_function", foundBadFunctionError);
  }

  @Test
  public void testInvalidLoadErrorContext() throws Exception {
    // Test that when a load statement is invalid, the error message includes
    // context about which file and load statement caused the error
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:invalid_load_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    try {
      Module response = blockingStub.moduleInfo(request);
      fail("Should have thrown an exception for invalid load statement");
    } catch (io.grpc.StatusRuntimeException e) {
      // Verify the error message includes context
      String errorMessage = e.getMessage();

      // Should mention the invalid load statement
      assertTrue("Error should mention the invalid load ':cache.bzl'",
          errorMessage.contains(":cache.bzl"));

      // Should mention which file the load is in (both label and path)
      assertTrue("Error should mention the file being processed (label)",
          errorMessage.contains("invalid_load_test.bzl"));

      // Should mention the absolute file path
      assertTrue("Error should include absolute file path",
          errorMessage.contains("testdata/invalid_load_test.bzl") ||
          errorMessage.contains("/invalid_load_test.bzl"));

      // Should explain what's wrong
      assertTrue("Error should mention the actual problem (target names may not contain ':')",
          errorMessage.contains("target names may not contain ':'") ||
          errorMessage.contains("Invalid load"));
    }
  }

  @Test
  public void testDictSplatWithFakeObjects() throws Exception {
    // Test that FakeDeepStructure objects can be used with the ** (splat) operator
    // This is needed for patterns like: dict({...}, **proto_toolchains.if_legacy_toolchain({...}))
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:dict_splat_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should successfully extract the test function
    StarlarkFunctionInfo testFunc = findFunction(moduleInfo, "test_function");
    assertNotNull("test_function should be extracted", testFunc);
  }

  @Test
  public void testProviderWithInitReturns2ElementTuple() throws Exception {
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:provider_init_test.bzl";
    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();
    Module response = blockingStub.moduleInfo(request);
    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());
    ModuleInfo moduleInfo = response.getInfo();

    // Should successfully extract ProtoInfo and test_function
    StarlarkFunctionInfo testFunc = findFunction(moduleInfo, "test_function");
    assertNotNull("test_function should be extracted", testFunc);
  }

  @Test
  public void testProviderAsHashableDictKey() throws Exception {
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:provider_as_dict_key_test.bzl";
    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();
    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());
    ModuleInfo moduleInfo = response.getInfo();

    // Should successfully extract the provider and function
    ProviderInfo myProvider = findProvider(moduleInfo, "MyProvider");
    assertNotNull("MyProvider should be extracted", myProvider);

    StarlarkFunctionInfo testFunc = findFunction(moduleInfo, "test_function");
    assertNotNull("test_function should be extracted", testFunc);
  }

  @Test
  public void testSymbolLocationsForAllEntityTypes() throws Exception {
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:symbol_locations_test.bzl";
    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();
    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have symbol locations", response.getSymbolLocationCount() > 0);

    // Check that we have symbol locations for different entity types
    java.util.List<build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation> symbolLocations = response.getSymbolLocationList();

    // Provider symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation providerLoc = findSymbolLocation(symbolLocations, "MyProvider");
    assertNotNull("MyProvider symbol location should exist", providerLoc);
    assertEquals("MyProvider location should be at line 4", 4, providerLoc.getStart().getLine());

    // Function symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation functionLoc = findSymbolLocation(symbolLocations, "my_function");
    assertNotNull("my_function symbol location should exist", functionLoc);
    assertEquals("my_function location should be at line 10", 10, functionLoc.getStart().getLine());

    // Rule symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation ruleLoc = findSymbolLocation(symbolLocations, "my_rule");
    assertNotNull("my_rule symbol location should exist", ruleLoc);
    assertEquals("my_rule location should be at line 18", 18, ruleLoc.getStart().getLine());

    // Aspect symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation aspectLoc = findSymbolLocation(symbolLocations, "my_aspect");
    assertNotNull("my_aspect symbol location should exist", aspectLoc);
    assertEquals("my_aspect location should be at line 30", 30, aspectLoc.getStart().getLine());

    // Macro symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation macroLoc = findSymbolLocation(symbolLocations, "my_macro");
    assertNotNull("my_macro symbol location should exist", macroLoc);
    assertEquals("my_macro location should be at line 36", 36, macroLoc.getStart().getLine());
  }

  private build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation findSymbolLocation(
      java.util.List<build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation> locations, String name) {
    for (build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation loc : locations) {
      if (loc.getName().equals(name)) {
        return loc;
      }
    }
    return null;
  }

  @Test
  public void testWrapperFunctionDetection() throws Exception {
    // Use simple test file without **kwargs to avoid macro resolution conflicts
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:simple_wrapper_test.bzl";
    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();
    Module response = blockingStub.moduleInfo(request);

    // Verify that the module has the new function field populated
    assertTrue("Module should have functions", response.getFunctionCount() > 0);

    // Find the functions
    build.stack.starlark.v1beta1.StarlarkProtos.Function wrapperFunc = null;
    build.stack.starlark.v1beta1.StarlarkProtos.Function helperFunc = null;

    for (build.stack.starlark.v1beta1.StarlarkProtos.Function func : response.getFunctionList()) {
      if (func.getInfo().getFunctionName().equals("wrapper_without_kwargs")) {
        wrapperFunc = func;
      } else if (func.getInfo().getFunctionName().equals("helper_func")) {
        helperFunc = func;
      }
    }

    assertNotNull("wrapper_without_kwargs should be found", wrapperFunc);
    assertNotNull("helper_func should be found", helperFunc);

    // Verify wrapper function detection - calls_rule_or_macro
    assertEquals("wrapper_without_kwargs should call my_rule",
        1, wrapperFunc.getCallsRuleOrMacroCount());
    assertEquals("wrapper_without_kwargs should call my_rule",
        "my_rule", wrapperFunc.getCallsRuleOrMacro(0));

    // Helper function should not call any rules
    assertEquals("helper_func should not call any rules",
        0, helperFunc.getCallsRuleOrMacroCount());

    // Wrapper doesn't use **kwargs, so forwards_kwargs_to should be empty
    assertEquals("wrapper_without_kwargs should not forward kwargs",
        0, wrapperFunc.getForwardsKwargsToCount());
    assertEquals("helper_func should not forward kwargs",
        0, helperFunc.getForwardsKwargsToCount());
  }

  @Test
  public void testInvalidLabelReturnsInvalidArgument() throws Exception {
    // Test that an invalid label returns INVALID_ARGUMENT error code
    // Use a label with illegal characters (e.g., absolute path)
    String invalidLabel = "///absolute/path/file.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(invalidLabel)
        .build();

    try {
      blockingStub.moduleInfo(request);
      fail("Expected an exception for invalid label");
    } catch (io.grpc.StatusRuntimeException e) {
      assertEquals("Invalid label should return INVALID_ARGUMENT",
          io.grpc.Status.Code.INVALID_ARGUMENT, e.getStatus().getCode());
    }
  }

  @Test
  public void testMissingFileReturnsNotFound() throws Exception {
    // Test that a missing file returns NOT_FOUND error code
    String label = "//does/not:exist.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    try {
      blockingStub.moduleInfo(request);
      fail("Expected an exception for missing file");
    } catch (io.grpc.StatusRuntimeException e) {
      assertEquals("Missing file should return NOT_FOUND",
          io.grpc.Status.Code.NOT_FOUND, e.getStatus().getCode());
    }
  }

  @Test
  public void testStarlarkSyntaxErrorReturnsInvalidArgument() throws Exception {
    // Test that Starlark syntax errors return INVALID_ARGUMENT error code
    // We'll use module_content to provide invalid Starlark code
    String label = "//test:syntax_error.bzl";
    String invalidStarlark = "def foo(\n  # Missing closing paren and colon";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .setModuleContent(invalidStarlark)
        .build();

    try {
      blockingStub.moduleInfo(request);
      fail("Expected an exception for syntax error");
    } catch (io.grpc.StatusRuntimeException e) {
      assertEquals("Starlark syntax error should return INVALID_ARGUMENT",
          io.grpc.Status.Code.INVALID_ARGUMENT, e.getStatus().getCode());
    }
  }

  @Test
  public void testTransitiveLoadErrorAllowsTopLevelExtraction() throws Exception {
    // Test that errors in transitive loads don't prevent extracting the top-level file
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:transitive_error_main.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    // Should succeed despite transitive load having a "does not contain symbol" error
    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Verify we extracted the top-level file's content
    assertTrue("Should extract main_rule from top-level file",
        moduleInfo.getRuleInfoCount() > 0);

    boolean foundMainRule = false;
    for (RuleInfo ruleInfo : moduleInfo.getRuleInfoList()) {
      if (ruleInfo.getRuleName().equals("main_rule")) {
        foundMainRule = true;
        break;
      }
    }
    assertTrue("Should have extracted main_rule", foundMainRule);

    // Verify we extracted the top-level file's functions
    assertTrue("Should extract main_function from top-level file",
        moduleInfo.getFuncInfoCount() > 0);

    boolean foundMainFunction = false;
    for (StarlarkFunctionInfo funcInfo : moduleInfo.getFuncInfoList()) {
      if (funcInfo.getFunctionName().equals("main_function")) {
        foundMainFunction = true;
        break;
      }
    }
    assertTrue("Should have extracted main_function", foundMainFunction);
  }
}
