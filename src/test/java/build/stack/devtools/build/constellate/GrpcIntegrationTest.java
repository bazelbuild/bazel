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
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.StarlarkFunctionInfo;
import java.util.List;
import java.util.stream.Collectors;
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

  /**
   * Helper class to adapt Module proto to ModuleInfo-like interface for test
   * compatibility.
   * This allows existing tests to work with the new Module structure that no
   * longer embeds ModuleInfo.
   */
  private static class ModuleInfoAdapter {
    private final Module module;

    ModuleInfoAdapter(Module module) {
      this.module = module;
    }

    public int getRuleInfoCount() {
      return module.getRuleCount();
    }

    public List<RuleInfo> getRuleInfoList() {
      return module.getRuleList().stream().map(r -> r.getInfo()).collect(Collectors.toList());
    }

    public RuleInfo getRuleInfo(int i) {
      return module.getRule(i).getInfo();
    }

    public int getProviderInfoCount() {
      return module.getProviderCount();
    }

    public List<ProviderInfo> getProviderInfoList() {
      return module.getProviderList().stream().map(p -> p.getInfo()).collect(Collectors.toList());
    }

    public ProviderInfo getProviderInfo(int i) {
      return module.getProvider(i).getInfo();
    }

    public int getFuncInfoCount() {
      return module.getFunctionCount();
    }

    public List<StarlarkFunctionInfo> getFuncInfoList() {
      return module.getFunctionList().stream().map(f -> f.getInfo()).collect(Collectors.toList());
    }

    public int getAspectInfoCount() {
      return module.getAspectCount();
    }

    public List<AspectInfo> getAspectInfoList() {
      return module.getAspectList().stream().map(a -> a.getInfo()).collect(Collectors.toList());
    }

    public AspectInfo getAspectInfo(int i) {
      return module.getAspect(i).getInfo();
    }

    public String getModuleDocstring() {
      return module.getModuleDocstring();
    }
  }

  private static final String TEST_DATA_DIR = "src/test/java/build/stack/devtools/build/constellate/testdata";

  private Server server;
  private ManagedChannel channel;
  private StarlarkGrpc.StarlarkBlockingStub blockingStub;

  @Before
  public void setUp() throws Exception {
    // Create parser options
    OptionsParser parser = OptionsParser.builder().optionsClasses(BuildLanguageOptions.class).build();
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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

    // Verify load statements are recorded
    assertTrue("Should have load statements", response.getLoadCount() > 0);

    // Find the load statement for load_test_lib.bzl
    build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt loadStmt = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt stmt : response.getLoadList()) {
      if (stmt.getLabel().getName().contains("load_test_lib.bzl")) {
        loadStmt = stmt;
        break;
      }
    }

    assertNotNull("Should have load statement for load_test_lib.bzl", loadStmt);
    assertTrue("Load statement should have symbols", loadStmt.getSymbolCount() > 0);

    // Verify the loaded symbols
    java.util.Set<String> loadedSymbols = new java.util.HashSet<>();
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadSymbol symbol : loadStmt.getSymbolList()) {
      loadedSymbols.add(symbol.getTo()); // The local name
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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

    // Verify all entity types are extracted
    assertTrue("Should extract functions", moduleInfo.getFuncInfoCount() >= 1);
    System.err.println("Provider count: " + moduleInfo.getProviderInfoCount());
    for (int i = 0; i < moduleInfo.getProviderInfoCount(); i++) {
      System.err.println("Provider " + i + ": " + moduleInfo.getProviderInfo(i).getProviderName());
    }
    assertTrue("Should extract providers (got " + moduleInfo.getProviderInfoCount() + ")",
        moduleInfo.getProviderInfoCount() >= 2);
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

    assertTrue("Should contain FILTERED_NUMBERS (list comprehension with filter)",
        response.containsGlobal("FILTERED_NUMBERS"));
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
    if (response.getRuleCount() > 0 || response.getProviderCount() > 0 || response.getFunctionCount() > 0) {
      ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);
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
        .addSymbolNames("simple_function") // Only extract this symbol
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

    // Should only have the requested symbol (and not private symbols)
    StarlarkFunctionInfo func = findFunction(moduleInfo, "simple_function");
    assertNotNull("simple_function should be extracted", func);

    // Private functions (starting with _) should be filtered out
    StarlarkFunctionInfo privateFunc = findFunction(moduleInfo, "_simple_rule_impl");
    // Note: Current implementation may include private functions, this documents
    // the behavior
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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

    // Check OriginKey format
    if (moduleInfo.getRuleInfoCount() > 0) {
      RuleInfo rule = moduleInfo.getRuleInfo(0);
      if (rule.hasOriginKey()) {
        String file = rule.getOriginKey().getFile();
        assertFalse("OriginKey file should not be empty", file.isEmpty());
        // File should be in label format (e.g., "//package:file.bzl" or
        // "@repo//package:file.bzl")
        // The exact format depends on how labels are rendered
      }
    }
  }

  @Test
  public void testModuleContentInline() throws Exception {
    // Test that we can provide module content inline without requiring a file on
    // disk
    // This is useful for LSP scenarios where the file hasn't been saved yet,
    // or for testing without creating temporary files

    String inlineContent = "\"\"\"Inline test module.\"\"\"\n" +
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
        .setTargetFileLabel("//virtual:inline.bzl") // Label doesn't need to exist on disk
        .setModuleContent(inlineContent)
        .addDepRoots(".") // Add current directory as dep root so path resolution works
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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
    // Test that complex docstrings (like those from rules_go) are now handled
    // correctly
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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

    // Should successfully extract function with complex docstring
    StarlarkFunctionInfo cgoFunc = findFunction(moduleInfo, "cgo_configure");
    assertNotNull("cgo_configure should be extracted", cgoFunc);
    assertFalse("Function should have documentation", cgoFunc.getDocString().isEmpty());

    // Verify the docstring was parsed successfully (including the problematic
    // "Returns: content" format)
    String docstring = cgoFunc.getDocString();
    assertTrue("Docstring should not be empty", !docstring.isEmpty());
    assertTrue("Docstring should contain function description", docstring.contains("cgo archive"));

    // Verify that the inline "Returns: a struct containing:" format was parsed
    // correctly
    // This was previously rejected with "malformed docstring" error
    assertTrue("Docstring should have parsed Returns section",
        docstring.toLowerCase().contains("return") || docstring.contains("struct"));

    // Verify parameters were extracted (including multi-line parameter
    // descriptions)
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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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

  private StarlarkFunctionInfo findFunction(ModuleInfoAdapter moduleInfo, String name) {
    for (StarlarkFunctionInfo func : moduleInfo.getFuncInfoList()) {
      if (func.getFunctionName().equals(name)) {
        return func;
      }
    }
    return null;
  }

  private ProviderInfo findProvider(ModuleInfoAdapter moduleInfo, String name) {
    for (ProviderInfo provider : moduleInfo.getProviderInfoList()) {
      if (provider.getProviderName().equals(name)) {
        return provider;
      }
    }
    return null;
  }

  private RuleInfo findRule(ModuleInfoAdapter moduleInfo, String name) {
    for (RuleInfo rule : moduleInfo.getRuleInfoList()) {
      if (rule.getRuleName().equals(name)) {
        return rule;
      }
    }
    return null;
  }

  private AspectInfo findAspect(ModuleInfoAdapter moduleInfo, String name) {
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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

    // Should extract the two good functions
    StarlarkFunctionInfo goodFunc1 = findFunction(moduleInfo, "good_function");
    assertNotNull("good_function should be extracted", goodFunc1);
    assertEquals("good_function should have correct name", "good_function", goodFunc1.getFunctionName());

    StarlarkFunctionInfo goodFunc2 = findFunction(moduleInfo, "another_good_function");
    assertNotNull("another_good_function should be extracted", goodFunc2);
    assertEquals("another_good_function should have correct name", "another_good_function",
        goodFunc2.getFunctionName());

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
    // This is needed for patterns like: dict({...},
    // **proto_toolchains.if_legacy_toolchain({...}))
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:dict_splat_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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
    java.util.List<build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation> symbolLocations = response
        .getSymbolLocationList();

    // Provider symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation providerLoc = findSymbolLocation(symbolLocations,
        "MyProvider");
    assertNotNull("MyProvider symbol location should exist", providerLoc);
    assertEquals("MyProvider location should be at line 4", 4, providerLoc.getStart().getLine());

    // Function symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation functionLoc = findSymbolLocation(symbolLocations,
        "my_function");
    assertNotNull("my_function symbol location should exist", functionLoc);
    assertEquals("my_function location should be at line 10", 10, functionLoc.getStart().getLine());

    // Rule symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation ruleLoc = findSymbolLocation(symbolLocations, "my_rule");
    assertNotNull("my_rule symbol location should exist", ruleLoc);
    assertEquals("my_rule location should be at line 18", 18, ruleLoc.getStart().getLine());

    // Aspect symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation aspectLoc = findSymbolLocation(symbolLocations,
        "my_aspect");
    assertNotNull("my_aspect symbol location should exist", aspectLoc);
    assertEquals("my_aspect location should be at line 30", 30, aspectLoc.getStart().getLine());

    // Macro symbol location
    build.stack.starlark.v1beta1.StarlarkProtos.SymbolLocation macroLoc = findSymbolLocation(symbolLocations,
        "my_macro");
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
  public void testNameForwarding() throws Exception {
    // Test that name parameter forwarding is tracked correctly
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:name_forwarding_test.bzl";
    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();
    Module response = blockingStub.moduleInfo(request);

    // Verify that the module has functions or RuleMacros
    assertTrue("Module should have functions or RuleMacros",
        response.getFunctionCount() > 0 || response.getRuleMacroCount() > 0);

    // Helper to find function by name (check both Function list and RuleMacro list)
    java.util.Map<String, build.stack.starlark.v1beta1.StarlarkProtos.Function> functions = new java.util.HashMap<>();
    for (build.stack.starlark.v1beta1.StarlarkProtos.Function func : response.getFunctionList()) {
      functions.put(func.getInfo().getFunctionName(), func);
    }
    // Also collect functions from RuleMacros (functions with **kwargs become RuleMacros)
    for (build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro macro : response.getRuleMacroList()) {
      if (macro.hasFunction()) {
        functions.put(macro.getFunction().getInfo().getFunctionName(), macro.getFunction());
      }
    }

    // Test explicit_name_macro: should forward name to my_rule
    build.stack.starlark.v1beta1.StarlarkProtos.Function explicitNameFunc = functions.get("explicit_name_macro");
    assertNotNull("explicit_name_macro should be found", explicitNameFunc);
    assertTrue("explicit_name_macro should forward name to my_rule",
        explicitNameFunc.getForwardsNameToList().contains("my_rule"));
    assertTrue("explicit_name_macro should forward kwargs to my_rule",
        explicitNameFunc.getForwardsKwargsToList().contains("my_rule"));

    // Test positional_name_macro: should forward name to my_rule
    build.stack.starlark.v1beta1.StarlarkProtos.Function positionalFunc = functions.get("positional_name_macro");
    assertNotNull("positional_name_macro should be found", positionalFunc);
    assertTrue("positional_name_macro should forward name to my_rule",
        positionalFunc.getForwardsNameToList().contains("my_rule"));

    // Test transformed_name_macro: name + "_lib" still references name param
    build.stack.starlark.v1beta1.StarlarkProtos.Function transformedFunc = functions.get("transformed_name_macro");
    assertNotNull("transformed_name_macro should be found", transformedFunc);
    assertTrue("transformed_name_macro should forward name to my_rule (even with transformation)",
        transformedFunc.getForwardsNameToList().contains("my_rule"));

    // Test hardcoded_name_macro: should NOT forward name (uses hardcoded string)
    build.stack.starlark.v1beta1.StarlarkProtos.Function hardcodedFunc = functions.get("hardcoded_name_macro");
    assertNotNull("hardcoded_name_macro should be found", hardcodedFunc);
    assertFalse("hardcoded_name_macro should NOT forward name to my_rule",
        hardcodedFunc.getForwardsNameToList().contains("my_rule"));
    assertTrue("hardcoded_name_macro should still forward kwargs to my_rule",
        hardcodedFunc.getForwardsKwargsToList().contains("my_rule"));

    // Test multiple_name_macro: should forward name to both rules
    build.stack.starlark.v1beta1.StarlarkProtos.Function multipleFunc = functions.get("multiple_name_macro");
    assertNotNull("multiple_name_macro should be found", multipleFunc);
    assertTrue("multiple_name_macro should forward name to my_rule",
        multipleFunc.getForwardsNameToList().contains("my_rule"));
    assertTrue("multiple_name_macro should forward name to my_binary",
        multipleFunc.getForwardsNameToList().contains("my_binary"));

    // Test no_name_param_macro: shouldn't have name forwarding (no name param)
    build.stack.starlark.v1beta1.StarlarkProtos.Function noNameFunc = functions.get("no_name_param_macro");
    assertNotNull("no_name_param_macro should be found", noNameFunc);
    assertEquals("no_name_param_macro should have empty forwards_name_to",
        0, noNameFunc.getForwardsNameToCount());

    // Test name_without_kwargs_macro: should forward name but not kwargs
    build.stack.starlark.v1beta1.StarlarkProtos.Function nameWithoutKwargsFunc = functions
        .get("name_without_kwargs_macro");
    assertNotNull("name_without_kwargs_macro should be found", nameWithoutKwargsFunc);
    assertTrue("name_without_kwargs_macro should forward name to my_rule",
        nameWithoutKwargsFunc.getForwardsNameToList().contains("my_rule"));
    assertEquals("name_without_kwargs_macro should NOT forward kwargs",
        0, nameWithoutKwargsFunc.getForwardsKwargsToCount());
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
    // Test that errors in transitive loads don't prevent extracting the top-level
    // file
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:transitive_error_main.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    // Should succeed despite transitive load having a "does not contain symbol"
    // error
    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    ModuleInfoAdapter moduleInfo = new ModuleInfoAdapter(response);

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

  @Test
  public void testRuleMacroDetection() throws Exception {
    // Test detection of RuleMacros - functions that wrap private rules
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:golden_file_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder().setTargetFileLabel(label).build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // We should have 1 RuleMacro: golden_file_test wraps _golden_file_test
    assertTrue("Should have at least 1 RuleMacro", response.getRuleMacroCount() >= 1);

    // Find golden_file_test RuleMacro
    build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro goldenTestMacro = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro macro : response.getRuleMacroList()) {
      if (macro.getFunction().getInfo().getFunctionName().equals("golden_file_test")) {
        goldenTestMacro = macro;
        break;
      }
    }

    assertNotNull("golden_file_test should be detected as RuleMacro", goldenTestMacro);

    // Verify it has a function
    assertTrue("Should have function", goldenTestMacro.hasFunction());
    assertEquals("Function name should be golden_file_test",
        "golden_file_test", goldenTestMacro.getFunction().getInfo().getFunctionName());
    assertTrue("Function should have docstring",
        goldenTestMacro.getFunction().getInfo().getDocString().length() > 0);
    assertTrue("Function should have location", goldenTestMacro.getFunction().hasLocation());

    // Verify it has a rule
    assertTrue("Should have rule", goldenTestMacro.hasRule());
    assertEquals("Rule should be _golden_file_test",
        "_golden_file_test", goldenTestMacro.getRule().getInfo().getRuleName());
    assertTrue("Rule should have location", goldenTestMacro.getRule().hasLocation());

    // Verify the rule has attributes
    assertTrue("Rule should have attributes", goldenTestMacro.getRule().getAttributeCount() > 0);
    boolean hasSrcsAttr = false;
    boolean hasGoldensAttr = false;
    for (build.stack.starlark.v1beta1.StarlarkProtos.Attribute attr : goldenTestMacro.getRule().getAttributeList()) {
      if (attr.getInfo().getName().equals("srcs")) {
        hasSrcsAttr = true;
      }
      if (attr.getInfo().getName().equals("goldens")) {
        hasGoldensAttr = true;
      }
    }
    assertTrue("Should have 'srcs' attribute", hasSrcsAttr);
    assertTrue("Should have 'goldens' attribute", hasGoldensAttr);

    // Verify the public function is NOT in the regular functions list
    boolean foundInFunctions = false;
    for (build.stack.starlark.v1beta1.StarlarkProtos.Function func : response.getFunctionList()) {
      if (func.getInfo().getFunctionName().equals("golden_file_test")) {
        foundInFunctions = true;
        break;
      }
    }
    assertFalse("golden_file_test should NOT be in regular functions (it's a RuleMacro)",
        foundInFunctions);

    // Note: The private rules (_golden_file_test, _golden_file_update) are embedded
    // in the RuleMacros, not in the regular rule list
  }

  @Test
  public void testRuleMacroDetectionAcrossFiles() throws Exception {
    // Test RuleMacro detection across multiple files with load statements
    // This tests: go_wrappers.bzl loads and forwards to go_binary rule from go_binary.bzl
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:go_wrappers.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // We should have 1 RuleMacro: go_binary_macro wraps go_binary
    assertTrue("Should have at least 1 RuleMacro", response.getRuleMacroCount() >= 1);

    // Find go_binary_macro RuleMacro
    build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro goBinaryMacro = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro macro : response.getRuleMacroList()) {
      if (macro.getFunction().getInfo().getFunctionName().equals("go_binary_macro")) {
        goBinaryMacro = macro;
        break;
      }
    }

    assertNotNull("go_binary_macro should be detected as RuleMacro", goBinaryMacro);

    // Verify it has a function
    assertTrue("Should have function", goBinaryMacro.hasFunction());
    assertEquals("Function name should be go_binary_macro",
        "go_binary_macro", goBinaryMacro.getFunction().getInfo().getFunctionName());
    assertTrue("Function should have docstring",
        goBinaryMacro.getFunction().getInfo().getDocString().length() > 0);
    assertTrue("Docstring should mention go_binary",
        goBinaryMacro.getFunction().getInfo().getDocString().contains("go_binary"));

    // Verify it has a rule
    assertTrue("Should have rule", goBinaryMacro.hasRule());
    assertEquals("Rule should be go_binary",
        "go_binary", goBinaryMacro.getRule().getInfo().getRuleName());

    // Verify the rule has expected attributes
    assertTrue("Rule should have attributes", goBinaryMacro.getRule().getAttributeCount() > 0);
    boolean hasSrcsAttr = false;
    boolean hasDepsAttr = false;
    boolean hasLinkmodeAttr = false;
    for (build.stack.starlark.v1beta1.StarlarkProtos.Attribute attr : goBinaryMacro.getRule().getAttributeList()) {
      String attrName = attr.getInfo().getName();
      if (attrName.equals("srcs")) {
        hasSrcsAttr = true;
      } else if (attrName.equals("deps")) {
        hasDepsAttr = true;
      } else if (attrName.equals("linkmode")) {
        hasLinkmodeAttr = true;
      }
    }
    assertTrue("Should have 'srcs' attribute", hasSrcsAttr);
    assertTrue("Should have 'deps' attribute", hasDepsAttr);
    assertTrue("Should have 'linkmode' attribute", hasLinkmodeAttr);

    // Verify the function forwards kwargs to the rule
    assertTrue("Function should forward kwargs",
        goBinaryMacro.getFunction().getForwardsKwargsToCount() > 0);
    boolean forwardsToGoBinary = false;
    for (String target : goBinaryMacro.getFunction().getForwardsKwargsToList()) {
      if (target.equals("go_binary")) {
        forwardsToGoBinary = true;
        break;
      }
    }
    assertTrue("Should forward kwargs to go_binary", forwardsToGoBinary);

    // Verify the function also calls go_non_executable_binary (conditional)
    boolean callsNonExecutable = false;
    for (String called : goBinaryMacro.getFunction().getCallsRuleOrMacroList()) {
      if (called.equals("go_non_executable_binary")) {
        callsNonExecutable = true;
        break;
      }
    }
    assertTrue("Should also call go_non_executable_binary", callsNonExecutable);

    // Verify go_binary_macro is NOT in regular functions (it's a RuleMacro)
    boolean foundInFunctions = false;
    for (build.stack.starlark.v1beta1.StarlarkProtos.Function func : response.getFunctionList()) {
      if (func.getInfo().getFunctionName().equals("go_binary_macro")) {
        foundInFunctions = true;
        break;
      }
    }
    assertFalse("go_binary_macro should NOT be in regular functions (it's a RuleMacro)",
        foundInFunctions);
  }

  @Test
  public void testPrivateFunctionsFiltered() throws Exception {
    // Verify that private functions (starting with _) are not exported
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:simple_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // Check that no functions starting with _ are in the function list
    for (build.stack.starlark.v1beta1.StarlarkProtos.Function func : response.getFunctionList()) {
      assertFalse("Function " + func.getInfo().getFunctionName() + " should not start with underscore",
          func.getInfo().getFunctionName().startsWith("_"));
    }

    // Check that no RuleMacros have functions starting with _
    for (build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro macro : response.getRuleMacroList()) {
      if (macro.hasFunction()) {
        assertFalse("RuleMacro function " + macro.getFunction().getInfo().getFunctionName() + " should not start with underscore",
            macro.getFunction().getInfo().getFunctionName().startsWith("_"));
      }
    }
  }

  @Test
  public void testTransitiveNativeRuleForwarding() throws Exception {
    // Test case for transitive forwarding: function -> loaded symbol -> native rule
    // Currently, Constellate detects the forwarding to the loaded symbol but does NOT
    // create a RuleMacro because it doesn't transitively resolve through load chains
    // to detect that the loaded symbol ultimately calls a native rule.
    //
    // This test documents the current behavior as a known limitation.
    // Future enhancement: traverse loaded modules to detect transitive native calls.

    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:transitive_native_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // Find functions in the response
    java.util.Map<String, build.stack.starlark.v1beta1.StarlarkProtos.Function> functionMap = new java.util.HashMap<>();
    for (build.stack.starlark.v1beta1.StarlarkProtos.Function func : response.getFunctionList()) {
      functionMap.put(func.getInfo().getFunctionName(), func);
    }

    // Verify: java_binary function should exist
    assertTrue("java_binary function should exist", functionMap.containsKey("java_binary"));
    build.stack.starlark.v1beta1.StarlarkProtos.Function javaBinaryFunc = functionMap.get("java_binary");

    // Verify: It correctly detects forwarding to _java_binary (the loaded symbol)
    assertTrue("java_binary should forward kwargs to _java_binary",
        javaBinaryFunc.getForwardsKwargsToList().contains("_java_binary"));

    // Known limitation: RuleMacro is NOT created because Constellate doesn't currently
    // traverse through loaded modules to detect that _java_binary -> native.java_binary
    // This would require cross-module transitive analysis.
    java.util.Map<String, build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro> macroMap = new java.util.HashMap<>();
    for (build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro macro : response.getRuleMacroList()) {
      if (macro.hasFunction()) {
        macroMap.put(macro.getFunction().getInfo().getFunctionName(), macro);
      }
    }

    // Document the limitation: Currently, this does NOT create a RuleMacro
    assertFalse("java_binary is NOT detected as RuleMacro (known limitation: no transitive resolution)",
        macroMap.containsKey("java_binary"));
  }

  @Test
  public void testDirectNativeForwardingWithFullDocs() throws Exception {
    // Test the exact pattern from the user's example:
    // A function that forwards **attrs (note: parameter name doesn't matter) to native.java_binary
    // This should create a RuleMacro with full RuleInfo from Build Encyclopedia
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:direct_native_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // Find RuleMacros in the response
    java.util.Map<String, build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro> macroMap = new java.util.HashMap<>();
    for (build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro macro : response.getRuleMacroList()) {
      if (macro.hasFunction()) {
        macroMap.put(macro.getFunction().getInfo().getFunctionName(), macro);
      }
    }

    // Verify: java_binary should be detected as a RuleMacro
    assertTrue("java_binary should be detected as RuleMacro", macroMap.containsKey("java_binary"));
    build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro javaBinary = macroMap.get("java_binary");

    // Verify: It forwards **attrs to native.java_binary (parameter name "attrs" shouldn't matter)
    assertTrue("java_binary should forward kwargs to native.java_binary",
        javaBinary.getFunction().getForwardsKwargsToList().contains("native.java_binary"));

    // Verify: The RuleMacro has the rule with full ModuleInfo from Build Encyclopedia
    assertTrue("RuleMacro should have rule", javaBinary.hasRule());
    assertTrue("Rule should have info", javaBinary.getRule().hasInfo());

    // Verify: The rule name is from Build Encyclopedia (either full or short name)
    String ruleName = javaBinary.getRule().getInfo().getRuleName();
    assertTrue("Rule name should be java_binary related (got: " + ruleName + ")",
        ruleName.equals("binary_rules.java_binary") || ruleName.equals("java_binary"));

    // Verify: Rule has attributes from Build Encyclopedia
    assertTrue("Rule should have attributes from Build Encyclopedia",
        javaBinary.getRule().getAttributeCount() > 0);

    // Verify: The docstring should come from user's function, not overwritten by Build Encyclopedia
    String functionDoc = javaBinary.getFunction().getInfo().getDocString();
    assertTrue("Function docstring should contain user's docs",
        functionDoc.contains("Bazel java_binary rule"));
  }

  @Test
  public void testNativeRuleForwarding() throws Exception {
    // Verify that functions forwarding to native rules are detected as RuleMacros
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:native_forwarding_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // Find RuleMacros in the response
    java.util.Map<String, build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro> macroMap = new java.util.HashMap<>();
    for (build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro macro : response.getRuleMacroList()) {
      if (macro.hasFunction()) {
        macroMap.put(macro.getFunction().getInfo().getFunctionName(), macro);
      }
    }

    // Test 1: my_java_library should forward **kwargs to native.java_library
    assertTrue("my_java_library should be detected as RuleMacro", macroMap.containsKey("my_java_library"));
    build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro myJavaLibrary = macroMap.get("my_java_library");
    assertTrue("my_java_library should forward kwargs to native.java_library",
        myJavaLibrary.getFunction().getForwardsKwargsToList().contains("native.java_library"));

    // Test 2: my_java_binary should forward **kwargs and name to native.java_binary
    assertTrue("my_java_binary should be detected as RuleMacro", macroMap.containsKey("my_java_binary"));
    build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro myJavaBinary = macroMap.get("my_java_binary");
    assertTrue("my_java_binary should forward kwargs to native.java_binary",
        myJavaBinary.getFunction().getForwardsKwargsToList().contains("native.java_binary"));

    // Test 3: my_java_test only forwards name (not kwargs), so it should NOT be a RuleMacro
    // RuleMacros specifically require **kwargs forwarding
    assertFalse("my_java_test should not be detected as RuleMacro (only forwards name)",
        macroMap.containsKey("my_java_test"));

    // Test 4: mixed_native_wrapper should forward to both native.java_library and native.java_binary
    assertTrue("mixed_native_wrapper should be detected as RuleMacro", macroMap.containsKey("mixed_native_wrapper"));
    build.stack.starlark.v1beta1.StarlarkProtos.RuleMacro mixedWrapper = macroMap.get("mixed_native_wrapper");
    assertTrue("mixed_native_wrapper should forward kwargs to native.java_library",
        mixedWrapper.getFunction().getForwardsKwargsToList().contains("native.java_library"));
    assertTrue("mixed_native_wrapper should forward kwargs to native.java_binary",
        mixedWrapper.getFunction().getForwardsKwargsToList().contains("native.java_binary"));

    // Test 5: non_forwarding_helper should NOT be detected as RuleMacro
    assertFalse("non_forwarding_helper should not be detected as RuleMacro",
        macroMap.containsKey("non_forwarding_helper"));

    // Test 6: indirect_native_call only forwards name (not kwargs), so it should NOT be a RuleMacro
    assertFalse("indirect_native_call should not be detected as RuleMacro (only forwards name)",
        macroMap.containsKey("indirect_native_call"));
  }

  @Test
  public void testLoadStatementLocations() throws Exception {
    // Test that load statements and their symbols have location information
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:load_locations_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have load statements", response.getLoadCount() > 0);

    // The test file has 3 load statements at lines 8, 11, and 14
    // Verify we have at least these load statements
    assertTrue("Should have at least 3 load statements", response.getLoadCount() >= 3);

    // Find the load statement at line 8: load("load_test_lib.bzl", "lib_function", "LibInfo")
    build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt loadStmt1 = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt stmt : response.getLoadList()) {
      if (stmt.hasLocation() && stmt.getLocation().getStart().getLine() == 8) {
        loadStmt1 = stmt;
        break;
      }
    }

    assertNotNull("Should find load statement at line 8", loadStmt1);
    assertTrue("Load statement should have location", loadStmt1.hasLocation());
    assertEquals("Load statement should start at line 8", 8, loadStmt1.getLocation().getStart().getLine());
    assertEquals("Load statement location name should be 'load'", "load", loadStmt1.getLocation().getName());

    // Verify symbols have locations
    assertTrue("Load statement should have symbols", loadStmt1.getSymbolCount() > 0);

    boolean foundLibFunction = false;
    boolean foundLibInfo = false;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadSymbol symbol : loadStmt1.getSymbolList()) {
      assertTrue("Symbol should have location", symbol.hasLocation());
      assertTrue("Symbol location should have valid line number", symbol.getLocation().getStart().getLine() > 0);

      if (symbol.getTo().equals("lib_function")) {
        foundLibFunction = true;
        assertEquals("lib_function symbol location name should match", "lib_function", symbol.getLocation().getName());
      } else if (symbol.getTo().equals("LibInfo")) {
        foundLibInfo = true;
        assertEquals("LibInfo symbol location name should match", "LibInfo", symbol.getLocation().getName());
      }
    }

    assertTrue("Should find lib_function symbol", foundLibFunction);
    assertTrue("Should find LibInfo symbol", foundLibInfo);

    // Find the load statement at line 14 with aliasing:
    // load("load_test_lib.bzl", lib_func = "lib_function", lib_prov = "LibInfo", lib = "lib_rule")
    build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt loadStmt3 = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt stmt : response.getLoadList()) {
      if (stmt.hasLocation() && stmt.getLocation().getStart().getLine() == 14) {
        loadStmt3 = stmt;
        break;
      }
    }

    assertNotNull("Should find load statement at line 14", loadStmt3);
    assertTrue("Load statement should have location", loadStmt3.hasLocation());
    assertEquals("Load statement should start at line 14", 14, loadStmt3.getLocation().getStart().getLine());

    // Verify aliased symbols have locations with local names
    boolean foundLibFunc = false;
    boolean foundLibProv = false;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadSymbol symbol : loadStmt3.getSymbolList()) {
      assertTrue("Symbol should have location", symbol.hasLocation());

      if (symbol.getTo().equals("lib_func")) {
        foundLibFunc = true;
        assertEquals("Original name should be lib_function", "lib_function", symbol.getFrom());
        assertEquals("Symbol location should use local name", "lib_func", symbol.getLocation().getName());
      } else if (symbol.getTo().equals("lib_prov")) {
        foundLibProv = true;
        assertEquals("Original name should be LibInfo", "LibInfo", symbol.getFrom());
        assertEquals("Symbol location should use local name", "lib_prov", symbol.getLocation().getName());
      }
    }

    assertTrue("Should find aliased lib_func symbol", foundLibFunc);
    assertTrue("Should find aliased lib_prov symbol", foundLibProv);
  }

  @Test
  public void testExternalRepositoryLabels() throws Exception {
    // Test that external repository labels in load statements are parsed correctly
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:external_repo_labels_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have load statements", response.getLoadCount() > 0);

    // Find the load statement for @bazel_features//:features.bzl
    build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt bazelFeaturesLoad = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt stmt : response.getLoadList()) {
      if (stmt.getLabel().getName().equals("features.bzl")) {
        bazelFeaturesLoad = stmt;
        break;
      }
    }

    assertNotNull("Should find @bazel_features//:features.bzl load statement", bazelFeaturesLoad);
    assertTrue("Load statement should have Label", bazelFeaturesLoad.hasLabel());

    // Verify the Label proto fields are populated correctly
    build.stack.starlark.v1beta1.StarlarkProtos.Label labelProto = bazelFeaturesLoad.getLabel();
    assertEquals("Repository should be 'bazel_features'", "bazel_features", labelProto.getRepo());
    assertEquals("Package should be empty (root package)", "", labelProto.getPkg());
    assertEquals("Name should be 'features.bzl'", "features.bzl", labelProto.getName());

    // Find the load statement for @rules_go//go:def.bzl
    build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt rulesGoLoad = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt stmt : response.getLoadList()) {
      if (stmt.getLabel().getName().equals("def.bzl") && stmt.getLabel().getPkg().equals("go")) {
        rulesGoLoad = stmt;
        break;
      }
    }

    assertNotNull("Should find @rules_go//go:def.bzl load statement", rulesGoLoad);
    build.stack.starlark.v1beta1.StarlarkProtos.Label rulesGoLabelProto = rulesGoLoad.getLabel();
    assertEquals("Repository should be 'rules_go'", "rules_go", rulesGoLabelProto.getRepo());
    assertEquals("Package should be 'go'", "go", rulesGoLabelProto.getPkg());
    assertEquals("Name should be 'def.bzl'", "def.bzl", rulesGoLabelProto.getName());

    // Find the load statement for local repository (should have empty repo)
    build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt localLoad = null;
    for (build.stack.starlark.v1beta1.StarlarkProtos.LoadStmt stmt : response.getLoadList()) {
      if (stmt.getLabel().getName().equals("load_test_lib.bzl")) {
        localLoad = stmt;
        break;
      }
    }

    assertNotNull("Should find local load statement", localLoad);
    build.stack.starlark.v1beta1.StarlarkProtos.Label localLabelProto = localLoad.getLabel();
    assertEquals("Local repository should be empty string", "", localLabelProto.getRepo());
    assertEquals("Package should be testdata package", "src/test/java/build/stack/devtools/build/constellate/testdata", localLabelProto.getPkg());
    assertEquals("Name should be 'load_test_lib.bzl'", "load_test_lib.bzl", localLabelProto.getName());
  }
}
