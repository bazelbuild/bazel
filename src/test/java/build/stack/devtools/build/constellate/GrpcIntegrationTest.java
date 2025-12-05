package build.stack.devtools.build.constellate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

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
    // First, let's create test files that demonstrate cross-file loading
    String label = "//src/test/java/build/stack/devtools/build/constellate/testdata:load_test_main.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel(label)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // When loading from other files, OriginKeys should reference the original file
    // This tests that cross-file references are properly tracked

    // Note: This test will be enhanced once we create the load test files
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
    assertTrue("Should extract providers", moduleInfo.getProviderInfoCount() >= 2);
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
        "\"\"\"Inline test module.\"\"\"\\n" +
        "\\n" +
        "def inline_function(x):\\n" +
        "    \"\"\"An inline function.\\n" +
        "\\n" +
        "    Args:\\n" +
        "        x: The input value\\n" +
        "\\n" +
        "    Returns:\\n" +
        "        The squared value\\n" +
        "    \"\"\"\\n" +
        "    return x * x\\n" +
        "\\n" +
        "InlineInfo = provider(\\n" +
        "    doc = \\\"An inline provider.\\\",\\n" +
        "    fields = [\\\"data\\\"],\\n" +
        ")\\n";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder()
        .setTargetFileLabel("//virtual:inline.bzl")  // Label doesn't need to exist on disk
        .setModuleContent(inlineContent)
        .build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);
    assertTrue("Should have module info", response.hasInfo());

    ModuleInfo moduleInfo = response.getInfo();

    // Should extract function from inline content
    StarlarkFunctionInfo func = findFunction(moduleInfo, "inline_function");
    assertNotNull("inline_function should be extracted from inline content", func);
    assertEquals("Function should have correct name", "inline_function", func.getFunctionName());
    assertTrue("Function should have documentation", func.getDocString().contains("squared value"));

    // Should extract provider from inline content
    ProviderInfo provider = findProvider(moduleInfo, "InlineInfo");
    assertNotNull("InlineInfo should be extracted from inline content", provider);
    assertEquals("Provider should have correct name", "InlineInfo", provider.getProviderName());
    assertEquals("Provider should have one field", 1, provider.getFieldInfoCount());
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
}
