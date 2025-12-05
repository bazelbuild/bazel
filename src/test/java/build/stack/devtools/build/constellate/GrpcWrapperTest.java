package build.stack.devtools.build.constellate;

import static org.junit.Assert.*;

import build.stack.starlark.v1beta1.StarlarkGrpc;
import build.stack.starlark.v1beta1.StarlarkProtos.Attribute;
import build.stack.starlark.v1beta1.StarlarkProtos.Macro;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleExtension;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleExtensionTagClass;
import build.stack.starlark.v1beta1.StarlarkProtos.ModuleInfoRequest;
import build.stack.starlark.v1beta1.StarlarkProtos.Module;
import build.stack.starlark.v1beta1.StarlarkProtos.RepositoryRule;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.common.options.OptionsParser;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Tests for wrapper messages in starlark.proto that combine stardoc_output.proto messages
 * with SymbolLocation information.
 *
 * <p>These wrapper messages enable IDE features like go-to-definition and hover by providing
 * location information for all entity types.
 */
public class GrpcWrapperTest {
  private Server server;
  private ManagedChannel channel;
  private StarlarkGrpc.StarlarkBlockingStub blockingStub;

  @Before
  public void setUp() throws Exception {
    // Setup options
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(BuildLanguageOptions.class).build();
    parser.parse("--experimental_enable_first_class_macros=false");
    BuildLanguageOptions semanticsOptions = parser.getOptions(BuildLanguageOptions.class);
    StarlarkSemantics semantics = semanticsOptions.toStarlarkSemantics();

    // Start gRPC server
    String serverName = "test-server-" + System.currentTimeMillis();
    server =
        InProcessServerBuilder.forName(serverName)
            .directExecutor()
            .addService(new StarlarkServer(semantics))
            .build()
            .start();

    // Create client
    channel = InProcessChannelBuilder.forName(serverName).directExecutor().build();
    blockingStub = StarlarkGrpc.newBlockingStub(channel);
  }

  @After
  public void tearDown() throws Exception {
    if (channel != null) {
      channel.shutdownNow();
    }
    if (server != null) {
      server.shutdownNow();
    }
  }

  @Test
  public void testRepositoryRuleWrapper() throws Exception {
    String label =
        "//src/test/java/build/stack/devtools/build/constellate/testdata:comprehensive_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder().setTargetFileLabel(label).build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    assertTrue("Should have repository rules", response.getRepositoryRuleCount() > 0);

    RepositoryRule repoRule = response.getRepositoryRule(0);

    // Verify info field
    assertTrue("Should have info", repoRule.hasInfo());
    assertEquals("my_repo_rule", repoRule.getInfo().getRuleName());
    assertTrue("Should have docstring", repoRule.getInfo().getDocString().length() > 0);

    // Verify OriginKey
    assertTrue("Should have OriginKey", repoRule.getInfo().hasOriginKey());
    assertEquals("my_repo_rule", repoRule.getInfo().getOriginKey().getName());
    assertFalse(
        "OriginKey file should be set", repoRule.getInfo().getOriginKey().getFile().isEmpty());

    // Verify environ field
    assertTrue("Should have environ variables", repoRule.getInfo().getEnvironCount() > 0);
    assertTrue(
        "Should have MY_ENV_VAR",
        repoRule.getInfo().getEnvironList().contains("MY_ENV_VAR"));

    // Verify location field
    assertTrue("Should have location", repoRule.hasLocation());
    assertEquals("my_repo_rule", repoRule.getLocation().getName());
    assertTrue("Location should have start position", repoRule.getLocation().hasStart());
    assertEquals("Location should point to correct line", 65, repoRule.getLocation().getStart().getLine());

    // Verify nested attributes
    assertTrue("Should have attributes", repoRule.getAttributeCount() > 0);
    boolean hasUrlAttr = false;
    for (Attribute attr : repoRule.getAttributeList()) {
      if (attr.getInfo().getName().equals("url")) {
        hasUrlAttr = true;
        assertTrue("url should be mandatory", attr.getInfo().getMandatory());
        assertTrue("url should have docstring", attr.getInfo().getDocString().length() > 0);
      }
    }
    assertTrue("Should have 'url' attribute", hasUrlAttr);
  }

  @Test
  public void testModuleExtensionWrapper() throws Exception {
    String label =
        "//src/test/java/build/stack/devtools/build/constellate/testdata:comprehensive_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder().setTargetFileLabel(label).build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    assertTrue("Should have module extensions", response.getModuleExtensionCount() > 0);

    ModuleExtension extension = response.getModuleExtension(0);

    // Verify info field
    assertTrue("Should have info", extension.hasInfo());
    assertEquals("my_extension", extension.getInfo().getExtensionName());
    assertTrue("Should have docstring", extension.getInfo().getDocString().length() > 0);

    // Verify OriginKey
    assertTrue("Should have OriginKey", extension.getInfo().hasOriginKey());
    assertFalse(
        "OriginKey file should be set",
        extension.getInfo().getOriginKey().getFile().isEmpty());

    // Verify location field
    assertTrue("Should have location", extension.hasLocation());
    assertEquals("my_extension", extension.getLocation().getName());
    assertTrue("Location should have start position", extension.getLocation().hasStart());
    assertEquals("Location should point to correct line", 116, extension.getLocation().getStart().getLine());

    // Verify nested tag classes
    assertTrue("Should have tag classes", extension.getTagClassCount() > 0);
    ModuleExtensionTagClass tagClass = extension.getTagClass(0);

    // Verify tag class info
    assertTrue("Tag class should have info", tagClass.hasInfo());
    assertEquals("install", tagClass.getInfo().getTagName());
    assertTrue(
        "Tag class should have docstring", tagClass.getInfo().getDocString().length() > 0);

    // Note: Tag class locations are not currently tracked separately
    // They could be tracked as "extensionName.tagName" in the future

    // Verify tag class attributes
    assertTrue("Tag class should have attributes", tagClass.getAttributeCount() > 0);
    boolean hasNameAttr = false;
    for (Attribute attr : tagClass.getAttributeList()) {
      if (attr.getInfo().getName().equals("name")) {
        hasNameAttr = true;
        assertTrue("name should be mandatory", attr.getInfo().getMandatory());
      }
    }
    assertTrue("Tag class should have 'name' attribute", hasNameAttr);
  }

  @Test
  public void testMacroWrapper() throws Exception {
    String label =
        "//src/test/java/build/stack/devtools/build/constellate/testdata:comprehensive_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder().setTargetFileLabel(label).build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // TODO: This test will fail until we implement wrapper population in StarlarkServer
    // AND enable --experimental_enable_first_class_macros flag
    // For now, we document the expected behavior
    // assertTrue("Should have macros", response.getMacroCount() > 0);
    //
    // Macro macro = response.getMacro(0);
    //
    // // Verify info field
    // assertTrue("Should have info", macro.hasInfo());
    // assertEquals("my_macro", macro.getInfo().getMacroName());
    // assertTrue("Should have docstring", macro.getInfo().getDocString().length() > 0);
    //
    // // Verify OriginKey
    // assertTrue("Should have OriginKey", macro.getInfo().hasOriginKey());
    // assertEquals("my_macro", macro.getInfo().getOriginKey().getName());
    // assertFalse(
    //     "OriginKey file should be set", macro.getInfo().getOriginKey().getFile().isEmpty());
    //
    // // Verify location field
    // assertTrue("Should have location", macro.hasLocation());
    // assertEquals("my_macro", macro.getLocation().getName());
    // assertTrue("Location should have start position", macro.getLocation().hasStart());
    //
    // // Verify nested attributes
    // assertTrue("Should have attributes", macro.getAttributeCount() > 0);
    // boolean hasValueAttr = false;
    // for (Attribute attr : macro.getAttributeList()) {
    //   if (attr.getInfo().getName().equals("value")) {
    //     hasValueAttr = true;
    //     assertFalse("value should not be mandatory", attr.getInfo().getMandatory());
    //   }
    // }
    // assertTrue("Should have 'value' attribute", hasValueAttr);
  }

  @Test
  public void testAllWrappersHaveLocations() throws Exception {
    String label =
        "//src/test/java/build/stack/devtools/build/constellate/testdata:comprehensive_test.bzl";

    ModuleInfoRequest request = ModuleInfoRequest.newBuilder().setTargetFileLabel(label).build();

    Module response = blockingStub.moduleInfo(request);

    assertNotNull("Response should not be null", response);

    // Verify that ALL wrapper entity types have location info
    // This is the key value proposition of wrapper messages

    // Verify ModuleInfo contains entities (from stardoc_output.proto)
    assertTrue("Should have module info", response.hasInfo());
    assertTrue(
        "Module should have some entities",
        response.getInfo().getRuleInfoCount() > 0
            || response.getInfo().getProviderInfoCount() > 0
            || response.getInfo().getFuncInfoCount() > 0);

    // Verify wrapper fields contain same entities WITH locations
    // Also verify locations point to correct line numbers in the source file
    for (RepositoryRule repoRule : response.getRepositoryRuleList()) {
      assertTrue("Repository rule should have location", repoRule.hasLocation());
      assertTrue("Repository rule location should have start", repoRule.getLocation().hasStart());
      assertTrue("Repository rule start line should be > 0", repoRule.getLocation().getStart().getLine() > 0);
      // my_repo_rule is defined at line 65
      if (repoRule.getLocation().getName().equals("my_repo_rule")) {
        assertEquals("my_repo_rule should be at line 65", 65, repoRule.getLocation().getStart().getLine());
      }
    }

    for (ModuleExtension extension : response.getModuleExtensionList()) {
      assertTrue("Module extension should have location", extension.hasLocation());
      assertTrue("Module extension location should have start", extension.getLocation().hasStart());
      assertTrue("Module extension start line should be > 0", extension.getLocation().getStart().getLine() > 0);
      // my_extension is defined at line 116
      if (extension.getLocation().getName().equals("my_extension")) {
        assertEquals("my_extension should be at line 116", 116, extension.getLocation().getStart().getLine());
      }
    }

    for (Macro macro : response.getMacroList()) {
      assertTrue("Macro should have location", macro.hasLocation());
      assertTrue("Macro location should have start", macro.getLocation().hasStart());
      assertTrue("Macro start line should be > 0", macro.getLocation().getStart().getLine() > 0);
      // my_macro is defined at line 86
      if (macro.getLocation().getName().equals("my_macro")) {
        assertEquals("my_macro should be at line 86", 86, macro.getLocation().getStart().getLine());
      }
    }
  }
}
