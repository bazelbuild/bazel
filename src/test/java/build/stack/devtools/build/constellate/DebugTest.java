package build.stack.devtools.build.constellate;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.AspectInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.MacroInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleExtensionInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ProviderInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RepositoryRuleInfo;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import com.google.devtools.common.options.OptionsParser;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.ParserInput;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Debug test to understand module globals. */
@RunWith(JUnit4.class)
public class DebugTest {

  private static final String TEST_DATA_DIR = "src/test/java/build/stack/devtools/build/constellate/testdata";

  @Test
  public void debugModuleGlobals() throws Exception {
    // Read the test file
    Path testFile = Paths.get(TEST_DATA_DIR, "simple_test.bzl");

    // Create parser options
    OptionsParser parser = OptionsParser.builder().optionsClasses(BuildLanguageOptions.class).build();
    BuildLanguageOptions semanticsOptions = parser.getOptions(BuildLanguageOptions.class);
    StarlarkSemantics semantics = semanticsOptions.toStarlarkSemantics();

    // Create constellate
    Constellate constellate = new Constellate(
        semantics,
        new FilesystemFileAccessor(),
        /* depRoots= */ ImmutableList.of());

    // Create label for the test file
    Label label = Label.parseCanonicalUnchecked("//test:simple_test.bzl");

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
    ParserInput input = ParserInput.fromLatin1(Files.readAllBytes(testFile), "simple_test.bzl");

    build.stack.starlark.v1beta1.StarlarkProtos.Module.Builder moduleBuilder = build.stack.starlark.v1beta1.StarlarkProtos.Module
        .newBuilder();

    Module module = constellate.eval(
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

    // Print module globals
    System.out.println("\n=== Module Globals ===");
    for (var entry : module.getGlobals().entrySet()) {
      Object value = entry.getValue();
      System.out.println(entry.getKey() + " = " + value.getClass().getSimpleName() + " : " + value);
    }

    System.out.println("\n=== Provider Info Map ===");
    for (var entry : providerInfoMap.build().entrySet()) {
      System.out.println(entry.getKey() + " = " + entry.getValue().getProviderName());
    }

    System.out.println("\n=== Rule Info Map ===");
    for (var entry : ruleInfoMap.build().entrySet()) {
      System.out.println(entry.getKey() + " = " + entry.getValue().getRuleName());
    }
  }
}
