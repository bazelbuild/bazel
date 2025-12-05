package build.stack.devtools.build.constellate;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.ModuleInfo;
import com.google.devtools.common.options.OptionsParser;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;

/**
 * Command-line interface for constellate tool.
 *
 * Evaluates a Starlark file and extracts documentation as a ModuleInfo protobuf.
 */
public final class ConstellateCli {

  /** CLI options for constellate. */
  public static class Options {
    public String input;
    public String output;

    public static Options parse(String[] args) {
      Options opts = new Options();
      for (String arg : args) {
        if (arg.startsWith("--input=")) {
          opts.input = arg.substring("--input=".length());
        } else if (arg.startsWith("--output=")) {
          opts.output = arg.substring("--output=".length());
        }
      }
      return opts;
    }
  }

  public static void main(String[] args) throws Exception {
    Options opts = Options.parse(args);

    if (opts.input == null || opts.output == null) {
      System.err.println("Usage: constellate --input=<file.bzl> --output=<output.pb>");
      System.exit(1);
    }

    // Check input file exists
    Path inputPath = Paths.get(opts.input);
    if (!Files.exists(inputPath)) {
      System.err.println("Error: Input file does not exist: " + opts.input);
      System.exit(1);
    }

    // Create a fake label for the input file
    String labelString = "//" + inputPath.getFileName().toString().replace(".bzl", "");
    Label label;
    try {
      label = Label.parseCanonical(labelString);
    } catch (LabelSyntaxException e) {
      // Fallback to a simple label
      label = Label.parseCanonicalUnchecked("//test:test.bzl");
    }

    // Create default options
    OptionsParser parser = OptionsParser.builder()
        .optionsClasses(BuildLanguageOptions.class)
        .build();
    BuildLanguageOptions semanticsOptions = parser.getOptions(BuildLanguageOptions.class);

    // Create file accessor that reads from filesystem
    StarlarkFileAccessor fileAccessor = new FilesystemFileAccessor(inputPath.getParent());

    // Run constellate
    System.out.println("Evaluating: " + opts.input);
    Constellate constellate = new Constellate(
        semanticsOptions.toStarlarkSemantics(),
        fileAccessor);

    ModuleInfo moduleInfo = constellate.run(label, Collections.emptyMap());

    // Write output
    try (FileOutputStream fos = new FileOutputStream(opts.output)) {
      moduleInfo.writeTo(fos);
      System.out.println("Output written to: " + opts.output);

      // Print summary
      System.out.println("\nExtracted:");
      System.out.println("  Rules: " + moduleInfo.getRuleInfoCount());
      System.out.println("  Providers: " + moduleInfo.getProviderInfoCount());
      System.out.println("  Functions: " + moduleInfo.getFuncInfoCount());
      System.out.println("  Aspects: " + moduleInfo.getAspectInfoCount());
      System.out.println("  Macros: " + moduleInfo.getMacroInfoCount());
      System.out.println("  Repository Rules: " + moduleInfo.getRepositoryRuleInfoCount());
      System.out.println("  Module Extensions: " + moduleInfo.getModuleExtensionInfoCount());
    }
  }

  private ConstellateCli() {}
}
