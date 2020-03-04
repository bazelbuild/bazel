package com.google.devtools.build.lib.runtime.commands;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * Provides support for implementations for {@link BlazeCommand} to read target patterns from file.
 */
final class TargetPatternFileSupport {

  private TargetPatternFileSupport() {}

  /**
   * Reads a list of target patterns, either from the command-line residue or by reading newline
   * delimited target patterns from the --target_pattern_file flag. If --target_pattern_file is
   * specified and options contain a residue, or file cannot be read it throws an exception instead.
   */
  public static List<String> handleTargetPatternFile(
      CommandEnvironment env, OptionsParsingResult options)
      throws TargetPatternFileSupportException {
    List<String> targets = options.getResidue();
    BuildRequestOptions buildRequestOptions = options.getOptions(BuildRequestOptions.class);
    if (!targets.isEmpty() && !buildRequestOptions.targetPatternFile.isEmpty()) {
      throw new TargetPatternFileSupportException(
          "Command-line target pattern and --target_pattern_file cannot both be specified");
    } else if (!buildRequestOptions.targetPatternFile.isEmpty()) {
      // Works for absolute or relative file.
      Path residuePath =
          env.getWorkingDirectory().getRelative(buildRequestOptions.targetPatternFile);
      try {
        targets =
            Lists.newArrayList(FileSystemUtils.readLines(residuePath, StandardCharsets.UTF_8));
      } catch (IOException e) {
        throw new TargetPatternFileSupportException(
            "I/O error reading from " + residuePath.getPathString());
      }
    } else {
      try (SilentCloseable closeable =
          Profiler.instance().profile("ProjectFileSupport.getTargets")) {
        targets = ProjectFileSupport.getTargets(env.getRuntime().getProjectFileProvider(), options);
      }
    }
    return targets;
  }

  /**
   * TargetPatternFileSupportException gets thrown when TargetPatternFileSupport cannot return a
   * list of targets based on the supplied command line options.
   */
  public static class TargetPatternFileSupportException extends Exception {
    public TargetPatternFileSupportException(String message) { super(message); }
  }
}
