package com.google.devtools.build.lib.profiler;

import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.time.Duration;
import java.util.List;
import java.util.stream.Collectors;
import java.util.zip.GZIPOutputStream;
import javax.annotation.Nullable;
import one.jfr.JfrReader;
import one.converter.jfr2pprof;
import one.profiler.AsyncProfiler;
import one.profiler.Events;

/**
 * Bazel module to record pprof-compatible profiles for single invocations.
 */
public class CommandProfilerModule extends BlazeModule {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final AsyncProfiler profiler;

  static {
    AsyncProfiler profilerInstance = null;
    try {
      System.loadLibrary("async_profiler");
      profilerInstance = AsyncProfiler.getInstance();
    } catch (UnsatisfiedLinkError e) {
      logger.atWarning().log("Failed to load async_profiler JNI: %s", e);
    }
    profiler = profilerInstance;
  }

  /**
   * The available profile options for --experimental_command_profile.
   */
  public enum ProfileSelection {
    CPU,
  }

  public static final class Options extends OptionsBase {

    @Option(
        name = "experimental_command_profile",
        converter = ProfileSelectionConverter.class,
        allowMultiple = true,
        defaultValue = "null",
        documentationCategory = OptionDocumentationCategory.LOGGING,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "Records pprof-compatible profiles of 'cpu', 'heap', 'garbage', or 'contention'."
                + " Multiple kinds of profiles may be requested by repeating the flag. Each creates"
                + " a corresponding file (e.g. CPU.pprof) in the output base directory.")
    public List<ProfileSelection> selectedProfiles;
  }

  /** Options converter for --experimental_command_profile. */
  public static final class ProfileSelectionConverter extends EnumConverter<ProfileSelection> {
    public ProfileSelectionConverter() {
      super(ProfileSelection.class, "--experimental_command_profile setting");
    }
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return ImmutableList.of(Options.class);
  }

  @Nullable private File temporaryDumpFile;
  @Nullable private List<ProfileSelection> selectedProfiles;
  @Nullable private Path outputBase;
  @Nullable private Reporter reporter;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    Options options = env.getOptions().getOptions(Options.class);
    this.selectedProfiles = options.selectedProfiles;
    this.outputBase = env.getBlazeWorkspace().getOutputBase();
    this.reporter = env.getReporter();

    if (profiler == null || selectedProfiles.isEmpty()) {
      return;
    }

    try {
      temporaryDumpFile = File.createTempFile("profile", "tmp");
      profiler.execute(
          String.format("start,event=cpu,interval=10000000,file=%s,jfr", temporaryDumpFile));
    } catch (IOException e) {
      reporter.handle(Event.error("Starting pprof-compatible profile failed: " + e));
    }
  }

  @Override
  public void afterCommand() {
    if (profiler == null || selectedProfiles.isEmpty()) {
      return;
    }

    profiler.stop();

    try (JfrReader jfrReader = new JfrReader(temporaryDumpFile.getAbsolutePath());
        OutputStream pprofOut = new GZIPOutputStream(
            getAndAnnounceProfilePath(ProfileSelection.CPU).getOutputStream())) {
      new jfr2pprof(jfrReader).dump(pprofOut);
    } catch (Exception e) {
      reporter.handle(Event.error("Dumping pprof-compatible profile failed: " + e));
    }

    temporaryDumpFile.delete();

    this.selectedProfiles = null;
    this.outputBase = null;
    this.reporter = null;
    this.temporaryDumpFile = null;
  }

  private Path getAndAnnounceProfilePath(ProfileSelection profileSelection) {
    Path path = outputBase.getRelative(profileSelection.name() + ".pprof");
    reporter.handle(
        Event.info("Writing pprof-compatible " + profileSelection.name() + " profile to: " + path));
    return path;
  }
}
