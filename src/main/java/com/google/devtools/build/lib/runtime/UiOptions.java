// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.runtime.UiStateTracker.ProgressMode;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/** Command-line UI options. */
public class UiOptions extends OptionsBase {

  /** Enum to select whether color output is enabled or not. */
  public enum UseColor {
    YES,
    NO,
    AUTO
  }

  /** Enum to select whether curses output is enabled or not. */
  public enum UseCurses {
    YES,
    NO,
    AUTO
  }

  /** Converter for {@link EventKind} filters * */
  public static class EventFiltersConverter implements Converter<List<EventKind>> {

    /** A converter for event kinds. */
    public static class EventKindConverter extends EnumConverter<EventKind> {

      public EventKindConverter(String typeName) {
        super(EventKind.class, typeName);
      }
    }

    private final CommaSeparatedOptionListConverter delegate;

    public EventFiltersConverter() {
      this.delegate = new CommaSeparatedOptionListConverter();
    }

    @Override
    public List<EventKind> convert(String input) throws OptionsParsingException {
      if (input.isEmpty()) {
        // This method is not called to convert the default value
        // Empty list means that the user wants to filter all events
        return new ArrayList<>(EventKind.ALL_EVENTS);
      }
      List<String> filters = this.delegate.convert(input);
      EnumConverter<EventKind> eventKindConverter = new EventKindConverter(input);

      HashSet<EventKind> filteredEvents = new HashSet<>();
      for (String filter : filters) {
        if (!filter.startsWith("+") && !filter.startsWith("-")) {
          filteredEvents.addAll(EventKind.ALL_EVENTS);
          break;
        }
      }

      for (String filter : filters) {
        if (filter.startsWith("+")) {
          filteredEvents.remove(eventKindConverter.convert(filter.substring(1)));
        } else if (filter.startsWith("-")) {
          filteredEvents.add(eventKindConverter.convert(filter.substring(1)));
        } else {
          filteredEvents.remove(eventKindConverter.convert(filter));
        }
      }
      return new ArrayList<>(filteredEvents);
    }

    @Override
    public String getTypeDescription() {
      return "Convert list of comma separated event kind to list of filters";
    }
  }

  /** Converter for {@link UseColor}. */
  public static class UseColorConverter extends EnumConverter<UseColor> {
    public UseColorConverter() {
      super(UseColor.class, "--color setting");
    }
  }

  /** Converter for {@link UseCurses}. */
  public static class UseCursesConverter extends EnumConverter<UseCurses> {
    public UseCursesConverter() {
      super(UseCurses.class, "--curses setting");
    }
  }

  /** Progress mode converter. */
  public static class ProgressModeConverter extends EnumConverter<ProgressMode> {
    public ProgressModeConverter() {
      super(ProgressMode.class, "--experimental_ui_mode setting");
    }
  }

  @Option(
      name = "show_progress",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Display progress messages during a build.")
  public boolean showProgress;

  @Option(
      name = "show_task_finish",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Display progress messages when tasks complete, not just when they start.")
  public boolean showTaskFinish;

  @Option(
      name = "show_progress_rate_limit",
      defaultValue = "0.2", // A nice middle ground; snappy but not too spammy in logs.
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Minimum number of seconds between progress messages in the output.")
  public double showProgressRateLimit;

  @Option(
      name = "color",
      defaultValue = "auto",
      converter = UseColorConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Use terminal controls to colorize output.")
  public UseColor useColorEnum;

  @Option(
      name = "curses",
      defaultValue = "auto",
      converter = UseCursesConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Use terminal cursor controls to minimize scrolling output.")
  public UseCurses useCursesEnum;

  @Option(
      name = "terminal_columns",
      defaultValue = "80",
      metadataTags = {OptionMetadataTag.HIDDEN},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "A system-generated parameter which specifies the terminal width in columns.")
  public int terminalColumns;

  @Option(
      name = "isatty",
      // TODO(b/137881511): Old name should be removed after 2020-01-01, or whenever is
      // reasonable.
      oldName = "is_stderr_atty",
      defaultValue = "false",
      metadataTags = {OptionMetadataTag.HIDDEN},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A system-generated parameter which is used to notify the "
              + "server whether this client is running in a terminal. "
              + "If this is set to false, then '--color=auto' will be treated as '--color=no'. "
              + "If this is set to true, then '--color=auto' will be treated as '--color=yes'.")
  public boolean isATty;

  // This lives here (as opposed to the more logical BuildRequest.Options)
  // because the client passes it to the server *always*.  We don't want the
  // client to have to figure out when it should or shouldn't to send it.
  @Option(
      name = "emacs",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "A system-generated parameter which is true iff EMACS=t or INSIDE_EMACS is set "
              + "in the environment of the client.  This option controls certain display "
              + "features.")
  public boolean runningInEmacs;

  @Option(
      name = "show_timestamps",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Include timestamps in messages")
  public boolean showTimestamp;

  @Option(
      name = "progress_in_terminal_title",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Show the command progress in the terminal title. "
              + "Useful to see what bazel is doing when having multiple terminal tabs.")
  public boolean progressInTermTitle;

  @Option(
      name = "attempt_to_print_relative_paths",
      oldName = "experimental_ui_attempt_to_print_relative_paths",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "When printing the location part of messages, attempt to use a path relative to the "
              + "workspace directory or one of the directories specified by --package_path.")
  public boolean attemptToPrintRelativePaths;

  @Option(
      name = "experimental_ui_debug_all_events",
      defaultValue = "false",
      metadataTags = {OptionMetadataTag.HIDDEN},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Report all events known to the Bazel UI.")
  public boolean experimentalUiDebugAllEvents;

  @Option(
      name = "ui_event_filters",
      converter = EventFiltersConverter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Specifies which events to show in the UI. It is possible to add or remove events "
              + "to the default ones using leading +/-, or override the default "
              + "set completely with direct assignment. The set of supported event kinds "
              + "include INFO, DEBUG, ERROR and more.",
      allowMultiple = true)
  public List<EventKind> eventFilters;

  @Option(
      name = "experimental_ui_mode",
      defaultValue = "oldest_actions",
      converter = ProgressModeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Determines what kind of data is shown in the detailed progress bar. By default, it is "
              + "set to show the oldest actions and their running time. The underlying data "
              + "source is usually sampled in a mode-dependend way to fit within the number of "
              + "lines given by --ui_actions_shown.")
  public ProgressMode uiProgressMode;

  @Option(
      name = "ui_actions_shown",
      oldName = "experimental_ui_actions_shown",
      defaultValue = "8",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Number of concurrent actions shown in the detailed progress bar; each "
              + "action is shown on a separate line. The progress bar always shows "
              + "at least one one, all numbers less than 1 are mapped to 1. "
              + "This option has no effect if --noui is set.")
  public int uiSamplesShown;

  @Option(
      name = "experimental_ui_limit_console_output",
      defaultValue = "0",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Number of bytes to which the UI will limit its output (non-positive "
              + "values indicate unlimited). Once the limit is approaching, the UI "
              + "will try hard to limit in a meaningful way, but will ultimately just drop all "
              + "output.")
  public int experimentalUiLimitConsoleOutput;

  @Option(
      name = "experimental_ui_deduplicate",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help = "Make the UI deduplicate messages to have a cleaner scroll-back log.")
  public boolean experimentalUiDeduplicate;

  public boolean useColor() {
    return useColorEnum == UseColor.YES || (useColorEnum == UseColor.AUTO && isATty);
  }

  public boolean useCursorControl() {
    return useCursesEnum == UseCurses.YES || (useCursesEnum == UseCurses.AUTO && isATty);
  }
}
