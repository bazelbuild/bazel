// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.runtime;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithOrderConstraint;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.ChunkList;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.CommandLine;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.CommandLineSection;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.Option;
import com.google.devtools.build.lib.runtime.proto.CommandLineOuterClass.OptionList;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.ParsedOptionDescription;
import com.google.devtools.common.options.proto.OptionFilters;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** A build event reporting the command line by which Bazel was invoked. */
public abstract class CommandLineEvent implements BuildEventWithOrderConstraint {
  protected final String productName;
  protected final OptionsProvider activeStartupOptions;
  protected final String commandName;
  protected final OptionsProvider commandOptions;

  CommandLineEvent(
      String productName,
      OptionsProvider activeStartupOptions,
      String commandName,
      OptionsProvider commandOptions) {
    this.productName = productName;
    this.activeStartupOptions = activeStartupOptions;
    this.commandName = commandName;
    this.commandOptions = commandOptions;
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(BuildEventId.buildStartedId());
  }

  CommandLineSection getExecutableSection() {
    return CommandLineSection.newBuilder()
        .setSectionLabel("executable")
        .setChunkList(ChunkList.newBuilder().addChunk(productName))
        .build();
  }

  CommandLineSection getCommandSection() {
    return CommandLineSection.newBuilder()
        .setSectionLabel("command")
        .setChunkList(ChunkList.newBuilder().addChunk(commandName))
        .build();
  }

  /**
   * Convert an array of tags to the equivalent proto-generated enum values.
   *
   * <p>The proto type is duplicate in order to not burden the OptionsParser with the proto
   * dependency. A test guarantees that the two enum types are kept in sync with matching indices.
   */
  static List<OptionFilters.OptionEffectTag> getProtoEffectTags(OptionEffectTag[] tagArray) {
    ArrayList<OptionFilters.OptionEffectTag> effectTags = new ArrayList<>(tagArray.length);
    for (OptionEffectTag tag : tagArray) {
      effectTags.add(OptionFilters.OptionEffectTag.forNumber(tag.getValue()));
    }
    return effectTags;
  }

  /**
   * Convert an array of tags to the equivalent proto-generated enum values.
   *
   * <p>The proto type is duplicate in order to not burden the OptionsParser with the proto
   * dependency. A test guarantees that the two enum types are kept in sync with matching indices.
   */
  static List<OptionFilters.OptionMetadataTag> getProtoMetadataTags(OptionMetadataTag[] tagArray) {
    ArrayList<OptionFilters.OptionMetadataTag> metadataTags = new ArrayList<>(tagArray.length);
    for (OptionMetadataTag tag : tagArray) {
      metadataTags.add(OptionFilters.OptionMetadataTag.forNumber(tag.getValue()));
    }
    return metadataTags;
  }

  List<Option> getOptionListFromParsedOptionDescriptions(
      List<ParsedOptionDescription> parsedOptionDescriptions) {
    List<Option> options = new ArrayList<>();
    for (ParsedOptionDescription parsedOption : parsedOptionDescriptions) {
      options.add(
          createOption(
              parsedOption.getOptionDefinition(),
              parsedOption.getCommandLineForm(),
              parsedOption.getUnconvertedValue()));
    }
    return options;
  }

  private Option createOption(
      OptionDefinition optionDefinition, String combinedForm, @Nullable String value) {
    Option.Builder option = Option.newBuilder();
    option.setCombinedForm(combinedForm);
    option.setOptionName(optionDefinition.getOptionName());
    if (value != null) {
      option.setOptionValue(value);
    }
    option.addAllEffectTags(getProtoEffectTags(optionDefinition.getOptionEffectTags()));
    option.addAllMetadataTags(getProtoMetadataTags(optionDefinition.getOptionMetadataTags()));
    return option.build();
  }

  /**
   * Returns the startup option section of the command line for the startup options as the server
   * received them at its startup. Since not all client options get passed to the server as startup
   * options, this might not represent the actual list of startup options as the user provided them.
   */
  CommandLineSection getActiveStartupOptions() {
    return CommandLineSection.newBuilder()
        .setSectionLabel("startup options")
        .setOptionList(
            OptionList.newBuilder()
                .addAllOption(
                    getOptionListFromParsedOptionDescriptions(
                        activeStartupOptions.asCompleteListOfParsedOptions())))
        .build();
  }

  /**
   * Returns the final part of the command line, containing whatever was left after obtaining the
   * command and its options.
   */
  CommandLineSection getResidual() {
    // Potential further split: how the residual, if any is accepted, gets interpreted depends on
    // the command. For example, for build commands, we might want to consider separating out
    // project files, as in runtime.commands.ProjectFileSupport. To properly report this, we would
    // need to let the command customize how the residual is listed. This catch-all could serve
    // as a default in this case.
    return CommandLineSection.newBuilder()
        .setSectionLabel("residual")
        .setChunkList(ChunkList.newBuilder().addAllChunk(commandOptions.getResidue()))
        .build();
  }

  /** This reports a reassembled version of the command line as Bazel received it. */
  public static class OriginalCommandLineEvent extends CommandLineEvent {
    public static final String LABEL = "original";
    private final Optional<List<Pair<String, String>>> originalStartupOptions;

    public OriginalCommandLineEvent(
        BlazeRuntime runtime,
        String commandName,
        OptionsProvider commandOptions,
        Optional<List<Pair<String, String>>> originalStartupOptions) {
      this(
          runtime.getProductName(),
          runtime.getStartupOptionsProvider(),
          commandName,
          commandOptions,
          originalStartupOptions);
    }

    @VisibleForTesting
    OriginalCommandLineEvent(
        String productName,
        OptionsProvider activeStartupOptions,
        String commandName,
        OptionsProvider commandOptions,
        Optional<List<Pair<String, String>>> originalStartupOptions) {
      super(productName, activeStartupOptions, commandName, commandOptions);
      this.originalStartupOptions = originalStartupOptions;
    }

    @Override
    public BuildEventId getEventId() {
      return BuildEventId.structuredCommandlineId(LABEL);
    }

    /**
     * Returns the literal command line options as received. These are not the final parsed values,
     * but are passed as is from the client, so we do not have the full OptionDefinition
     * information. In this form, only set the "combinedForm" field.
     */
    private CommandLineSection getStartupOptionSection() {
      if (originalStartupOptions.isPresent()) {
        List<Option> options = new ArrayList<>();
        for (Pair<String, String> sourceToOptionPair : originalStartupOptions.get()) {
          // Only add the options that were added by the command line.
          // TODO(b/19881919) decide the format that option source information should take and then
          // add all options, tagged with the source, instead of filtering out the rc options.
          if (sourceToOptionPair.first != null && sourceToOptionPair.first.isEmpty()) {
            options.add(
                Option.newBuilder().setCombinedForm(sourceToOptionPair.getSecond()).build());
          }
        }
        return CommandLineSection.newBuilder()
            .setSectionLabel("startup options")
            .setOptionList(OptionList.newBuilder().addAllOption(options))
            .build();
      } else {
        // If we were not provided with the startup options, fallback to reporting the active ones
        // stored by the Bazel Runtime.
        return getActiveStartupOptions();
      }
    }

    private CommandLineSection getExplicitCommandOptions() {
      List<ParsedOptionDescription> explicitOptions =
          commandOptions
              .asListOfExplicitOptions()
              .stream()
              .filter(
                  parsedOptionDescription ->
                      parsedOptionDescription.getPriority() == OptionPriority.COMMAND_LINE)
              .collect(Collectors.toList());
      return CommandLineSection.newBuilder()
          .setSectionLabel("command options")
          .setOptionList(
              OptionList.newBuilder()
                  .addAllOption(getOptionListFromParsedOptionDescriptions(explicitOptions)))
          .build();
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
      return GenericBuildEvent.protoChaining(this)
          .setStructuredCommandLine(
              CommandLine.newBuilder()
                  .setCommandLineLabel(LABEL)
                  .addSections(getExecutableSection())
                  .addSections(getStartupOptionSection())
                  .addSections(getCommandSection())
                  .addSections(getExplicitCommandOptions())
                  .addSections(getResidual())
                  .build())
          .build();
    }
  }

  /** This reports the canonical form of the command line. */
  public static class CanonicalCommandLineEvent extends CommandLineEvent {
    public static final String LABEL = "canonical";

    public CanonicalCommandLineEvent(
        BlazeRuntime runtime, String commandName, OptionsProvider commandOptions) {
      this(
          runtime.getProductName(),
          runtime.getStartupOptionsProvider(),
          commandName,
          commandOptions);
    }

    @VisibleForTesting
    CanonicalCommandLineEvent(
        String productName,
        OptionsProvider activeStartupOptions,
        String commandName,
        OptionsProvider commandOptions) {
      super(productName, activeStartupOptions, commandName, commandOptions);
    }

    @Override
    public BuildEventId getEventId() {
      return BuildEventId.structuredCommandlineId(LABEL);
    }

    /**
     * Returns the effective startup options.
     *
     * <p>Since in this command line the command options include invocation policy's and blazercs'
     * contents expanded fully, the list of startup options should prevent reapplication of these
     * contents.
     *
     * <p>The options parser does not understand the effect of these flags, since the relationship
     * between these startup options and the command options is not held within the options parser,
     * so instead, we add a small hack. Remove any explicit mentions of these flags, and explicitly
     * add the options that prevent Blaze from looking for the default rc files.
     */
    private CommandLineSection getCanonicalStartupOptions() {
      List<Option> unfilteredOptions = getActiveStartupOptions().getOptionList().getOptionList();
      // Create the fake ones to prevent reapplication of the original rc file contents.
      OptionsParser fakeOptions = OptionsParser.newOptionsParser(BlazeServerStartupOptions.class);
      try {
        fakeOptions.parse("--nomaster_blazerc", "--blazerc=/dev/null");
      } catch (OptionsParsingException e) {
        // Unless someone changes the definition of these flags, this is impossible.
        throw new IllegalStateException(e);
      }

      // Remove any instances of the applied, and add the new blocking ones.
      return CommandLineSection.newBuilder()
          .setSectionLabel("startup options")
          .setOptionList(
              OptionList.newBuilder()
                  .addAllOption(
                      unfilteredOptions
                          .stream()
                          .filter(
                              option -> {
                                String optionName = option.getOptionName();
                                return !optionName.equals("blazerc")
                                    && !optionName.equals("master_blazerc")
                                    && !optionName.equals("invocation_policy");
                              })
                          .collect(Collectors.toList()))
                  .addAllOption(
                      getOptionListFromParsedOptionDescriptions(
                          fakeOptions.asCompleteListOfParsedOptions())))
          .build();
    }

    /** Returns the canonical command options, overridden and default values are not listed. */
    // TODO(b/19881919) this should use OptionValueDescription's tracking of relevant option
    // instances, but as this is not yet possible, list the full options list.
    private CommandLineSection getCanonicalCommandOptions() {
      return CommandLineSection.newBuilder()
          .setSectionLabel("command options")
          .setOptionList(
              OptionList.newBuilder()
                  .addAllOption(
                      getOptionListFromParsedOptionDescriptions(
                          commandOptions.asCompleteListOfParsedOptions())))
          .build();
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
      return GenericBuildEvent.protoChaining(this)
          .setStructuredCommandLine(
              CommandLine.newBuilder()
                  .setCommandLineLabel(LABEL)
                  .addSections(getExecutableSection())
                  .addSections(getCanonicalStartupOptions())
                  .addSections(getCommandSection())
                  .addSections(getCanonicalCommandOptions())
                  .addSections(getResidual())
                  .build())
          .build();
    }
  }
}
