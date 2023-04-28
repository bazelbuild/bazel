// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Streams.stream;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.PathStripper.PathMapper;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * A class that keeps a list of command lines and optional associated parameter file info.
 *
 * <p>This class is used by {@link com.google.devtools.build.lib.exec.SpawnRunner} implementations
 * to expand the command lines into a master argument list + any param files needed to be written.
 */
public class CommandLines {
  /**
   * An object that can apply the {@code stripPaths} map to optionally strip config prefixes before
   * returning output artifact exec paths
   */
  public interface PathStrippable {
    String expand(Function<PathFragment, PathFragment> stripPaths);
  }

  // A (hopefully) conservative estimate of how much long each param file arg would be
  // eg. the length of '@path/to/param_file'.
  private static final int PARAM_FILE_ARG_LENGTH_ESTIMATE = 512;
  private static final UUID PARAM_FILE_UUID =
      UUID.fromString("106c1389-88d7-4cc1-8f05-f8a61fd8f7b1");

  /** A simple tuple of a {@link CommandLine} and a {@link ParamFileInfo}. */
  public static class CommandLineAndParamFileInfo {
    public final CommandLine commandLine;
    @Nullable public final ParamFileInfo paramFileInfo;

    public CommandLineAndParamFileInfo(
        CommandLine commandLine, @Nullable ParamFileInfo paramFileInfo) {
      this.commandLine = commandLine;
      this.paramFileInfo = paramFileInfo;
    }
  }

  /**
   * Memory optimization: Store as Object instead of <code>List<CommandLineAndParamFileInfo></code>.
   *
   * <p>We store either a single CommandLine or CommandLineAndParamFileInfo, or list of Objects
   * where each item is either a CommandLine or CommandLineAndParamFileInfo. This minimizes unneeded
   * wrapper objects.
   *
   * <p>In the case of actions with a single CommandLine, this saves 48 bytes per action.
   */
  private final Object commandLines;

  private CommandLines(Object commandLines) {
    this.commandLines = commandLines;
  }

  /**
   * Expands this object into a single primary command line and (0-N) param files. The spawn runner
   * is expected to write these param files prior to execution of an action.
   *
   * @param artifactExpander The artifact expander to use.
   * @param paramFileBasePath Used to derive param file names. Often the first output of an action
   * @param pathMapper function to strip configuration prefixes from output paths, in accordance
   *     with the logic in {@link PathStripper}
   * @param limits The command line limits the host OS can support.
   * @return The expanded command line and its param files (if any).
   */
  public ExpandedCommandLines expand(
      ArtifactExpander artifactExpander,
      PathFragment paramFileBasePath,
      PathMapper pathMapper,
      CommandLineLimits limits)
      throws CommandLineExpansionException, InterruptedException {
    return expand(
        artifactExpander, paramFileBasePath, limits, pathMapper, PARAM_FILE_ARG_LENGTH_ESTIMATE);
  }

  @VisibleForTesting
  ExpandedCommandLines expand(
      ArtifactExpander artifactExpander,
      PathFragment paramFileBasePath,
      CommandLineLimits limits,
      PathMapper pathMapper,
      int paramFileArgLengthEstimate)
      throws CommandLineExpansionException, InterruptedException {
    // Optimize for simple case of single command line
    if (commandLines instanceof CommandLine) {
      CommandLine commandLine = (CommandLine) commandLines;
      Iterable<String> arguments = commandLine.arguments(artifactExpander);
      return new ExpandedCommandLines(arguments, ImmutableList.of());
    }
    List<CommandLineAndParamFileInfo> commandLines = getCommandLines();
    IterablesChain.Builder<String> arguments = IterablesChain.builder();
    ArrayList<ParamFileActionInput> paramFiles = new ArrayList<>(commandLines.size());
    int conservativeMaxLength = limits.maxLength - commandLines.size() * paramFileArgLengthEstimate;
    int cmdLineLength = 0;
    // We name based on the output, starting at <output>-0.params and then incrementing
    int paramFileNameSuffix = 0;
    for (CommandLineAndParamFileInfo pair : commandLines) {
      CommandLine commandLine = pair.commandLine;
      ParamFileInfo paramFileInfo = pair.paramFileInfo;
      if (paramFileInfo == null) {
        Iterable<String> args = commandLine.arguments(artifactExpander, pathMapper);
        arguments.add(args);
        cmdLineLength += totalArgLen(args);
      } else {
        Preconditions.checkNotNull(paramFileInfo); // If null, we would have just had a CommandLine
        Iterable<String> args = commandLine.arguments(artifactExpander, pathMapper);
        boolean useParamFile = true;
        if (!paramFileInfo.always()) {
          int tentativeCmdLineLength = cmdLineLength + totalArgLen(args);
          if (tentativeCmdLineLength <= conservativeMaxLength) {
            arguments.add(args);
            cmdLineLength = tentativeCmdLineLength;
            useParamFile = false;
          }
        }
        if (useParamFile) {
          PathFragment paramFileExecPath =
              ParameterFile.derivePath(paramFileBasePath, Integer.toString(paramFileNameSuffix));
          ++paramFileNameSuffix;

          String paramArg =
              SingleStringArgFormatter.format(
                  paramFileInfo.getFlagFormatString(),
                  pathMapper.map(paramFileExecPath).getPathString());
          arguments.addElement(paramArg);
          cmdLineLength += paramArg.length() + 1;

          if (paramFileInfo.flagsOnly()) {
            // Move just the flags into the file, and keep the positional parameters on the command
            // line.
            paramFiles.add(
                new ParamFileActionInput(
                    paramFileExecPath,
                    ParameterFile.flagsOnly(args),
                    paramFileInfo.getFileType(),
                    paramFileInfo.getCharset()));
            for (String positionalArg : ParameterFile.nonFlags(args)) {
              arguments.addElement(positionalArg);
              cmdLineLength += positionalArg.length() + 1;
            }
          } else {
            paramFiles.add(
                new ParamFileActionInput(
                    paramFileExecPath,
                    args,
                    paramFileInfo.getFileType(),
                    paramFileInfo.getCharset()));
          }
        }
      }
    }
    return new ExpandedCommandLines(arguments.build(), paramFiles);
  }

  /**
   * Returns all arguments, including ones inside of param files.
   *
   * <p>Suitable for debugging and printing messages to users. This expands all command lines, so it
   * is potentially expensive.
   */
  public ImmutableList<String> allArguments()
      throws CommandLineExpansionException, InterruptedException {
    return allArguments(PathMapper.NOOP);
  }

  /** Variation of {@link #allArguments()} that supports output path stripping. */
  public ImmutableList<String> allArguments(PathMapper stripPaths)
      throws CommandLineExpansionException, InterruptedException {
    ImmutableList.Builder<String> arguments = ImmutableList.builder();
    for (CommandLineAndParamFileInfo pair : getCommandLines()) {
      arguments.addAll(pair.commandLine.arguments(/*artifactExpander=*/ null, stripPaths));
    }
    return arguments.build();
  }

  public void addToFingerprint(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fingerprint)
      throws CommandLineExpansionException, InterruptedException {
    // Optimize for simple case of single command line
    if (commandLines instanceof CommandLine) {
      CommandLine commandLine = (CommandLine) commandLines;
      commandLine.addToFingerprint(actionKeyContext, artifactExpander, fingerprint);
      return;
    }
    List<CommandLineAndParamFileInfo> commandLines = getCommandLines();
    for (CommandLineAndParamFileInfo pair : commandLines) {
      CommandLine commandLine = pair.commandLine;
      ParamFileInfo paramFileInfo = pair.paramFileInfo;
      commandLine.addToFingerprint(actionKeyContext, artifactExpander, fingerprint);
      if (paramFileInfo != null) {
        addParamFileInfoToFingerprint(paramFileInfo, fingerprint);
      }
    }
  }

  /**
   * Expanded command lines.
   *
   * <p>The spawn runner implementation is expected to ensure the param files are available once the
   * spawn is executed.
   */
  public static class ExpandedCommandLines {
    private final Iterable<String> arguments;
    private final List<ParamFileActionInput> paramFiles;

    ExpandedCommandLines(
        Iterable<String> arguments,
        List<ParamFileActionInput> paramFiles) {
      this.arguments = arguments;
      this.paramFiles = paramFiles;
    }

    /** Returns the primary command line of the command. */
    public Iterable<String> arguments() {
      return arguments;
    }

    /** Returns the param file action inputs needed to execute the command. */
    public List<ParamFileActionInput> getParamFiles() {
      return paramFiles;
    }
  }

  /** An in-memory param file virtual action input. */
  public static final class ParamFileActionInput extends VirtualActionInput {
    private final PathFragment paramFileExecPath;
    private final Iterable<String> arguments;
    private final ParameterFileType type;
    private final Charset charset;

    public ParamFileActionInput(
        PathFragment paramFileExecPath,
        Iterable<String> arguments,
        ParameterFileType type,
        Charset charset) {
      this.paramFileExecPath = paramFileExecPath;
      this.arguments = arguments;
      this.type = type;
      this.charset = charset;
    }

    /**
     * Returns a cloned copy of this {@link ParamFileActionInput} replacing each command line
     * argument with an adjusted version determined by a given function.
     */
    public ParamFileActionInput withAdjustedArgs(Function<String, String> adjuster) {
      return new ParamFileActionInput(
          paramFileExecPath,
          stream(arguments).map(adjuster).collect(toImmutableList()),
          type,
          charset);
    }

    @Override
    public boolean isDirectory() {
      return false;
    }

    @Override
    public boolean isSymlink() {
      return false;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      ParameterFile.writeParameterFile(out, arguments, type, charset);
    }

    @Override
    @CanIgnoreReturnValue
    public byte[] atomicallyWriteTo(Path outputPath, String uniqueSuffix) throws IOException {
      // This is needed for internal path wrangling reasons :(
      return super.atomicallyWriteTo(outputPath, uniqueSuffix);
    }

    @Override
    public ByteString getBytes() throws IOException {
      ByteString.Output out = ByteString.newOutput();
      writeTo(out);
      return out.toByteString();
    }

    @Override
    public String getExecPathString() {
      return paramFileExecPath.getPathString();
    }

    @Override
    public PathFragment getExecPath() {
      return paramFileExecPath;
    }

    public ImmutableList<String> getArguments() {
      return ImmutableList.copyOf(arguments);
    }
  }

  // Helper function to unpack the optimized storage format into a list
  public List<CommandLineAndParamFileInfo> getCommandLines() {
    if (commandLines instanceof CommandLine) {
      return ImmutableList.of(new CommandLineAndParamFileInfo((CommandLine) commandLines, null));
    } else if (commandLines instanceof CommandLineAndParamFileInfo) {
      return ImmutableList.of((CommandLineAndParamFileInfo) commandLines);
    } else if (commandLines instanceof Object[]) {
      List<Object> commandLines = Arrays.asList((Object[]) this.commandLines);
      ImmutableList.Builder<CommandLineAndParamFileInfo> result =
          ImmutableList.builderWithExpectedSize(commandLines.size());
      for (Object commandLine : commandLines) {
        if (commandLine instanceof CommandLine) {
          result.add(new CommandLineAndParamFileInfo((CommandLine) commandLine, null));
        } else if (commandLine instanceof CommandLineAndParamFileInfo) {
          result.add((CommandLineAndParamFileInfo) commandLine);
        } else {
          result.add(new CommandLineAndParamFileInfo(new SingletonCommandLine(commandLine), null));
        }
      }
      return result.build();
    } else {
      return ImmutableList.of(
          new CommandLineAndParamFileInfo(new SingletonCommandLine(commandLines), null));
    }
  }

  private static int totalArgLen(Iterable<String> args) {
    int result = 0;
    for (String s : args) {
      result += s.length() + 1;
    }
    return result;
  }

  private static void addParamFileInfoToFingerprint(
      ParamFileInfo paramFileInfo, Fingerprint fingerprint) {
    fingerprint.addUUID(PARAM_FILE_UUID);
    fingerprint.addString(paramFileInfo.getFlagFormatString());
    fingerprint.addString(paramFileInfo.getFileType().toString());
    fingerprint.addString(paramFileInfo.getCharset().toString());
  }

  public static Builder builder() {
    return new Builder();
  }

  public static Builder builder(Builder other) {
    return new Builder(other);
  }

  /** Returns an instance with a single command line. */
  public static CommandLines of(CommandLine commandLine) {
    return new CommandLines(commandLine);
  }

  /** Returns an instance with a single trivial command line. */
  public static CommandLines of(Iterable<String> args) {
    return new CommandLines(CommandLine.of(args));
  }

  public static CommandLines concat(CommandLine commandLine, CommandLines commandLines) {
    Builder builder = builder();
    builder.addCommandLine(commandLine);
    for (CommandLineAndParamFileInfo pair : commandLines.getCommandLines()) {
      builder.addCommandLine(pair);
    }
    return builder.build();
  }

  /** Builder for {@link CommandLines}. */
  public static class Builder {
    private final List<Object> commandLines;

    Builder() {
      commandLines = new ArrayList<>();
    }

    Builder(Builder other) {
      commandLines = new ArrayList<>(other.commandLines);
    }

    @CanIgnoreReturnValue
    public Builder addSingleArgument(Object argument) {
      commandLines.add(argument);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addCommandLine(CommandLine commandLine) {
      commandLines.add(commandLine);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addCommandLine(CommandLine commandLine, ParamFileInfo paramFileInfo) {
      if (paramFileInfo == null) {
        commandLines.add(commandLine);
      } else {
        commandLines.add(new CommandLineAndParamFileInfo(commandLine, paramFileInfo));
      }
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addCommandLine(CommandLineAndParamFileInfo pair) {
      if (pair.paramFileInfo == null) {
        commandLines.add(pair.commandLine);
      } else {
        commandLines.add(pair);
      }
      return this;
    }

    public CommandLines build() {
      final Object commandLines;
      if (this.commandLines.size() == 1) {
        commandLines = this.commandLines.get(0);
      } else {
        Object[] result = new Object[this.commandLines.size()];
        for (int i = 0; i < this.commandLines.size(); ++i) {
          result[i] = this.commandLines.get(i);
        }
        commandLines = result;
      }
      return new CommandLines(commandLines);
    }
  }

  private static class SingletonCommandLine extends CommandLine {
    private final Object arg;

    public SingletonCommandLine(Object arg) {
      this.arg = arg;
    }

    @Override
    public Iterable<String> arguments() throws CommandLineExpansionException, InterruptedException {
      return arguments(null, PathMapper.NOOP);
    }

    @Override
    public Iterable<String> arguments(
        @Nullable ArtifactExpander artifactExpander, PathMapper pathMapper)
        throws CommandLineExpansionException, InterruptedException {
      if (arg instanceof PathStrippable) {
        return ImmutableList.of(((PathStrippable) arg).expand(pathMapper::map));
      }
      return ImmutableList.of(CommandLineItem.expandToCommandLine(arg));
    }
  }
}
