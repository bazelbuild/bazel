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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * A class that keeps a list of command lines and optional associated parameter file info.
 *
 * <p>This class is used by {@link com.google.devtools.build.lib.exec.SpawnRunner} implementations
 * to expand the command lines into a master argument list + any param files needed to be written.
 */
public class CommandLines {

  // A (hopefully) conservative estimate of how much long each param file arg would be
  // eg. the length of '@path/to/param_file'.
  private static final int PARAM_FILE_ARG_LENGTH_ESTIMATE = 512;
  private static final UUID PARAM_FILE_UUID =
      UUID.fromString("106c1389-88d7-4cc1-8f05-f8a61fd8f7b1");

  /** Command line OS limitations, such as the max length. */
  public static class CommandLineLimits {
    public final int maxLength;

    public CommandLineLimits(int maxLength) {
      this.maxLength = maxLength;
    }
  }

  /** A simple tuple of a {@link CommandLine} and a {@link ParamFileInfo}. */
  public static class CommandLineAndParamFileInfo {
    private final CommandLine commandLine;
    @Nullable private final ParamFileInfo paramFileInfo;

    private CommandLineAndParamFileInfo(
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
   * @param paramFileBasePath Used to derive param file names. Often the first output of an action.
   * @param limits The command line limits the host OS can support.
   * @return The expanded command line and its param files (if any).
   */
  public ExpandedCommandLines expand(
      ArtifactExpander artifactExpander, PathFragment paramFileBasePath, CommandLineLimits limits)
      throws CommandLineExpansionException {
    return expand(artifactExpander, paramFileBasePath, limits, PARAM_FILE_ARG_LENGTH_ESTIMATE);
  }

  /**
   * Returns all arguments, including ones inside of param files.
   *
   * <p>Suitable for debugging and printing messages to users. This expands all command lines, so it
   * is potentially expensive.
   */
  public ImmutableList<String> allArguments() throws CommandLineExpansionException {
    ImmutableList.Builder<String> arguments = ImmutableList.builder();
    for (CommandLineAndParamFileInfo pair : getCommandLines()) {
      arguments.addAll(pair.commandLine.arguments());
    }
    return arguments.build();
  }

  @VisibleForTesting
  ExpandedCommandLines expand(
      ArtifactExpander artifactExpander,
      PathFragment paramFileBasePath,
      CommandLineLimits limits,
      int paramFileArgLengthEstimate)
      throws CommandLineExpansionException {
    // Optimize for simple case of single command line
    if (commandLines instanceof CommandLine) {
      CommandLine commandLine = (CommandLine) commandLines;
      Iterable<String> arguments = commandLine.arguments(artifactExpander);
      return new ExpandedCommandLines(arguments, arguments, ImmutableList.of());
    }
    List<CommandLineAndParamFileInfo> commandLines = getCommandLines();
    IterablesChain.Builder<String> arguments = IterablesChain.builder();
    IterablesChain.Builder<String> allArguments = IterablesChain.builder();
    ArrayList<ParamFileActionInput> paramFiles = new ArrayList<>(commandLines.size());
    int conservativeMaxLength = limits.maxLength - commandLines.size() * paramFileArgLengthEstimate;
    int cmdLineLength = 0;
    // We name based on the output, starting at <output>-0.params and then incrementing
    int paramFileNameSuffix = 0;
    for (CommandLineAndParamFileInfo pair : commandLines) {
      CommandLine commandLine = pair.commandLine;
      ParamFileInfo paramFileInfo = pair.paramFileInfo;
      if (paramFileInfo == null) {
        Iterable<String> args = commandLine.arguments(artifactExpander);
        allArguments.add(args);
        arguments.add(args);
        cmdLineLength += totalArgLen(args);
      } else {
        Preconditions.checkNotNull(paramFileInfo); // If null, we would have just had a CommandLine
        Iterable<String> args = commandLine.arguments(artifactExpander);
        allArguments.add(args);
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
              paramFileInfo
                  .getFlagFormatString()
                  .replaceFirst("%s", paramFileExecPath.getPathString());
          arguments.addElement(paramArg);
          cmdLineLength += paramArg.length() + 1;
          paramFiles.add(new ParamFileActionInput(paramFileExecPath, args, paramFileInfo));
        }
      }
    }
    return new ExpandedCommandLines(arguments.build(), allArguments.build(), paramFiles);
  }

  public void addToFingerprint(ActionKeyContext actionKeyContext, Fingerprint fingerprint)
      throws CommandLineExpansionException {
    // Optimize for simple case of single command line
    if (commandLines instanceof CommandLine) {
      CommandLine commandLine = (CommandLine) commandLines;
      commandLine.addToFingerprint(actionKeyContext, fingerprint);
      return;
    }
    List<CommandLineAndParamFileInfo> commandLines = getCommandLines();
    for (CommandLineAndParamFileInfo pair : commandLines) {
      CommandLine commandLine = pair.commandLine;
      ParamFileInfo paramFileInfo = pair.paramFileInfo;
      commandLine.addToFingerprint(actionKeyContext, fingerprint);
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
    private final Iterable<String> allArguments;
    private final List<ParamFileActionInput> paramFiles;

    ExpandedCommandLines(
        Iterable<String> arguments,
        Iterable<String> allArguments,
        List<ParamFileActionInput> paramFiles) {
      this.arguments = arguments;
      this.allArguments = allArguments;
      this.paramFiles = paramFiles;
    }

    /** Returns the primary command line of the command. */
    public Iterable<String> arguments() {
      return arguments;
    }

    /**
     * Returns all arguments, including ones inside of param files.
     *
     * <p>Suitable for debugging and printing messages to users.
     */
    public Iterable<String> allArguments() {
      return allArguments;
    }

    /** Returns the param file action inputs needed to execute the command. */
    public List<ParamFileActionInput> getParamFiles() {
      return paramFiles;
    }

    /** Convenience function to write all param files locally under the given exec root. */
    public void writeParamFiles(Path execRoot) throws IOException {
      for (ParamFileActionInput actionInput : paramFiles) {
        Path paramFilePath = execRoot.getRelative(actionInput.paramFileExecPath);
        paramFilePath.getParentDirectory().createDirectoryAndParents();
        actionInput.writeTo(paramFilePath.getOutputStream());
      }
    }
  }

  static final class ParamFileActionInput implements VirtualActionInput {
    final PathFragment paramFileExecPath;
    final Iterable<String> arguments;
    final ParamFileInfo paramFileInfo;

    ParamFileActionInput(
        PathFragment paramFileExecPath, Iterable<String> arguments, ParamFileInfo paramFileInfo) {
      this.paramFileExecPath = paramFileExecPath;
      this.arguments = arguments;
      this.paramFileInfo = paramFileInfo;
    }

    @Override
    public void writeTo(OutputStream out) throws IOException {
      ParameterFile.writeParameterFile(
          out, arguments, paramFileInfo.getFileType(), paramFileInfo.getCharset());
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
  }

  // Helper function to unpack the optimized storage format into a list
  @SuppressWarnings("unchecked")
  private List<CommandLineAndParamFileInfo> getCommandLines() {
    if (commandLines instanceof CommandLine) {
      return ImmutableList.of(new CommandLineAndParamFileInfo((CommandLine) commandLines, null));
    } else if (commandLines instanceof CommandLineAndParamFileInfo) {
      return ImmutableList.of((CommandLineAndParamFileInfo) commandLines);
    } else {
      List<Object> commandLines = Arrays.asList((Object[]) this.commandLines);
      ImmutableList.Builder<CommandLineAndParamFileInfo> result =
          ImmutableList.builderWithExpectedSize(commandLines.size());
      for (Object commandLine : commandLines) {
        if (commandLine instanceof CommandLine) {
          result.add(new CommandLineAndParamFileInfo((CommandLine) commandLine, null));
        } else {
          result.add((CommandLineAndParamFileInfo) commandLine);
        }
      }
      return result.build();
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

  /** Returns an instance with a single trivial command line. */
  public static CommandLines fromArguments(Iterable<String> args) {
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
    List<CommandLineAndParamFileInfo> commandLines = new ArrayList<>();

    public Builder addCommandLine(CommandLine commandLine) {
      commandLines.add(new CommandLineAndParamFileInfo(commandLine, null));
      return this;
    }

    public Builder addCommandLine(CommandLine commandLine, ParamFileInfo paramFileInfo) {
      return addCommandLine(new CommandLineAndParamFileInfo(commandLine, paramFileInfo));
    }

    public Builder addCommandLine(CommandLineAndParamFileInfo pair) {
      commandLines.add(pair);
      return this;
    }

    public CommandLines build() {
      final Object commandLines;
      if (this.commandLines.size() == 1) {
        CommandLineAndParamFileInfo pair = this.commandLines.get(0);
        if (pair.paramFileInfo != null) {
          commandLines = pair;
        } else {
          commandLines = pair.commandLine;
        }
      } else {
        Object[] result = new Object[this.commandLines.size()];
        for (int i = 0; i < this.commandLines.size(); ++i) {
          CommandLineAndParamFileInfo pair = this.commandLines.get(i);
          if (pair.paramFileInfo != null) {
            result[i] = pair;
          } else {
            result[i] = pair.commandLine;
          }
        }
        commandLines = result;
      }
      return new CommandLines(commandLines);
    }
  }
}
