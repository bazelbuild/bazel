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
 * to resolve the command lines into a master argument list + any param files needed to be written.
 */
public class CommandLinesAndParamFiles {

  // A (hopefully) conservative estimate of how much long each param file arg would be
  // eg. the length of '@path/to/param_file'.
  private static final int PARAM_FILE_ARG_LENGTH_ESTIMATE = 512;
  private static final UUID PARAM_FILE_UUID =
      UUID.fromString("106c1389-88d7-4cc1-8f05-f8a61fd8f7b1");

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

  public CommandLinesAndParamFiles(List<CommandLineAndParamFileInfo> commandLines) {
    if (commandLines.size() == 1) {
      CommandLineAndParamFileInfo pair = commandLines.get(0);
      if (pair.paramFileInfo != null) {
        this.commandLines = pair;
      } else {
        this.commandLines = pair.commandLine;
      }
    } else {
      Object[] result = new Object[commandLines.size()];
      for (int i = 0; i < commandLines.size(); ++i) {
        CommandLineAndParamFileInfo pair = commandLines.get(i);
        if (pair.paramFileInfo != null) {
          result[i] = pair;
        } else {
          result[i] = pair.commandLine;
        }
      }
      this.commandLines = result;
    }
  }

  /**
   * Resolves this object into a single primary command line and (0-N) param files. The spawn runner
   * is expected to write these param files prior to execution of an action.
   *
   * @param artifactExpander The artifact expander to use.
   * @param primaryOutput The primary output of the action. Used to derive param file names.
   * @param maxLength The maximum command line length the executing host system can tolerate.
   * @return The resolved command line and its param files (if any).
   */
  public ResolvedCommandLineAndParamFiles resolve(
      ArtifactExpander artifactExpander, Artifact primaryOutput, int maxLength)
      throws CommandLineExpansionException {
    return resolve(
        artifactExpander, primaryOutput.getExecPath(), maxLength, PARAM_FILE_ARG_LENGTH_ESTIMATE);
  }

  @VisibleForTesting
  ResolvedCommandLineAndParamFiles resolve(
      ArtifactExpander artifactExpander,
      PathFragment primaryOutputExecPath,
      int maxLength,
      int paramFileArgLengthEstimate)
      throws CommandLineExpansionException {
    // Optimize for simple case of single command line
    if (commandLines instanceof CommandLine) {
      CommandLine commandLine = (CommandLine) commandLines;
      return new ResolvedCommandLineAndParamFiles(
          commandLine.arguments(artifactExpander), ImmutableList.of());
    }
    List<Object> commandLines =
        (this.commandLines instanceof CommandLineAndParamFileInfo)
            ? ImmutableList.of(this.commandLines)
            : Arrays.asList((Object[]) this.commandLines);
    IterablesChain.Builder<String> arguments = IterablesChain.builder();
    ArrayList<ParamFileActionInput> paramFiles = new ArrayList<>(commandLines.size());
    int conservativeMaxLength = maxLength - commandLines.size() * paramFileArgLengthEstimate;
    int cmdLineLength = 0;
    // We name based on the output, starting at <output>-0.params and then incrementing
    int paramFileNameSuffix = 0;
    for (Object object : commandLines) {
      if (object instanceof CommandLine) {
        CommandLine commandLine = (CommandLine) object;
        Iterable<String> args = commandLine.arguments(artifactExpander);
        arguments.add(args);
        cmdLineLength += totalArgLen(args);
      } else {
        CommandLineAndParamFileInfo pair = (CommandLineAndParamFileInfo) object;
        CommandLine commandLine = pair.commandLine;
        ParamFileInfo paramFileInfo = pair.paramFileInfo;
        Preconditions.checkNotNull(paramFileInfo); // If null, we would have just had a CommandLine
        Iterable<String> args = commandLine.arguments(artifactExpander);
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
              ParameterFile.derivePath(
                  primaryOutputExecPath, Integer.toString(paramFileNameSuffix));
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
    return new ResolvedCommandLineAndParamFiles(arguments.build(), paramFiles);
  }

  public void addToFingerprint(ActionKeyContext actionKeyContext, Fingerprint fingerprint)
      throws CommandLineExpansionException {
    // Optimize for simple case of single command line
    if (commandLines instanceof CommandLine) {
      CommandLine commandLine = (CommandLine) commandLines;
      commandLine.addToFingerprint(actionKeyContext, fingerprint);
      return;
    }
    List<Object> commandLines =
        (this.commandLines instanceof CommandLineAndParamFileInfo)
            ? ImmutableList.of(this.commandLines)
            : Arrays.asList((Object[]) this.commandLines);
    for (Object object : commandLines) {
      if (object instanceof CommandLine) {
        CommandLine commandLine = (CommandLine) object;
        commandLine.addToFingerprint(actionKeyContext, fingerprint);
      } else {
        CommandLineAndParamFileInfo pair = (CommandLineAndParamFileInfo) object;
        CommandLine commandLine = pair.commandLine;
        commandLine.addToFingerprint(actionKeyContext, fingerprint);
        ParamFileInfo paramFileInfo = pair.paramFileInfo;
        Preconditions.checkNotNull(paramFileInfo); // If null, we would have just had a CommandLine
        addParamFileInfoToFingerprint(paramFileInfo, fingerprint);
      }
    }
  }

  /**
   * Resolved command lines.
   *
   * <p>The spawn runner implementation is expected to ensure the param files are available once the
   * spawn is executed.
   */
  public static class ResolvedCommandLineAndParamFiles {
    private final Iterable<String> arguments;
    private final List<ParamFileActionInput> paramFiles;

    ResolvedCommandLineAndParamFiles(
        Iterable<String> arguments, List<ParamFileActionInput> paramFiles) {
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

  /** Builder for {@link CommandLinesAndParamFiles}. */
  public static class Builder {
    List<CommandLineAndParamFileInfo> commandLines = new ArrayList<>();

    public Builder addCommandLine(CommandLine commandLine) {
      commandLines.add(new CommandLineAndParamFileInfo(commandLine, null));
      return this;
    }

    public Builder addCommandLine(CommandLine commandLine, ParamFileInfo paramFileInfo) {
      commandLines.add(new CommandLineAndParamFileInfo(commandLine, paramFileInfo));
      return this;
    }

    public CommandLinesAndParamFiles build() {
      return new CommandLinesAndParamFiles(commandLines);
    }
  }
}
