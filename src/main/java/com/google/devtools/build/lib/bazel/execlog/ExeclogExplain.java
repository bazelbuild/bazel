package com.google.devtools.build.lib.bazel.execlog;

import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import com.google.devtools.build.lib.util.io.AnsiTerminal.Color;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.exec.Protos;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.Message;
import com.google.protobuf.Duration;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Generates an explanation of the actions performed by the build by comparing
 * the current action spawn with the one from the previous build.
 */
public final class ExeclogExplain implements MessageOutputStream {

  private final PrintWriter writer;
  private final ExeclogHistory history;

  public ExeclogExplain(CommandEnvironment env, PathFragment explainPath) throws IOException {
    history = new ExeclogHistory(env.getOutputBase());
    OutputStream out = env.getWorkspace().getRelative(explainPath).getOutputStream();
    writer = new PrintWriter(new OutputStreamWriter(out, StandardCharsets.UTF_8), true);

    env.getReporter().handle(Event.info("Writing detail explanation of rebuilds to '" + explainPath + "'"));
    writeHeader();
  }

  private void writeHeader() throws IOException {
    writer.println("This file lists the build actions executed using the following format:");
    // seconds num_inputs->num_outputs build action changed_inputs -> changed_outputs
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    AnsiTerminal terminal = new AnsiTerminal(out);
    terminal.textGreen();
    terminal.writeString("seconds");
    terminal.textCyan();
    terminal.writeString(" num_inputs");
    terminal.textGray();
    terminal.writeString("->");
    terminal.textMagenta();
    terminal.writeString("num_outputs");
    terminal.resetTerminal();
    terminal.writeString(" build action");
    terminal.textCyan();
    terminal.writeString(" changed_inputs");
    terminal.textGray();
    terminal.writeString(" ->");
    terminal.textMagenta();
    terminal.writeString(" changed_outputs\n");
    terminal.flush();
    terminal.resetTerminal();
    writer.println(out);
  }

  @Override
  public void write(Message m) throws IOException {
    Protos.SpawnExec spawn = (Protos.SpawnExec) m;
    Protos.SpawnExec baseSpawn = history.get(spawn);
    Duration walltime = spawn.getWalltime();
    double seconds = ((double) walltime.getNanos()) / 1_000_000_000 + walltime.getSeconds();
    List<Protos.File> inputs = spawn.getInputsList();
    List<Protos.File> outputs = spawn.getActualOutputsList();

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    AnsiTerminal terminal = new AnsiTerminal(out);
    terminal.setTextColor(seconds >= 100? Color.RED : seconds >= 10? Color.YELLOW : Color.GREEN);
    terminal.writeString(String.format("%7.2f", seconds));
    terminal.writeString(String.format(" %c", spawn.getRemoteCacheHit()? 'C': ' '));
    terminal.textCyan();
    terminal.writeString(String.format("%5d", inputs.size()));
    terminal.textGray();
    terminal.writeString("->");
    terminal.textMagenta();
    terminal.writeString(String.format("%4d ", outputs.size()));
    terminal.resetTerminal();
    terminal.writeString(spawn.getProgressMessage());

    if (baseSpawn != null) {
      // show changed/removed/new inputs
      List<Protos.File> baseInputs = baseSpawn.getInputsList();
      Map<String, Protos.File> baseInputsMap = new HashMap();
      Map<String, Protos.File> baseInputsLeftMap = new HashMap();
      for (Protos.File baseInput: baseInputs) {
        baseInputsMap.put(baseInput.getPath(), baseInput);
        baseInputsLeftMap.put(baseInput.getPath(), baseInput);
      }
      int numDiffInputs = 0;
      List<Protos.File> newInputs = new ArrayList<>();
      terminal.textCyan();
      // changed inputs
      for (Protos.File input: inputs) {
        String path = input.getPath();
        baseInputsLeftMap.remove(path);
        Protos.File baseInput = baseInputsMap.get(path);
        if (baseInput != null) {
          if (!input.getDigest().equals(baseInput.getDigest())) {
            numDiffInputs++;
            terminal.writeString(' ' + basename(path));
          }
        } else {
          newInputs.add(input);
        }
      }
      // removed inputs
      for (Protos.File input: baseInputsLeftMap.values()) {
        numDiffInputs++;
        terminal.writeString(" -" + basename(input.getPath()));
      }
      // new inputs
      for (Protos.File input: newInputs) {
        numDiffInputs++;
        terminal.writeString(" +" + basename(input.getPath()));
      }
      if (numDiffInputs == 0) {
        terminal.textYellow();
        terminal.writeString(" [unchanged]");
      }

      // show changed/removed/new outputs
      terminal.textGray();
      terminal.writeString(" ->");
      List<Protos.File> baseOutputs = baseSpawn.getActualOutputsList();
      Map<String, Protos.File> baseOutputsMap = new HashMap();
      Map<String, Protos.File> baseOutputsLeftMap = new HashMap();
      for (Protos.File baseOutput: baseOutputs) {
        baseOutputsMap.put(baseOutput.getPath(), baseOutput);
        baseOutputsLeftMap.put(baseOutput.getPath(), baseOutput);
      }
      int numDiffOutputs = 0;
      List<Protos.File> newOutputs = new ArrayList<>();
      terminal.textMagenta();
      // changed outputs
      for (Protos.File output: outputs) {
        String path = output.getPath();
        baseOutputsLeftMap.remove(path);
        Protos.File baseOutput = baseOutputsMap.get(path);
        if (baseOutput != null) {
          if (!output.getDigest().equals(baseOutput.getDigest())) {
            numDiffOutputs++;
            terminal.writeString(' ' + basename(path));
          }
        } else {
          newOutputs.add(output);
        }
      }
      // removed outputs
      for (Protos.File output: baseOutputsLeftMap.values()) {
        numDiffOutputs++;
        terminal.writeString(" -" + basename(output.getPath()));
      }
      // new outputs
      for (Protos.File output: newOutputs) {
        numDiffOutputs++;
        terminal.writeString(" +" + basename(output.getPath()));
      }
      if (numDiffOutputs == 0) {
        terminal.textYellow();
        terminal.writeString(" [unchanged]");
      }
    } else {
      terminal.textYellow();
      terminal.writeString(" [no history]");
    }

    terminal.flush();
    terminal.resetTerminal();
    writer.println(out);

    history.put(spawn);
  }

  @Override
  public void close() throws IOException {
    writer.close();
  }

  private static String basename(String path) {
    return path.substring(path.lastIndexOf(File.separatorChar) + 1);
  }
}
