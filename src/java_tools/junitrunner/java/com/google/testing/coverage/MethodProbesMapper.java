// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.coverage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.jacoco.core.internal.analysis.Instruction;
import org.jacoco.core.internal.flow.IFrame;
import org.jacoco.core.internal.flow.LabelInfo;
import org.jacoco.core.internal.flow.MethodProbesVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Label;

/**
 * The mapper is a probes visitor that will cache control flow information as well as keeping track
 * of the probes as the main driver generates the probe ids. Upon finishing the method it uses the
 * information collected to generate the mapping information between probes and the instructions.
 */
public class MethodProbesMapper extends MethodProbesVisitor {
  /*
   * The implementation roughly follows the same pattern of the Analyzer class of Jacoco.
   *
   * The mapper has a few states:
   *
   * - lineMappings: a mapping between line number and labels
   *
   * - a sequence of "instructions", where each instruction has one or more predecessors. The
   * predecessor field has a sole purpose of propagating probe id. The 'merge' nodes in the CFG has
   * no predecessors, since the branch stops at theses points.
   *
   * - The instructions each has states that keep track of the probes that are associated with the
   * instruction.
   *
   * Initially the probe ids are assigned to the instructions that immediately precede the probe. At
   * the end of visiting the methods, the probe ids are propagated through the predecessor chains.
   */

  // States
  //
  // These are state variables that needs to be updated in the visitor methods.
  // The values usually changes as we traverse the byte code.
  private Instruction lastInstruction = null;
  private int currentLine = -1;
  private List<Label> currentLabels = new ArrayList<>();

  // Result
  private Map<Integer, BranchExp> lineToBranchExp = new TreeMap();
  public Map<Integer, BranchExp> result() {
    return lineToBranchExp;
  }

  // Intermediate results
  //
  // These values are built up during the visitor methods. They will be used to compute
  // the final results.
  private List<Instruction> instructions = new ArrayList<Instruction>();
  private List<Jump> jumps = new ArrayList<>();
  private Map<Integer, Instruction> probeToInsn = new TreeMap<>();

  // A map which associates intructions with their coverage expressions.
  private final Map<Instruction, CovExp> insnToCovExp = new HashMap();

  // A map which associates a instruction to the branch index in its predecessor
  // e.g., the instruction that follows a conditional jump instruction must exists in
  // this map.
  private final Map<Instruction, Integer> insnToIdx = new HashMap();

  // Local cache
  //
  // These are maps corresponding to data structures available in JaCoCo in other form.
  // We use local data structure to avoid need to change the JaCoCo internal code.
  private Map<Instruction, Instruction> predecessors = new HashMap<>();
  private Map<Label, Instruction> labelToInsn = new HashMap<>();

  /** Visitor method to append a new Instruction */
  private void visitInsn() {
    Instruction instruction = new Instruction(currentLine);
    instructions.add(instruction);
    if (lastInstruction != null) {
      lastInstruction.addBranch(instruction, 0); // the first branch from last instruction
      predecessors.put(instruction, lastInstruction); // Update local cache
    }

    for (Label label : currentLabels) {
      labelToInsn.put(label, instruction);
    }
    currentLabels.clear(); // Update states
    lastInstruction = instruction;
  }

  // Plain visitors: called from adapter when no probe is needed
  @Override
  public void visitInsn(int opcode) {
    visitInsn();
  }

  @Override
  public void visitIntInsn(int opcode, int operand) {
    visitInsn();
  }

  @Override
  public void visitVarInsn(int opcode, int variable) {
    visitInsn();
  }

  @Override
  public void visitTypeInsn(int opcode, String type) {
    visitInsn();
  }

  @Override
  public void visitFieldInsn(int opcode, String owner, String name, String desc) {
    visitInsn();
  }

  @Override
  public void visitMethodInsn(int opcode, String owner, String name, String desc, boolean itf) {
    visitInsn();
  }

  @Override
  public void visitInvokeDynamicInsn(String name, String desc, Handle handle, Object... args) {
    visitInsn();
  }

  @Override
  public void visitLdcInsn(Object cst) {
    visitInsn();
  }

  @Override
  public void visitIincInsn(int var, int inc) {
    visitInsn();
  }

  @Override
  public void visitMultiANewArrayInsn(String desc, int dims) {
    visitInsn();
  }


  // Methods that need to update the states
  @Override
  public void visitJumpInsn(int opcode, Label label) {
    visitInsn();
    jumps.add(new Jump(lastInstruction, label, 1));
  }

  @Override
  public void visitLabel(Label label) {
    currentLabels.add(label);
    if (!LabelInfo.isSuccessor(label)) {
      lastInstruction = null;
    }
  }

  @Override
  public void visitLineNumber(int line, Label start) {
    currentLine = line;
  }

  /** Visit a switch instruction with no probes */
  private void visitSwitchInsn(Label dflt, Label[] labels) {
    visitInsn();

    // Handle default transition
    LabelInfo.resetDone(dflt);
    int branch = 0;
    jumps.add(new Jump(lastInstruction, dflt, branch));
    LabelInfo.setDone(dflt);

    // Handle other transitions
    LabelInfo.resetDone(labels);
    for (Label label : labels) {
      if (!LabelInfo.isDone(label)) {
        jumps.add(new Jump(lastInstruction, label, branch));
        LabelInfo.setDone(label);
      }
    }
  }

  @Override
  public void visitTableSwitchInsn(int min, int max, Label dflt, Label... labels) {
    visitSwitchInsn(dflt, labels);
  }

  @Override
  public void visitLookupSwitchInsn(Label dflt, int[] keys, Label[] labels) {
    visitSwitchInsn(dflt, labels);
  }

  private void addProbe(int probeId) {
    // We do not add probes to the flow graph, but we need to update
    // the branch count of the predecessor of the probe
    lastInstruction.addBranch(false, 0);
    probeToInsn.put(probeId, lastInstruction);
  }

  // Probe visit methods
  @Override
  public void visitProbe(int probeId) {
    // This function is only called when visiting a merge node which
    // is a successor.
    // It adds an probe point to the last instruction
    assert (lastInstruction != null);

    addProbe(probeId);
    lastInstruction = null; // Merge point should have no predecessor.
  }

  @Override
  public void visitJumpInsnWithProbe(int opcode, Label label, int probeId, IFrame frame) {
    visitInsn();
    addProbe(probeId);
  }

  @Override
  public void visitInsnWithProbe(int opcode, int probeId) {
    visitInsn();
    addProbe(probeId);
  }

  @Override
  public void visitTableSwitchInsnWithProbes(
      int min, int max, Label dflt, Label[] labels, IFrame frame) {
    visitSwitchInsnWithProbes(dflt, labels);
  }

  @Override
  public void visitLookupSwitchInsnWithProbes(
      Label dflt, int[] keys, Label[] labels, IFrame frame) {
    visitSwitchInsnWithProbes(dflt, labels);
  }

  private void visitSwitchInsnWithProbes(Label dflt, Label[] labels) {
    visitInsn();
    LabelInfo.resetDone(dflt);
    LabelInfo.resetDone(labels);
    int branch = 0;
    visitTargetWithProbe(dflt, branch);
    for (Label l : labels) {
      visitTargetWithProbe(l, branch);
    }
  }

  private void visitTargetWithProbe(Label label, int branch) {
    if (!LabelInfo.isDone(label)) {
      int id = LabelInfo.getProbeId(label);
      if (id == LabelInfo.NO_PROBE) {
        jumps.add(new Jump(lastInstruction, label, branch));
      } else {
        // Note, in this case the instrumenter should insert intermediate labels
        // for the probes. These probes will be added for the switch instruction.
        //
        // There is no direct jump between lastInstruction and the label either.
        addProbe(id);
      }
      LabelInfo.setDone(label);
    }
  }

  // If a CovExp of pred is ProbeExp, create a single-branch BranchExp and put it in the map.
  // Also update the index of insn.
  private BranchExp getPredBranchExp(Instruction predecessor) {
    BranchExp result = null;
    CovExp exp = insnToCovExp.get(predecessor);
    if (exp instanceof ProbeExp) {
      result = new BranchExp(exp); // Change ProbeExp to BranchExp
      insnToCovExp.put(predecessor, result);
      // This can only happen if the internal data of Jacoco is inconsistent:
      // the instruction is the predecessor of more than one instructions,
      // but its branch count is not > 1.
    } else {
      result = (BranchExp) exp;
    }
    return result;
  }

  // Update a branch predecessor and returns whether the BranchExp of the predecessor is new.
  private boolean updateBranchPredecessor(Instruction predecessor, Instruction insn, CovExp exp) {
    CovExp predExp = insnToCovExp.get(predecessor);
    if (predExp == null) {
      BranchExp branchExp = new BranchExp(exp);
      insnToCovExp.put(predecessor, branchExp);
      insnToIdx.put(insn, 0); // current insn is the first branch
      return true;
    }

    BranchExp branchExp = getPredBranchExp(predecessor);
    Integer branchIdx = insnToIdx.get(insn);
    if (branchIdx == null) {
      // Keep track of the instructions in the branches that are already added
      branchIdx = branchExp.add(exp);
      insnToIdx.put(insn, branchIdx);
    }
    // If the branch where the instruction is on is already added, no need to do anything as
    // branchExp has a reference to exp already.
    return false;
  }

  /** Finishing the method */
  @Override
  public void visitEnd() {

    for (Jump jump : jumps) {
      Instruction insn = labelToInsn.get(jump.target);
      jump.source.addBranch(insn, jump.branch);
      predecessors.put(insn, jump.source);
    }

    // Compute CovExp for every instruction.
    for (Map.Entry<Integer, Instruction> entry : probeToInsn.entrySet()) {
      int probeId = entry.getKey();
      Instruction ins = entry.getValue();

      Instruction insn = ins;
      CovExp exp = new ProbeExp(probeId);

      // Compute CovExp for the probed instruction.
      CovExp existingExp = insnToCovExp.get(insn);
      if (existingExp != null) {
        // The instruction already has a branch, add the probeExp as
        // a new branch.
        if (existingExp instanceof BranchExp) {
          BranchExp branchExp = (BranchExp) existingExp;
          branchExp.add(exp);
        } else {
          // This can only happen if the internal data is inconsistent.
          // The instruction is a predecessor of another instruction and also
          // has a probe, but the branch count is not > 1.
        }
      } else {
        if (insn.getBranchCounter().getTotalCount() > 1) {
          exp = new BranchExp(exp);
        }
        insnToCovExp.put(insn, exp);
      }

      Instruction predecessor = predecessors.get(insn);
      while (predecessor != null) {
        if (predecessor.getBranchCounter().getTotalCount() > 1) {
          boolean isNewBranch = updateBranchPredecessor(predecessor, insn, exp);
          if (!isNewBranch) {
            // If the branch already exists, no need to visit predecessors any more.
            break;
          }
        } else {
          // No branch at predecessor, use the same CovExp
          insnToCovExp.put(predecessor, exp);
        }
        insn = predecessor;
        exp = insnToCovExp.get(predecessor);
        predecessor = predecessors.get(insn);
      }
    }

    // Merge branches in the instructions on the same line
    for (Instruction insn : instructions) {
      if (insn.getBranchCounter().getTotalCount() > 1) {
        CovExp insnExp = insnToCovExp.get(insn);
        if (insnExp != null && (insnExp instanceof BranchExp)) {
          BranchExp exp = (BranchExp) insnExp;
          BranchExp lineExp = lineToBranchExp.get(insn.getLine());
          if (lineExp == null) {
            lineToBranchExp.put(insn.getLine(), exp);
          } else {
            lineExp.merge(exp);
          }
        } else {
          // If we reach here, the internal data of the mapping is inconsistent, either
          // 1) An instruction has branches but we do not create BranchExp for it.
          // 2) An instruction has branches but it does not have an associated CovExp.
        }
      }
    }
  }

  /** Jumps between instructions and labels */
  class Jump {
    public final Instruction source;
    public final Label target;
    public final int branch;

    public Jump(Instruction i, Label l, int b) {
      source = i;
      target = l;
      branch = b;
    }
  }
}
