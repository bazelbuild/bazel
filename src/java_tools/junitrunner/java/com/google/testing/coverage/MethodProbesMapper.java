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
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.jacoco.core.internal.analysis.filter.IFilter;
import org.jacoco.core.internal.analysis.filter.IFilterContext;
import org.jacoco.core.internal.analysis.filter.IFilterOutput;
import org.jacoco.core.internal.analysis.filter.Replacements;
import org.jacoco.core.internal.analysis.filter.Replacements.InstructionBranch;
import org.jacoco.core.internal.flow.IFrame;
import org.jacoco.core.internal.flow.LabelInfo;
import org.jacoco.core.internal.flow.MethodProbesVisitor;
import org.objectweb.asm.Handle;
import org.objectweb.asm.Label;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.MethodNode;

/**
 * The mapper is a probes visitor that will cache control flow information as well as keeping track
 * of the probes as the main driver generates the probe ids. Upon finishing the method it uses the
 * information collected to generate the mapping information between probes and the instructions.
 */
public class MethodProbesMapper extends MethodProbesVisitor implements IFilterOutput {
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
  private AbstractInsnNode currentInstructionNode = null;
  private final Map<AbstractInsnNode, Instruction> instructionMap = new HashMap<>();

  // Filtering
  private final IFilter filter;
  private final IFilterContext filterContext;
  private final HashSet<AbstractInsnNode> ignored = new HashSet<>();
  private final Map<AbstractInsnNode, AbstractInsnNode> unioned = new HashMap<>();
  private final Map<AbstractInsnNode, Replacements> branchReplacements = new HashMap<>();

  // Result
  private Map<Integer, BranchExp> lineToBranchExp = new TreeMap<>();

  public Map<Integer, BranchExp> result() {
    return lineToBranchExp;
  }

  // Intermediate results
  //
  // These values are built up during the visitor methods. They will be used to compute
  // the final results.
  private final InstructionSet instructions = new InstructionSet();
  private final List<Jump> jumps = new ArrayList<>();
  private final List<Instruction> probedInstructions = new ArrayList<>();
  private final Map<Label, Instruction> labelToInsn = new HashMap<>();

  public MethodProbesMapper(IFilterContext filterContext, IFilter filter) {
    this.filterContext = filterContext;
    this.filter = filter;
  }

  @Override
  public void accept(MethodNode methodNode, MethodVisitor methodVisitor) {
    methodVisitor.visitCode();
    for (AbstractInsnNode i : methodNode.instructions) {
      currentInstructionNode = i;
      i.accept(methodVisitor);
    }
    if (filter != null) {
      filter.filter(methodNode, filterContext, this);
    }
    methodVisitor.visitEnd();
  }

  /** Visitor method to append a new Instruction */
  private void visitInsn() {
    Instruction instruction = new Instruction(currentLine);
    instructions.add(instruction);
    if (lastInstruction != null) {
      lastInstruction.addBranch(instruction, /* branchIndex= */ 0);
    }

    for (Label label : currentLabels) {
      labelToInsn.put(label, instruction);
    }
    currentLabels.clear(); // Update states
    lastInstruction = instruction;
    instructionMap.put(currentInstructionNode, instruction);
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
        branch++;
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

  private void addProbe(int probeId, int branchIdx) {
    // We do not add probes to the flow graph, but we need to update
    // the branch count of the predecessor of the probe
    lastInstruction.addBranch(new ProbeExp(probeId), branchIdx);
    probedInstructions.add(lastInstruction);
  }

  // Probe visit methods
  @Override
  public void visitProbe(int probeId) {
    // This function is only called when visiting a merge node which
    // is a successor.
    // It adds a probe point to the last instruction
    assert (lastInstruction != null);

    addProbe(probeId, /* branchIdx= */ 0);
    lastInstruction = null; // Merge point should have no predecessor.
  }

  @Override
  public void visitJumpInsnWithProbe(int opcode, Label label, int probeId, IFrame frame) {
    visitInsn();
    addProbe(probeId, /* branchIdx= */ 1);
  }

  @Override
  public void visitInsnWithProbe(int opcode, int probeId) {
    visitInsn();
    addProbe(probeId, /* branchIdx= */ 0);
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
      branch++;
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
        addProbe(id, branch);
      }
      LabelInfo.setDone(label);
    }
  }

  /** Finishing the method */
  @Override
  public void visitEnd() {
    for (Jump jump : jumps) {
      Instruction insn = labelToInsn.get(jump.target);
      jump.source.addBranch(insn, jump.branch);
    }

    for (Instruction insn : probedInstructions) {
      Instruction.wireBranchPredecessors(insn);
    }

    // Handle merged instructions
    for (AbstractInsnNode node : unioned.keySet()) {
      AbstractInsnNode rep = findRepresentative(node);
      Instruction insn = instructionMap.get(node);
      Instruction repInsn = instructionMap.get(rep);
      BranchExp branch = BranchExp.ensureIsBranchExp(insn.branchExp);
      BranchExp repBranch = BranchExp.ensureIsBranchExp(repInsn.branchExp);
      repInsn.branchExp = BranchExp.zip(repBranch, branch);
      ignored.add(node);
    }

    // Handle branch replacements
    for (Map.Entry<AbstractInsnNode, Replacements> entry : branchReplacements.entrySet()) {
      BranchExp newBranchExp = BranchExp.initializeEmptyBranches();
      int branchIndex = 0;
      for (Collection<InstructionBranch> replacements : entry.getValue().values()) {
        BranchExp subExp = BranchExp.initializeEmptyBranches();
        int subBranchIndex = 0;
        for (InstructionBranch replacement : replacements) {
          BranchExp branchExp = instructionMap.get(replacement.instruction).branchExp;
          subExp.setBranchAtIndex(subBranchIndex, branchExp.getBranchAtIndex(replacement.branch));
          subBranchIndex++;
        }
        newBranchExp.setBranchAtIndex(branchIndex, subExp);
        branchIndex++;
      }
      Instruction oldInsn = instructionMap.get(entry.getKey());
      Instruction newInsn = new Instruction(oldInsn.line);
      newInsn.logicalBranches = branchIndex;
      newInsn.branchExp = newBranchExp;
      instructionMap.put(entry.getKey(), newInsn);
      instructions.replace(oldInsn, newInsn);
    }

    HashSet<Instruction> ignoredInstructions = new HashSet<>();
    for (Map.Entry<AbstractInsnNode, Instruction> entry : instructionMap.entrySet()) {
      if (ignored.contains(entry.getKey())) {
        ignoredInstructions.add(entry.getValue());
      }
    }

    // Merge branches in the instructions on the same line
    for (Instruction insn : instructions) {
      if (ignoredInstructions.contains(insn)) {
        continue;
      }
      if (insn.logicalBranches > 1) {
        CovExp insnExp = insn.branchExp;
        if (insnExp != null && (insnExp instanceof BranchExp)) {
          BranchExp exp = (BranchExp) insnExp;
          BranchExp lineExp = lineToBranchExp.get(insn.line);
          if (lineExp == null) {
            lineToBranchExp.put(insn.line, exp);
          } else {
            lineToBranchExp.put(insn.line, BranchExp.concatenate(lineExp, exp));
          }
        } else {
          // If we reach here, the internal data of the mapping is inconsistent, either
          // 1) An instruction has branches but we do not create BranchExp for it.
          // 2) An instruction has branches but it does not have an associated CovExp.
        }
      }
    }
  }

  /** IFilterOutput */
  // Handle only ignore for now; most filters only use this.
  @Override
  public void ignore(AbstractInsnNode fromInclusive, AbstractInsnNode toInclusive) {
    for (AbstractInsnNode n = fromInclusive; n != toInclusive; n = n.getNext()) {
      ignored.add(n);
    }
    ignored.add(toInclusive);
  }

  @Override
  public void merge(AbstractInsnNode i1, AbstractInsnNode i2) {
    // Track nodes to be merged using a union-find algorithm.
    i1 = findRepresentative(i1);
    i2 = findRepresentative(i2);
    if (i1 != i2) {
      unioned.put(i1, i2);
    }
  }

  @Override
  public void replaceBranches(AbstractInsnNode source, Replacements replacements) {
    branchReplacements.put(source, replacements);
  }

  private AbstractInsnNode findRepresentative(AbstractInsnNode node) {
    // The "find" part of union-find. Walk the chain of nodes to find the representative node
    // (at the root), flattening the tree a little as we go.
    AbstractInsnNode parent;
    AbstractInsnNode grandParent;
    while ((parent = unioned.get(node)) != null) {
      if ((grandParent = unioned.get(parent)) != null) {
        unioned.put(node, grandParent);
      }
      node = parent;
    }
    return node;
  }

  /** Jumps between instructions and labels */
  private static class Jump {
    public final Instruction source;
    public final Label target;
    public final int branch;

    Jump(Instruction i, Label l, int b) {
      source = i;
      target = l;
      branch = b;
    }
  }

  /** Associate an instruction with a CovExp and its predecessor. */
  private static class Instruction {

    final int line;

    BranchExp branchExp = BranchExp.initializeEmptyBranches();

    Instruction predecessor = null;

    int predecessorBranchIndex = -1;

    int logicalBranches = 0;

    Instruction(int line) {
      this.line = line;
    }

    void addBranch(Instruction target, int branchIndex) {
      logicalBranches++;
      target.predecessor = this;
      target.predecessorBranchIndex = branchIndex;
    }

    void addBranch(ProbeExp probeExp, int branchIndex) {
      logicalBranches++;
      branchExp.setBranchAtIndex(branchIndex, probeExp);
    }

    /** Sets the target for a given branch. */
    void setBranchTarget(CovExp targetExp, int branchIndex) {
      branchExp.setBranchAtIndex(branchIndex, targetExp);
    }

    static void wireBranchPredecessors(Instruction root) {
      // This is not a recursive method because some of these chains can be quite long
      Instruction current = root;
      Instruction predecessor = root.predecessor;
      while (predecessor != null) {
        boolean alreadyHasBranches = predecessor.branchExp.hasBranches();
        predecessor.setBranchTarget(current.branchExp, current.predecessorBranchIndex);
        if (alreadyHasBranches) {
          // if the predecessor already had a configured branchExp we don't need to continue the
          // walk; it should already have wired up its predecessors.
          break;
        }
        current = predecessor;
        predecessor = current.predecessor;
      }
    }
  }

  /**
   * Permit efficient replacement of one instruction with another while preserving original
   * insertion order. A replacement instruction takes the place of the old instruction for iteration
   * order.
   */
  private static class InstructionSet implements Iterable<Instruction> {

    private final List<Instruction> instructions = new ArrayList<>();

    private final Map<Instruction, Integer> instructionIndex = new HashMap<>();

    void add(Instruction instruction) {
      instructionIndex.put(instruction, instructions.size());
      instructions.add(instruction);
    }

    void replace(Instruction oldInstruction, Instruction newInstruction) {
      int index = instructionIndex.get(oldInstruction);
      instructions.set(index, newInstruction);
      instructionIndex.put(newInstruction, index);
      instructionIndex.remove(oldInstruction);
    }

    @Override
    public Iterator<Instruction> iterator() {
      return instructions.iterator();
    }
  }
}
