/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.optimize.peephole;

import proguard.classfile.constant.Constant;
import proguard.classfile.editor.CodeAttributeEditor;
import proguard.classfile.instruction.Instruction;
import proguard.classfile.instruction.visitor.*;

/**
 * This InstructionVisitor replaces multiple instruction sequences at once.
 *
 * The replacement sequences are optional, defaulting to the empty sequence,
 * to delete the matched pattern sequences.
 *
 * @see InstructionSequenceReplacer
 * @author Eric Lafortune
 */
public class InstructionSequencesReplacer
extends      MultiInstructionVisitor
implements   InstructionVisitor
{
    private static final int PATTERN_INDEX     = 0;
    private static final int REPLACEMENT_INDEX = 1;

    private static final Instruction[] EMPTY_INSTRUCTIONS = new Instruction[0];


    /**
     * Creates a new InstructionSequencesReplacer.
     * @param constants               any constants referenced by the pattern
     *                                instructions and replacement instructions.
     * @param instructionSequences    the instruction sequences to be replaced,
     *                                with subsequently the sequence pair index,
     *                                the patten/replacement index (0 or 1),
     *                                and the instruction index in the sequence.
     * @param branchTargetFinder      a branch target finder that has been
     *                                initialized to indicate branch targets
     *                                in the visited code.
     * @param codeAttributeEditor     a code editor that can be used for
     *                                accumulating changes to the code.
     */
    public InstructionSequencesReplacer(Constant[]          constants,
                                        Instruction[][][]   instructionSequences,
                                        BranchTargetFinder  branchTargetFinder,
                                        CodeAttributeEditor codeAttributeEditor)
    {
        this(constants,
             instructionSequences,
             branchTargetFinder,
             codeAttributeEditor,
             null);
    }


    /**
     * Creates a new InstructionSequenceReplacer.
     * @param constants               any constants referenced by the pattern
     *                                instructions and replacement instructions.
     * @param instructionSequences    the instruction sequences to be replaced,
     *                                with subsequently the sequence pair index,
     *                                the patten/replacement index (0 or 1),
     *                                and the instruction index in the sequence.
     * @param branchTargetFinder      a branch target finder that has been
     *                                initialized to indicate branch targets
     *                                in the visited code.
     * @param codeAttributeEditor     a code editor that can be used for
     *                                accumulating changes to the code.
     * @param extraInstructionVisitor an optional extra visitor for all deleted
     *                                load instructions.
     */
    public InstructionSequencesReplacer(Constant[]          constants,
                                        Instruction[][][]   instructionSequences,
                                        BranchTargetFinder  branchTargetFinder,
                                        CodeAttributeEditor codeAttributeEditor,
                                        InstructionVisitor  extraInstructionVisitor)
    {
        super(createInstructionSequenceReplacers(constants,
                                                 instructionSequences,
                                                 branchTargetFinder,
                                                 codeAttributeEditor,
                                                 extraInstructionVisitor));
    }


    /**
     * Creates an array of InstructionSequenceReplacer instances.
     * @param constants               any constants referenced by the pattern
     *                                instructions and replacement instructions.
     * @param instructionSequences    the instruction sequences to be replaced,
     *                                with subsequently the sequence pair index,
     *                                the from/to index (0 or 1), and the
     *                                instruction index in the sequence.
     * @param branchTargetFinder      a branch target finder that has been
     *                                initialized to indicate branch targets
     *                                in the visited code.
     * @param codeAttributeEditor     a code editor that can be used for
     *                                accumulating changes to the code.
     * @param extraInstructionVisitor an optional extra visitor for all deleted
     *                                load instructions.
     */
    private static InstructionVisitor[] createInstructionSequenceReplacers(Constant[]          constants,
                                                                           Instruction[][][]   instructionSequences,
                                                                           BranchTargetFinder  branchTargetFinder,
                                                                           CodeAttributeEditor codeAttributeEditor,
                                                                           InstructionVisitor  extraInstructionVisitor)
    {
        InstructionVisitor[] instructionSequenceReplacers =
            new InstructionVisitor[instructionSequences.length];

        for (int index = 0; index < instructionSequenceReplacers.length; index++)
        {
            Instruction[][] instructionSequencePair =
                instructionSequences[index];

            Instruction[] patternInstructions =
                instructionSequencePair[PATTERN_INDEX];

            // The replacement sequence is optional.
            Instruction[] replacementInstructions =
                instructionSequencePair.length > REPLACEMENT_INDEX ?
                    instructionSequencePair[REPLACEMENT_INDEX] :
                    EMPTY_INSTRUCTIONS;

            instructionSequenceReplacers[index] =
                new InstructionSequenceReplacer(constants,
                                                patternInstructions,
                                                constants,
                                                replacementInstructions,
                                                branchTargetFinder,
                                                codeAttributeEditor,
                                                extraInstructionVisitor);
        }

        return instructionSequenceReplacers;
    }
}
