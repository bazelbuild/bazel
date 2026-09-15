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
package proguard.configuration;


import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.Constant;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.Instruction;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.optimize.peephole.*;

import static proguard.configuration.ConfigurationLoggingInstructionSequenceConstants.LOCAL_VARIABLE_INDEX_1;
import static proguard.configuration.ConfigurationLoggingInstructionSequenceConstants.LOCAL_VARIABLE_INDEX_2;
import static proguard.configuration.ConfigurationLoggingInstructionSequenceConstants.LOCAL_VARIABLE_INDEX_3;

/**
 * This InstructionSequencesReplacer appends logging instructions to all
 * instructions calling reflection methods.
 *
 * @see InstructionSequenceReplacer
 *
 * @author Johan Leys
 */
public class ConfigurationLoggingInstructionSequenceReplacer extends InstructionSequenceReplacer
{
    public ConfigurationLoggingInstructionSequenceReplacer(InstructionSequenceMatcher instructionSequenceMatcher,
                                                           Constant[]                 patternConstants,
                                                           Instruction[]              patternInstructions,
                                                           Constant[]                 replacementConstants,
                                                           Instruction[]              replacementInstructions,
                                                           BranchTargetFinder         branchTargetFinder,
                                                           CodeAttributeEditor        codeAttributeEditor,
                                                           InstructionVisitor         extraInstructionVisitor    )
    {
        super(instructionSequenceMatcher,
              patternConstants,
              patternInstructions,
              replacementConstants,
              replacementInstructions,
              branchTargetFinder,
              codeAttributeEditor,
              extraInstructionVisitor    );
    }


    public ConfigurationLoggingInstructionSequenceReplacer(Constant[]          patternConstants,
                                                           Instruction[]       patternInstructions,
                                                           Constant[]          replacementConstants,
                                                           Instruction[]       replacementInstructions,
                                                           BranchTargetFinder  branchTargetFinder,
                                                           CodeAttributeEditor codeAttributeEditor     )
    {
        super(patternConstants,
              patternInstructions,
              replacementConstants,
              replacementInstructions,
              branchTargetFinder,
              codeAttributeEditor     );
    }


    public ConfigurationLoggingInstructionSequenceReplacer(Constant[]          patternConstants,
                                                           Instruction[]       patternInstructions,
                                                           Constant[]          replacementConstants,
                                                           Instruction[]       replacementInstructions,
                                                           BranchTargetFinder  branchTargetFinder,
                                                           CodeAttributeEditor codeAttributeEditor,
                                                           InstructionVisitor  extraInstructionVisitor )
    {
        super(patternConstants,
              patternInstructions,
              replacementConstants,
              replacementInstructions,
              branchTargetFinder,
              codeAttributeEditor,
              extraInstructionVisitor );
    }


    @Override
    protected int matchedArgument(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, int argument)
    {
        switch (argument)
        {
            case LOCAL_VARIABLE_INDEX_1:
                return codeAttribute.u2maxLocals;
            case LOCAL_VARIABLE_INDEX_2:
                return codeAttribute.u2maxLocals + 1;
            case LOCAL_VARIABLE_INDEX_3:
                return codeAttribute.u2maxLocals + 2;
            default:
                return super.matchedArgument(clazz, argument);
        }
    }


    @Override
    protected int matchedConstantIndex(ProgramClass programClass, int constantIndex)
    {
        switch (constantIndex)
        {
            case ConfigurationLoggingInstructionSequenceConstants.CLASS_NAME:
                return new ConstantPoolEditor(programClass)
                    .addStringConstant(ClassUtil.externalClassName(programClass.getName()), programClass, null);
            default:
                return super.matchedConstantIndex(programClass, constantIndex);
        }
    }
}
