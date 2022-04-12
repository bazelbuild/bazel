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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.editor.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.MemberVisitor;

/**
 * This AttributeVisitor optimizes variable allocation based on their the liveness,
 * in the code attributes that it visits.
 *
 * @author Eric Lafortune
 */
public class VariableOptimizer
extends      SimplifiedVisitor
implements   AttributeVisitor,
             LocalVariableInfoVisitor,
             LocalVariableTypeInfoVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = true;
    //*/

    private static final int MAX_VARIABLES_SIZE = 64;


    private final boolean       reuseThis;
    private final MemberVisitor extraVariableMemberVisitor;

    private final LivenessAnalyzer livenessAnalyzer = new LivenessAnalyzer();
    private final VariableRemapper variableRemapper = new VariableRemapper();
    private       VariableCleaner  variableCleaner  = new VariableCleaner();

    private int[] variableMap = new int[ClassConstants.TYPICAL_VARIABLES_SIZE];


    /**
     * Creates a new VariableOptimizer.
     * @param reuseThis specifies whether the 'this' variable can be reused.
     *                  Many JVMs for JME and IBM's JVMs for JSE can't handle
     *                  such reuse.
     */
    public VariableOptimizer(boolean reuseThis)
    {
        this(reuseThis, null);
    }


    /**
     * Creates a new VariableOptimizer with an extra visitor.
     * @param reuseThis                  specifies whether the 'this' variable
     *                                   can be reused. Many JVMs for JME and
     *                                   IBM's JVMs for JSE can't handle such
     *                                   reuse.
     * @param extraVariableMemberVisitor an optional extra visitor for all
     *                                   removed variables.
     */
    public VariableOptimizer(boolean       reuseThis,
                             MemberVisitor extraVariableMemberVisitor)
    {
        this.reuseThis                  = reuseThis;
        this.extraVariableMemberVisitor = extraVariableMemberVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
//        DEBUG =
//            clazz.getName().equals("abc/Def") &&
//            method.getName(clazz).equals("abc");

        // Initialize the global arrays.
        initializeArrays(codeAttribute);

        // Analyze the liveness of the variables in the code.
        livenessAnalyzer.visitCodeAttribute(clazz, method, codeAttribute);

        // Trim the variables in the local variable tables, because even
        // clipping the tables individually may leave some inconsistencies
        // between them.
        codeAttribute.attributesAccept(clazz, method, this);

        int startIndex =
            (method.getAccessFlags() & ClassConstants.ACC_STATIC) != 0 ||
            reuseThis ? 0 : 1;

        int parameterSize =
            ClassUtil.internalMethodParameterSize(method.getDescriptor(clazz),
                                                  method.getAccessFlags());

        int variableSize = codeAttribute.u2maxLocals;
        int codeLength   = codeAttribute.u4codeLength;

        boolean remapping = false;

        // Loop over all variables.
        for (int oldIndex = 0; oldIndex < variableSize; oldIndex++)
        {
            // By default, the variable will be mapped onto itself.
            variableMap[oldIndex] = oldIndex;

            // Only try remapping the variable if it's not a parameter.
            if (oldIndex >= parameterSize &&
                oldIndex < MAX_VARIABLES_SIZE)
            {
                // Try to remap the variable to a variable with a smaller index.
                for (int newIndex = startIndex; newIndex < oldIndex; newIndex++)
                {
                    if (areNonOverlapping(oldIndex, newIndex, codeLength))
                    {
                        variableMap[oldIndex] = newIndex;

                        updateLiveness(oldIndex, newIndex, codeLength);

                        remapping = true;

                        // This variable has been remapped. Go to the next one.
                        break;
                    }
                }
            }
        }

        // Have we been able to remap any variables?
        if (remapping)
        {
            if (DEBUG)
            {
                System.out.println("VariableOptimizer: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
                for (int index= 0; index < variableSize; index++)
                {
                    System.out.println("  v"+index+" -> "+variableMap[index]);
                }
            }

            // Remap the variables.
            variableRemapper.setVariableMap(variableMap);
            variableRemapper.visitCodeAttribute(clazz, method, codeAttribute);

            // Visit the method, if required.
            if (extraVariableMemberVisitor != null)
            {
                method.accept(clazz, extraVariableMemberVisitor);
            }
        }
        else
        {
            // Just clean up any empty variables.
            variableCleaner.visitCodeAttribute(clazz, method, codeAttribute);
        }
    }


    public void visitLocalVariableTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTableAttribute localVariableTableAttribute)
    {
        // Trim the variables in the local variable table.
        localVariableTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    public void visitLocalVariableTypeTableAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeTableAttribute localVariableTypeTableAttribute)
    {
        // Trim the variables in the local variable type table.
        localVariableTypeTableAttribute.localVariablesAccept(clazz, method, codeAttribute, this);
    }


    // Implementations for LocalVariableInfoVisitor.

    public void visitLocalVariableInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableInfo localVariableInfo)
    {
        // Trim the local variable to the instructions at which it is alive.
        int variable = localVariableInfo.u2index;
        int startPC  = localVariableInfo.u2startPC;
        int endPC    = startPC + localVariableInfo.u2length;

        startPC = firstLiveness(startPC, endPC, variable);
        endPC   = lastLiveness(startPC, endPC, variable);

        // Leave the start address of unused variables unchanged.
        int length = endPC - startPC;
        if (length > 0)
        {
            localVariableInfo.u2startPC = startPC;
        }

        localVariableInfo.u2length  = length;
    }


    // Implementations for LocalVariableTypeInfoVisitor.

    public void visitLocalVariableTypeInfo(Clazz clazz, Method method, CodeAttribute codeAttribute, LocalVariableTypeInfo localVariableTypeInfo)
    {
        // Trim the local variable type to the instructions at which it is alive.
        int variable = localVariableTypeInfo.u2index;
        int startPC  = localVariableTypeInfo.u2startPC;
        int endPC    = startPC + localVariableTypeInfo.u2length;

        startPC = firstLiveness(startPC, endPC, variable);
        endPC   = lastLiveness(startPC, endPC, variable);

        // Leave the start address of unused variables unchanged.
        int length = endPC - startPC;
        if (length > 0)
        {
            localVariableTypeInfo.u2startPC = startPC;
        }

        localVariableTypeInfo.u2length  = length;
    }


    // Small utility methods.

    /**
     * Initializes the global arrays.
     */
    private void initializeArrays(CodeAttribute codeAttribute)
    {
        int codeLength = codeAttribute.u4codeLength;

        // Create new arrays for storing information at each instruction offset.
        if (variableMap.length < codeLength)
        {
            variableMap = new int[codeLength];
        }
    }


    /**
     * Returns whether the given variables are never alive at the same time.
     */
    private boolean areNonOverlapping(int variableIndex1,
                                      int variableIndex2,
                                      int codeLength)
    {
        // Loop over all instructions.
        for (int offset = 0; offset < codeLength; offset++)
        {
            if ((livenessAnalyzer.isAliveBefore(offset, variableIndex1) &&
                 livenessAnalyzer.isAliveBefore(offset, variableIndex2)) ||

                (livenessAnalyzer.isAliveAfter(offset, variableIndex1) &&
                 livenessAnalyzer.isAliveAfter(offset, variableIndex2)) ||

                // For now, exclude Category 2 variables.
                livenessAnalyzer.isCategory2(offset, variableIndex1))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Updates the liveness resulting from mapping the given old variable on
     * the given new variable.
     */
    private void updateLiveness(int oldVariableIndex,
                                int newVariableIndex,
                                int codeLength)
    {
        // Loop over all instructions.
        for (int offset = 0; offset < codeLength; offset++)
        {
            // Update the liveness before the instruction.
            if (livenessAnalyzer.isAliveBefore(offset, oldVariableIndex))
            {
                livenessAnalyzer.setAliveBefore(offset, oldVariableIndex, false);
                livenessAnalyzer.setAliveBefore(offset, newVariableIndex, true);
            }

            // Update the liveness after the instruction.
            if (livenessAnalyzer.isAliveAfter(offset, oldVariableIndex))
            {
                livenessAnalyzer.setAliveAfter(offset, oldVariableIndex, false);
                livenessAnalyzer.setAliveAfter(offset, newVariableIndex, true);
            }
        }
    }


    /**
     * Returns the first instruction offset between the given offsets at which
     * the given variable goes alive.
     */
    private int firstLiveness(int startOffset, int endOffset, int variableIndex)
    {
        for (int offset = startOffset; offset < endOffset; offset++)
        {
            if (livenessAnalyzer.isTraced(offset) &&
                livenessAnalyzer.isAliveBefore(offset, variableIndex))
            {
                return offset;
            }
        }

        return endOffset;
    }


    /**
     * Returns the last instruction offset between the given offsets before
     * which the given variable is still alive.
     */
    private int lastLiveness(int startOffset, int endOffset, int variableIndex)
    {
        int previousOffset = endOffset;

        for (int offset = endOffset-1; offset >= startOffset; offset--)
        {
            if (livenessAnalyzer.isTraced(offset))
            {
                if (livenessAnalyzer.isAliveBefore(offset, variableIndex))
                {
                    return previousOffset;
                }

                previousOffset = offset;
            }
        }

        return endOffset;
    }
}
