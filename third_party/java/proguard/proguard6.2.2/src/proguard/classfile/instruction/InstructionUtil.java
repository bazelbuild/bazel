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
package proguard.classfile.instruction;

import proguard.classfile.ClassConstants;

import static proguard.classfile.ClassConstants.*;
import static proguard.classfile.instruction.InstructionConstants.*;

/**
 * Utility methods for converting between representations of names and
 * descriptions.
 *
 * @author Eric Lafortune
 */
public class InstructionUtil
{
    /**
     * Returns the internal type corresponding to the given 'newarray' type.
     * @param arrayType <code>InstructionConstants.ARRAY_T_BOOLEAN</code>,
     *                  <code>InstructionConstants.ARRAY_T_BYTE</code>,
     *                  <code>InstructionConstants.ARRAY_T_CHAR</code>,
     *                  <code>InstructionConstants.ARRAY_T_SHORT</code>,
     *                  <code>InstructionConstants.ARRAY_T_INT</code>,
     *                  <code>InstructionConstants.ARRAY_T_LONG</code>,
     *                  <code>InstructionConstants.ARRAY_T_FLOAT</code>, or
     *                  <code>InstructionConstants.ARRAY_T_DOUBLE</code>.
     * @return <code>ClassConstants.TYPE_BOOLEAN</code>,
     *         <code>ClassConstants.TYPE_BYTE</code>,
     *         <code>ClassConstants.TYPE_CHAR</code>,
     *         <code>ClassConstants.TYPE_SHORT</code>,
     *         <code>ClassConstants.TYPE_INT</code>,
     *         <code>ClassConstants.TYPE_LONG</code>,
     *         <code>ClassConstants.TYPE_FLOAT</code>, or
     *         <code>ClassConstants.TYPE_DOUBLE</code>.
     */
    public static char internalTypeFromArrayType(byte arrayType)
    {
        switch (arrayType)
        {
            case InstructionConstants.ARRAY_T_BOOLEAN: return ClassConstants.TYPE_BOOLEAN;
            case InstructionConstants.ARRAY_T_CHAR:    return ClassConstants.TYPE_CHAR;
            case InstructionConstants.ARRAY_T_FLOAT:   return ClassConstants.TYPE_FLOAT;
            case InstructionConstants.ARRAY_T_DOUBLE:  return ClassConstants.TYPE_DOUBLE;
            case InstructionConstants.ARRAY_T_BYTE:    return ClassConstants.TYPE_BYTE;
            case InstructionConstants.ARRAY_T_SHORT:   return ClassConstants.TYPE_SHORT;
            case InstructionConstants.ARRAY_T_INT:     return ClassConstants.TYPE_INT;
            case InstructionConstants.ARRAY_T_LONG:    return ClassConstants.TYPE_LONG;
            default: throw new IllegalArgumentException("Unknown array type ["+arrayType+"]");
        }
    }

    /**
     * Returns the newarray type constant for the given internal primitive
     * type.
     *
     * @param internalType a primitive type ('Z','B','I',...)
     * @return the array type constant corresponding to the given
     *         primitive type.
     * @see #internalTypeFromArrayType(byte)
     */
    public static byte arrayTypeFromInternalType(char internalType)
    {
        switch (internalType)
        {
            case TYPE_BOOLEAN: return ARRAY_T_BOOLEAN;
            case TYPE_BYTE:    return ARRAY_T_BYTE;
            case TYPE_CHAR:    return ARRAY_T_CHAR;
            case TYPE_SHORT:   return ARRAY_T_SHORT;
            case TYPE_INT:     return ARRAY_T_INT;
            case TYPE_LONG:    return ARRAY_T_LONG;
            case TYPE_FLOAT:   return ARRAY_T_FLOAT;
            case TYPE_DOUBLE:  return ARRAY_T_DOUBLE;
            default:           throw new IllegalArgumentException("Unknown primitive: " + internalType);
        }
    }
}
