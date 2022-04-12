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

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;
import proguard.optimize.KeepMarker;

/**
 * This <code>ClassVisitor</code> makes the program classes that it visits
 * final, if possible.
 *
 * @author Eric Lafortune
 */
public class ClassFinalizer
extends      SimplifiedVisitor
implements   ClassVisitor
{
    private final ClassVisitor extraClassVisitor;


    /**
     * Creates a new ClassFinalizer.
     */
    public ClassFinalizer()
    {
        this(null);
    }


    /**
     * Creates a new ClassFinalizer.
     * @param extraClassVisitor an optional extra visitor for all finalized
     *                          classes.
     */
    public ClassFinalizer(ClassVisitor  extraClassVisitor)
    {
        this.extraClassVisitor = extraClassVisitor;
    }


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // If the class is not final/interface/abstract,
        // and it is not being kept,
        // and it doesn't have any subclasses,
        // then make it final.
        if ((programClass.u2accessFlags & (ClassConstants.ACC_FINAL     |
                                           ClassConstants.ACC_INTERFACE |
                                           ClassConstants.ACC_ABSTRACT)) == 0 &&
            !KeepMarker.isKept(programClass)                                           &&
            programClass.subClasses == null)
        {
            programClass.u2accessFlags |= ClassConstants.ACC_FINAL;

            // Visit the class, if required.
            if (extraClassVisitor != null)
            {
                extraClassVisitor.visitProgramClass(programClass);
            }
        }
    }
}
