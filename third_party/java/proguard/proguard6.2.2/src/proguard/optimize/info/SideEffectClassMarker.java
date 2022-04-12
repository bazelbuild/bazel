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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

/**
 * This ClassVisitor marks all classes that it visits as having side effects.
 *
 * @author Eric Lafortune
 */
public class SideEffectClassMarker
extends      SimplifiedVisitor
implements   ClassVisitor
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        markSideEffects(programClass);
    }


    // Small utility methods.

    private static void markSideEffects(Clazz clazz)
    {
        ProgramClassOptimizationInfo.getProgramClassOptimizationInfo(clazz).setSideEffects();
    }


    public static boolean hasSideEffects(Clazz clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(clazz).hasSideEffects();
    }
}