/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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

import proguard.classfile.ProgramClass;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;
import proguard.optimize.KeepMarker;

/**
 * This ClassVisitor attaches a ClassOptimizationInfo instance to every class
 * that is not being kept that it visits.
 *
 * @author Eric Lafortune
 */
public class ClassOptimizationInfoSetter
extends      SimplifiedVisitor
implements   ClassVisitor
{
    // Implementations for MemberVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        if (!KeepMarker.isKept(programClass))
        {
            ClassOptimizationInfo.setClassOptimizationInfo(programClass);
        }
    }
}