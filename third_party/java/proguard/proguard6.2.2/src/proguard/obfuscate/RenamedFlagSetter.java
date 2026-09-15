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
package proguard.obfuscate;

import proguard.classfile.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;


/**
 * This ClassVisitor sets the ACC_RENAMED flag for classes or class members
 * that have been renamed.
 *
 * @author Johan Leys
 */
public class RenamedFlagSetter
extends      SimplifiedVisitor
implements   ClassVisitor,

             // Implementation interfaces.
             MemberVisitor,
             AttributeVisitor
{
    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        String oldName = programClass.getName();
        String newName = ClassObfuscator.newClassName(programClass);

        if (newName != null && !oldName.equals(newName))
        {
            programClass.u2accessFlags |= ClassConstants.ACC_RENAMED;
        }

        // Print out the class members.
        programClass.fieldsAccept(this);
        programClass.methodsAccept(this);
    }


    // Implementations for MemberVisitor.

    public void visitProgramMember(ProgramClass programClass, ProgramMember programMember)
    {
        String oldName = programMember.getName(programClass);
        String newName = MemberObfuscator.newMemberName(programMember);

        if (newName != null && !newName.equals(oldName))
        {
            programMember.u2accessFlags |= ClassConstants.ACC_RENAMED;
        }
    }
}
