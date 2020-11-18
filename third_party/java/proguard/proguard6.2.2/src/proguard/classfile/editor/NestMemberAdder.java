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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.ClassVisitor;
import proguard.util.ArrayUtil;

/**
 * This ConstantVisitor and ClassVisitor adds the class constants or the
 * classes that it visits to the given target nest member attribute.
 */
public class NestMemberAdder
extends      SimplifiedVisitor
implements   ConstantVisitor,
             ClassVisitor

{
    private final ConstantPoolEditor   constantPoolEditor;
    private final NestMembersAttribute targetNestMembersAttribute;


    /**
     * Creates a new NestMemberAdder that will add classes to the
     * given target nest members attribute.
     */
    public NestMemberAdder(ProgramClass         targetClass,
                           NestMembersAttribute targetNestMembersAttribute)
    {
        this.constantPoolEditor         = new ConstantPoolEditor(targetClass);
        this.targetNestMembersAttribute = targetNestMembersAttribute;
    }


    // Implementations for ConstantVisitor.

    public void visitAnyConstant(Clazz clazz, Constant constant) {}

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        targetNestMembersAttribute.u2classes =
            ArrayUtil.add(targetNestMembersAttribute.u2classes,
                          targetNestMembersAttribute.u2classesCount++,
                          constantPoolEditor.addClassConstant(classConstant.getName(clazz),
                                                              classConstant.referencedClass));
    }


    // Implementations for ClassVisitor.

    public void visitAnyClass(Clazz clazz)
    {
        targetNestMembersAttribute.u2classes =
            ArrayUtil.add(targetNestMembersAttribute.u2classes,
                          targetNestMembersAttribute.u2classesCount++,
                          constantPoolEditor.addClassConstant(clazz));
    }
}
