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
package proguard.classfile.attribute.module.visitor;

import proguard.classfile.Clazz;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.module.ModuleAttribute;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AttributeVisitor lets a given ExportsInfoVisitor visit all
 * ExportsInfo objects of the ModuleAttribute objects it visits.
 *
 * @author Joachim Vandersmissen
 */
public class AllExportsInfoVisitor
extends SimplifiedVisitor
    implements AttributeVisitor
{
    private final ExportsInfoVisitor exportsInfoVisitor;


    public AllExportsInfoVisitor(ExportsInfoVisitor exportsInfoVisitor)
    {
        this.exportsInfoVisitor = exportsInfoVisitor;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitModuleAttribute(Clazz clazz, ModuleAttribute moduleAttribute)
    {
        moduleAttribute.exportsAccept(clazz, exportsInfoVisitor);
    }
}
