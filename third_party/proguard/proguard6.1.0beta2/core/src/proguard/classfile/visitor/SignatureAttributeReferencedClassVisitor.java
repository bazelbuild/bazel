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
package proguard.classfile.visitor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.util.SimplifiedVisitor;

/**
 * This AttributeVisitor lets a given ClassVisitor visit all the
 * classes referenced by the type descriptors of the signatures that it visits.
 *
 * @author Joachim Vandersmissen
 */
public class SignatureAttributeReferencedClassVisitor
extends      SimplifiedVisitor
implements   AttributeVisitor
{
    private final ClassVisitor classVisitor;


    public SignatureAttributeReferencedClassVisitor(ClassVisitor classVisitor)
    {
        this.classVisitor = classVisitor;
    }


    // Implementations for AttributeVisitor


    public void visitAnyAttribute(Clazz clazz, Attribute attribute)
    {

    }


    public void visitSignatureAttribute(Clazz clazz,
                                        SignatureAttribute signatureAttribute)
    {
        signatureAttribute.referencedClassesAccept(classVisitor);
    }


    public void visitSignatureAttribute(Clazz clazz,
                                        Field field,
                                        SignatureAttribute signatureAttribute)
    {
        signatureAttribute.referencedClassesAccept(classVisitor);
    }


    public void visitSignatureAttribute(Clazz clazz,
                                        Method method,
                                        SignatureAttribute signatureAttribute)
    {
        signatureAttribute.referencedClassesAccept(classVisitor);
    }
}
